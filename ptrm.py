"""PTRM inference: K noisy rollouts of a trained TRM, then selection.

python ptrm.py --checkpoint <run_dir>/step_<N> --data-paths <dataset_dir> --K 100 --sigma 0.2
"""
import argparse
import json
import os

import numpy as np
import torch
import yaml

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
from utils.functions import load_model_class

IGNORE_LABEL_ID = -100


def load_model(checkpoint, config_path, metadata, batch_size, compile_model):
    with open(config_path) as f:
        arch = dict(yaml.safe_load(f)["arch"])
    name, loss = arch.pop("name"), arch.pop("loss")
    loss_name = loss.pop("name")
    model_cfg = dict(arch, batch_size=batch_size, vocab_size=metadata.vocab_size, seq_len=metadata.seq_len,
                     num_puzzle_identifiers=metadata.num_puzzle_identifiers, causal=False)
    with torch.device("cuda"):
        model = load_model_class(loss_name)(load_model_class(name)(model_cfg), **loss)
    state = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()}, assign=True)
    model.eval()
    return (torch.compile(model) if compile_model else model), model


def noise_hook(module, args):
    """Add noise to z_H and z_L at every supervision step, with one generator per rollout."""
    sigma, gens = module.noise_sigma, module.noise_gens
    if sigma <= 0:
        return
    carry = args[0]
    n = carry.z_H.shape[0] // len(gens)
    for name in ("z_H", "z_L"): # TODO: only add noise to one
        z = getattr(carry, name)
        eps = torch.cat([torch.randn((n, *z.shape[1:]), generator=g, dtype=z.dtype, device=z.device) for g in gens])
        setattr(carry, name, z + sigma * eps)


@torch.inference_mode()
def rollouts(model, raw, batch, K, D, sigma, seed, group):
    """Per-step Q values and correctness [D, K, B], and final predictions [K, B, L]."""
    inner = raw.model.inner
    raw.model.config.halt_max_steps = D
    batch = {k: v.cuda() for k, v in batch.items()}
    labels = batch["labels"]
    qs, oks, preds = [], [], []
    for start in range(0, K, group):
        g = min(group, K - start)
        rep = {k: v.repeat((g,) + (1,) * (v.ndim - 1)) for k, v in batch.items()}
        inner.noise_sigma = sigma
        inner.noise_gens = [torch.Generator(device="cuda").manual_seed(seed + start + j) for j in range(g)]
        with torch.device("cuda"):
            carry = raw.initial_carry(rep)
        q, ok = [], []
        for _ in range(D):
            carry, _, _, out, _ = model(carry=carry, batch=rep, return_keys=["preds", "q_halt_logits"])
            p = out["preds"].view(g, *labels.shape)
            ok.append(((p == labels) | (labels == IGNORE_LABEL_ID)).all(-1))
            q.append(torch.sigmoid(out["q_halt_logits"].float()).view(g, -1))
        qs.append(torch.stack(q)), oks.append(torch.stack(ok)), preds.append(p)
    return torch.cat(qs, 1).cpu().numpy(), torch.cat(oks, 1).cpu().numpy(), torch.cat(preds, 0).cpu().numpy()


def mode_correct(preds, ok):
    """Plurality vote over whole answers, ties to the earliest rollout."""
    K, B = ok.shape
    out = np.zeros(B, bool)
    for b in range(B):
        counts = {}
        for j in range(K):
            counts.setdefault(preds[j, b].tobytes(), [0, j])[0] += 1
        out[b] = ok[max(counts.values(), key=lambda c: (c[0], -c[1]))[1], b]
    return out


def halting(q, ok, tau, b):
    """Q-based halting: rollouts run in groups of b, and a group stops at the first step where a rollout has q > tau."""
    D, K, B = q.shape
    good, steps = np.zeros(B, bool), np.zeros(B)
    for p in range(B):
        for start in range(0, K, b):
            hit = np.argwhere(q[:, start:start + b, p] > tau)
            n = min(b, K - start)
            if len(hit):
                t = hit[:, 0].min()
                good[p] = ok[t, start + q[t, start:start + b, p].argmax(), p]
                steps[p] += (t + 1) * n
                break
            steps[p] += D * n
        else:
            good[p] = ok[-1, q[-1, :, p].argmax(), p]
    return good, steps


def evaluate(model, raw, data, args, D, group):
    acc = {"best-Q@K": [], "mode@K": [], "pass@K": []}
    halt = {(t, b): ([], []) for t in args.tau for b in args.b}
    for _, batch, _ in data:
        q, ok, preds = rollouts(model, raw, batch, args.K, D, args.sigma, args.seed, group)
        valid = (batch["labels"].numpy() != IGNORE_LABEL_ID).any(-1)
        final_ok = ok[-1]
        acc["best-Q@K"].append(final_ok[q[-1].argmax(0), np.arange(final_ok.shape[1])][valid])
        acc["mode@K"].append(mode_correct(preds, final_ok)[valid])
        acc["pass@K"].append(final_ok.any(0)[valid])
        for (t, b), (goods, steps) in halt.items():
            g, s = halting(q, ok, t, b)
            goods.append(g[valid]), steps.append(s[valid])
    res = {k: float(np.concatenate(v).mean() * 100) for k, v in acc.items()}
    for (t, b), (goods, steps) in halt.items():
        res[f"halting tau={t} b={b}"] = float(np.concatenate(goods).mean() * 100)
        res[f"halting tau={t} b={b} steps"] = float(np.concatenate(steps).mean())
    res["n"] = int(sum(len(v) for v in acc["pass@K"]))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="training config (default: all_config.yaml next to the checkpoint)")
    ap.add_argument("--data-paths", nargs="+", required=True, help="dataset folders (test split)")
    ap.add_argument("--K", type=int, default=100, help="number of rollouts")
    ap.add_argument("--D", type=int, default=None, help="supervision steps per rollout (default: the checkpoint's)")
    ap.add_argument("--sigma", type=float, default=0.2, help="noise scale (0: deterministic TRM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, nargs="*", default=[], help="Q-based halting thresholds (off by default)")
    ap.add_argument("--b", type=int, nargs="*", default=[1, 16], help="Q-based halting group sizes")
    ap.add_argument("--batch-size", type=int, default=256, help="puzzles per batch")
    ap.add_argument("--rollouts-per-pass", type=int, default=None, help="rollouts run together (default: K)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--output", default=None, help="save the metrics to this JSON file")
    args = ap.parse_args()

    config = args.config or os.path.join(os.path.dirname(args.checkpoint), "all_config.yaml")
    with open(config) as f:
        D = args.D or yaml.safe_load(f)["arch"]["halt_max_steps"]
    group = args.rollouts_per_pass or args.K
    results, model = {}, None
    for path in args.data_paths:
        data = PuzzleDataset(PuzzleDatasetConfig(seed=args.seed, dataset_paths=[path], global_batch_size=args.batch_size,
                                                 test_set_mode=True, epochs_per_iter=1, rank=0, num_replicas=1), split="test")
        if model is None:
            model, raw = load_model(args.checkpoint, config, data.metadata, args.batch_size * group, args.compile)
            raw.model.inner.register_forward_pre_hook(noise_hook)
        results[path] = evaluate(model, raw, data, args, D, group)
        print(f"{path} (n={results[path]['n']}): " + ", ".join(f"{k} {v:.1f}" for k, v in results[path].items() if k != "n"), flush=True)
    if len(results) > 1:
        n = sum(r["n"] for r in results.values())
        results["all"] = {k: sum(r[k] * r["n"] for r in results.values()) / n for k in next(iter(results.values())) if k != "n"}
        results["all"]["n"] = n
        print("all: " + ", ".join(f"{k} {v:.1f}" for k, v in results["all"].items() if k != "n"))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(dict(args=vars(args), results=results), f, indent=1)


if __name__ == "__main__":
    main()
