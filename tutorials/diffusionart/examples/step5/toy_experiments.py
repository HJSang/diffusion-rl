"""Intuition experiments on the CPU tensor fixture, driven through the repo's own kernels.

Toy only: a 4-D 'latent', an MLP velocity field, reward -(x0 - c)^2. Not a paper result.
"""
import sys

import torch
from visual_rl.algorithms import art_pairwise_covariance, art_scalar_covariance
from visual_rl.backends.tensor_fixture import TensorPolicy
from visual_rl.sampling import shifted_schedule
from visual_rl.training import objectives
from visual_rl.training.objectives import ARTObjective, disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step

STEPS, LR, ROLLOUTS = 6, 3e-3, 40


def run(seed, *, P=4, K=4, updates=2, mask=True, pairwise=False, direction_mode="historical",
        scale_mode="unit_normalized_covariance", rollouts=ROLLOUTS):
    torch.manual_seed(seed)
    net = TensorPolicy()
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    groups = torch.arange(P).repeat_interleave(K)
    cond = groups.float() / P
    sigmas = shifted_schedule(STEPS, 1.0)
    predict = lambda s, t, rows: net(s, t, cond[rows])
    reward = lambda x0: -(x0 - cond[:, None]).square().mean(1)
    gen = torch.Generator().manual_seed(1000 + seed)
    eval_noise = torch.randn(P * K, 4, generator=torch.Generator().manual_seed(7))
    seen, masked_fracs, drift = [], [], []
    orig = objectives.art_loss
    objectives.art_loss = lambda *a, **k: seen.append(orig(*a, **k)) or seen[-1]
    try:
        for it in range(rollouts):
            cache = collect_rollout(predict, torch.randn(P * K, 4, generator=gen), sigmas,
                                    selected=tuple(range(STEPS)), microbatch=64)
            R = reward(cache.endpoints)
            if pairwise:
                st = art_pairwise_covariance(cache.endpoints, R, K, seed=seed, round_index=it)
                mags, anchors = st["row_guidance_magnitudes"], st["row_restore_anchor"]
            else:
                st = art_scalar_covariance(cache.endpoints, R, groups, P, direction_mode=direction_mode,
                                           scale_mode=scale_mode, zero_signal_mode="exact")
                mags, anchors = st["group_guidance_magnitudes"][groups], torch.zeros(P * K, dtype=torch.bool)
            obj = ARTObjective(cache, predict, st["group_directions"][groups], mags, anchors,
                               torch.device("cpu"), options={"corrective_mask_enabled": mask})
            for w, items in enumerate(disjoint_windows(P * K, STEPS, updates=updates)):
                seen.clear()
                accumulated_step(net, opt, items, obj, microbatch=64, denominator=len(items),
                                 max_grad_norm=1.0)
                if w == 1:
                    masked_fracs.append(torch.cat([o["masked"] for o in seen]).float().mean().item())
                    drift.append(torch.cat([o["movement_score"] for o in seen]).mean().item())
        with torch.no_grad():
            final = collect_rollout(predict, eval_noise, sigmas, selected=(0,), microbatch=64)
        return reward(final.endpoints).mean().item(), masked_fracs, drift
    finally:
        objectives.art_loss = orig


def summary(label, seeds=range(5), **kw):
    finals, masks, drifts = [], [], []
    for s in seeds:
        f, m, d = run(s, **kw)
        finals.append(f); masks += m; drifts += d
    f = torch.tensor(finals)
    extra = (f"  mask frac (upd 2) {sum(masks) / len(masks):.3f}  drift m(s) {sum(drifts) / len(drifts):.2e}"
             if masks else "")
    print(f"{label:<44} eval reward {f.mean():+.4f} ± {f.std():.4f}{extra}")


if __name__ == "__main__":
    print(f"toy: steps={STEPS} lr={LR} rollouts={ROLLOUTS}, 5 seeds; higher reward is better (0 is perfect)")
    summary("no training (rollouts=0)", rollouts=0)
    summary("ART  P4 K4, two updates, mask on (default)")
    summary("ART  P4 K4, ONE update per rollout", updates=1)
    summary("ART  P4 K4, two updates, mask OFF", mask=False)
    summary("ART  P4 K2", K=2)
    summary("ART  P4 K8", K=8)
    summary("ART  P8 K2 (same 16 samples, more prompts)", P=8, K=2)
    summary("ART-Pairwise P4 K4", pairwise=True)
    summary("ART-Pairwise P4 K2", K=2, pairwise=True)
    summary("ART  normalized_aggregate direction", direction_mode="normalized_aggregate")
    summary("ART  raw_covariance scale (no unit norm)", scale_mode="raw_covariance")
