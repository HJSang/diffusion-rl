"""Re-trace the first ART iteration of configs/examples/cpu_smoke.yaml with prints at each stage."""
import random

import numpy as np
import torch
from visual_rl.algorithms import art_scalar_covariance
from visual_rl.backends.tensor_fixture import TensorPolicy
from visual_rl.sampling import shifted_schedule
from visual_rl.training import objectives
from visual_rl.training.objectives import ARTObjective, disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step

torch.manual_seed(42); random.seed(42); np.random.seed(42)          # same order as _run_owned
model = TensorPolicy().eval()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
gen = torch.Generator().manual_seed(42)
p, k, steps = 2, 4, 4
groups = torch.arange(p).repeat_interleave(k); condition = groups.float() / p
sigmas = shifted_schedule(steps, 1.0); selected = tuple(range(steps - 1))
predict = lambda s, t, rows: model(s, t, condition[rows])
print("sigmas", sigmas.tolist(), "selected", selected)
noise = torch.randn((p * k, 4), generator=gen)
cache = collect_rollout(predict, noise, sigmas, selected=selected, stochastic=(), innovations={}, microbatch=4)
print("cache.states", tuple(cache.states.shape), "old_velocities", tuple(cache.old_velocities.shape),
      "endpoints", tuple(cache.endpoints.shape), "old_log_probs", cache.old_log_probs)
rewards = -(cache.endpoints - condition[:, None]).square().mean(1)
print("rewards", [round(r, 4) for r in rewards.tolist()])
st = art_scalar_covariance(cache.endpoints, rewards, groups, p)
for key in ("group_reward_means", "group_reward_abs_means", "group_covariance_rms", "group_guidance_magnitudes", "group_active"):
    print(f"{key:<26}", st[key].tolist())
print("group_directions", [[round(v, 4) for v in row] for row in st["group_directions"].tolist()])
seen = []
orig = objectives.art_loss
objectives.art_loss = lambda *a, **kw: seen.append(orig(*a, **kw)) or seen[-1]
obj = ARTObjective(cache, predict, st["group_directions"][groups], st["group_guidance_magnitudes"][groups],
                   torch.zeros(p * k, dtype=torch.bool), torch.device("cpu"))
windows = disjoint_windows(p * k, len(selected))
print("window sizes", [len(w) for w in windows], "first items of window 2", windows[1][:4])
for u, items in enumerate(windows, 1):
    seen.clear()
    m = accumulated_step(model, opt, items, obj, microbatch=3, denominator=len(items), max_grad_norm=1.0)
    keep = torch.cat([o["keep_mask"] for o in seen]); score = torch.cat([o["movement_score"] for o in seen])
    align = torch.cat([o["outward_alignment"] for o in seen])
    print(f"update {u}: train/loss={m['train/loss']:.10f} grad_norm={m['train/grad_norm/before_clip']:.4f} "
          f"microbatch calls={len(seen)} masked={int((~keep).sum())}/{len(keep)} "
          f"max m(s)={score.max():.2e} outward rows={int((align > 0).sum())}")
