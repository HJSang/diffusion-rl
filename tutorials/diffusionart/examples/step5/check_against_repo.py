"""Run the minimal re-implementation and the repo kernels side by side on identical inputs."""
import copy
import sys

sys.path.insert(0, "/Users/hejian.sang/Downloads/DiffusionART-paper/tutorial/scratch/step5")
import torch
import art_minimal as mini
from visual_rl.algorithms import art_scalar_covariance
from visual_rl.backends.tensor_fixture import TensorPolicy
from visual_rl.sampling import shifted_schedule
from visual_rl.training import objectives
from visual_rl.training.objectives import ARTObjective, disjoint_windows
from visual_rl.training.rollout import collect_rollout
from visual_rl.training.update import accumulated_step

torch.manual_seed(0)
P, K, STEPS, LR = 3, 4, 6, float(sys.argv[1]) if len(sys.argv) > 1 else 1e-2
groups = torch.arange(P).repeat_interleave(K)
cond = groups.float() / P
sigmas = shifted_schedule(STEPS, 1.0)                    # 1, 5/6, ..., 0
noise = torch.randn(P * K, 4)
reward = lambda x0: -(x0 - cond[:, None]).square().mean(1)  # fixture's quadratic reward

net_a = TensorPolicy()
net_b = copy.deepcopy(net_a)
opt = lambda n: torch.optim.AdamW(n.parameters(), lr=LR, weight_decay=0.0)
opt_a, opt_b = opt(net_a), opt(net_b)

# ---- minimal re-implementation
R_a, G_a, log_a = mini.art_iteration(net_a, opt_a, noise, sigmas, cond, groups, reward)

# ---- repo pipeline (same structure as backends/tensor_fixture.py)
seen = []
_orig = objectives.art_loss
def spy(*args, **kwargs):
    out = _orig(*args, **kwargs)
    seen.append(out)
    return out
objectives.art_loss = spy
predict = lambda s, t, rows: net_b(s, t, cond[rows])
selected = tuple(range(STEPS))
cache = collect_rollout(predict, noise, sigmas, selected=selected, microbatch=5)
R_b = reward(cache.endpoints)
stats = art_scalar_covariance(cache.endpoints, R_b, groups, P)
obj = ARTObjective(cache, predict, stats["group_directions"][groups],
                   stats["group_guidance_magnitudes"][groups],
                   torch.zeros(P * K, dtype=torch.bool), torch.device("cpu"))
log_b = []
for items in disjoint_windows(P * K, len(selected)):
    seen.clear()
    m = accumulated_step(net_b, opt_b, items, obj, microbatch=7,
                         denominator=len(items), max_grad_norm=1.0)
    keep = torch.cat([o["keep_mask"] for o in seen])
    log_b.append((m["train/loss"], 1 - keep.float().mean().item()))

print(f"lr={LR}  P={P} K={K} steps={STEPS}  windows of {len(items)} (row, t) items")
print("rewards        max |diff| :", (R_a - R_b).abs().max().item())
print("guidance G     max |diff| :", (G_a - stats['group_directions'][groups]).abs().max().item())
print("rms(G) per prompt         :", [round(v, 6) for v in mini.rms(stats['group_directions']).tolist()])
for u, ((la, ma), (lb, mb)) in enumerate(zip(log_a, log_b), 1):
    print(f"update {u}: loss mini={la:.6f} repo={lb:.6f} | masked frac mini={ma:.3f} repo={mb:.3f}")
diff = max((pa - pb).abs().max().item() for pa, pb in zip(net_a.parameters(), net_b.parameters()))
print("params after 2 updates max |diff| :", diff)
