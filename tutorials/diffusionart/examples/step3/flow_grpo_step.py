"""Toy Flow-GRPO: SDE rollouts with Gaussian log-probs, group advantages, PPO-clip update.

Paper convention: t goes 1 -> 0, velocity points toward data, Euler step x <- x + h v.
The SDE step follows the paper's reference transition (appendix, nu = noise level).
"""
import torch

torch.manual_seed(0)
D, K, N, NU, CLIP = 2, 8, 6, 0.7, 0.2
times = torch.linspace(1.0, 0.0, N + 1)
target = torch.tensor([2.0, -1.0])                       # reward: be close to this point
net = torch.nn.Sequential(torch.nn.Linear(D + 1, 64), torch.nn.SiLU(), torch.nn.Linear(64, D))
opt = torch.optim.Adam(net.parameters(), lr=1e-2)

def velocity(x, t):
    return net(torch.cat([x, t.expand(len(x), 1)], dim=1))

def sde_mean_std(x, v, t, t_next):
    h = t - t_next
    t_eff = torch.where(t >= 1, t_next, t)                # same t=1 guard idea as the repo
    s2 = NU**2 * t / (1 - t_eff)                          # sigma_t^2 = nu^2 t / (1 - t)
    mean = x + h * (v * (1 + s2 * (1 - t) / (2 * t)) - s2 / (2 * t) * x)
    return mean, (s2 * h).sqrt()

def log_prob(x_next, mean, std):                          # Gaussian log density, summed over D
    return (-(x_next - mean) ** 2 / (2 * std**2) - std.log() - 0.5 * torch.log(torch.tensor(2 * torch.pi))).sum(1)

# ---- rollout with the frozen old policy: store (x_k, x_{k-1}, old log-prob) ----
states, nexts, old_lp = [], [], []
with torch.no_grad():
    x = torch.randn(K, D)
    for k in range(N):
        t, tn = times[k], times[k + 1]
        mean, std = sde_mean_std(x, velocity(x, t.view(1, 1)), t, tn)
        x_next = mean + std * torch.randn_like(x)
        states.append(x); nexts.append(x_next); old_lp.append(log_prob(x_next, mean, std))
        x = x_next
R = -((x - target) ** 2).sum(1)                           # terminal reward only
A = (R - R.mean()) / (R.std() + 1e-4)                     # group-normalized advantage
print("rewards   ", R.numpy().round(2))
print("advantages", A.numpy().round(2))

# ---- two optimizer steps on the same rollout; every transition gets its trajectory's A ----
for update in range(2):
    loss, ratios = 0.0, []
    for k in range(N):
        t, tn = times[k], times[k + 1]
        mean, std = sde_mean_std(states[k], velocity(states[k], t.view(1, 1)), t, tn)
        ratio = (log_prob(nexts[k], mean, std) - old_lp[k]).exp()
        surrogate = torch.minimum(ratio * A, ratio.clamp(1 - CLIP, 1 + CLIP) * A)
        loss = loss - surrogate.mean() / N
        ratios.append(ratio.detach())
    opt.zero_grad(); loss.backward(); opt.step()
    ratios = torch.stack(ratios)
    print(f"update {update}: loss={loss.item():+.4f}  ratio in [{ratios.min():.3f}, {ratios.max():.3f}]"
          f"  clip_frac={((ratios - 1).abs() > CLIP).float().mean():.3f}")
