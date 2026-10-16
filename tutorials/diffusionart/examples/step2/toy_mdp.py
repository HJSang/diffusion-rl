import math, torch, torch.nn as nn
torch.manual_seed(0)
D, T, NU, K = 2, 10, 0.8, 4            # latent dim, sampling steps, SDE noise level, samples/prompt

class Velocity(nn.Module):              # v_theta(x_t, t, c), points toward the data (x0 - eps)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D + 2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, D))
    def forward(self, x, t, c):         # x: [B,D], t: [B,1], c: [B,1]
        return self.net(torch.cat([x, t, c], 1))

def data(n):                            # "pretraining data": two blobs at (+-2, 0)
    side = torch.randint(0, 2, (n, 1)) * 2 - 1.0
    return torch.cat([2 * side, torch.zeros(n, 1)], 1) + 0.3 * torch.randn(n, D)

model = Velocity()
opt = torch.optim.Adam(model.parameters(), 3e-3)
for _ in range(3000):                   # flow-matching pretraining (Step 1); c is ignored by the data
    x0, eps, t = data(512), torch.randn(512, D), torch.rand(512, 1)
    c = torch.randint(0, 2, (512, 1)) * 2 - 1.0
    loss = (model((1 - t) * x0 + t * eps, t, c) - (x0 - eps)).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

ts = torch.linspace(1, 0, T + 1)        # t_0 = 1 (noise) ... t_T = 0 (endpoint)

def sde_step_params(x, v, t, t_next):   # Flow-GRPO SDE, contract sign (v = x0 - eps)
    h = t - t_next
    g = t_next if t == 1 else t         # repo guard: at t = 1 use the next sigma in 1/(1-t)
    k = NU**2 / (2 * (1 - g))
    mean = x + h * ((1 + k * (1 - t)) * v - k * x)
    std = NU * math.sqrt(t * h / (1 - g))
    return mean, std

def gauss_logp(x_next, mean, std):      # summed over all D coordinates
    return (-(x_next - mean).pow(2) / (2 * std**2) - math.log(std) - 0.5 * math.log(2 * math.pi)).sum(1)

@torch.no_grad()
def rollout(c):                         # c: [B,1]; returns per-step states, next states, old logp
    x = torch.randn(len(c), D); states, nexts, logps = [], [], []
    for k in range(T):
        t, tn = ts[k].item(), ts[k + 1].item()
        mean, std = sde_step_params(x, model(x, torch.full_like(c, t), c), t, tn)
        x_next = mean + std * torch.randn_like(x)          # the "action"
        states.append(x); nexts.append(x_next); logps.append(gauss_logp(x_next, mean, std))
        x = x_next
    return x, torch.stack(states), torch.stack(nexts), torch.stack(logps)   # [T,B,...]

c = torch.tensor([[-1.0]] * K + [[1.0]] * K)                 # two prompts, K samples each
x0, S, N, old_logp = rollout(c)
R = c[:, 0] * x0[:, 0]                                       # reward: land on the side named by c
A = torch.empty_like(R)
for g in (slice(0, K), slice(K, 2 * K)):                     # GRPO group normalisation
    A[g] = (R[g] - R[g].mean()) / (R[g].std() + 1e-4)

def new_logp(k):                                             # replay step k under current params
    t, tn = ts[k].item(), ts[k + 1].item()
    mean, std = sde_step_params(S[k], model(S[k], torch.full_like(c, t), c), t, tn)
    return gauss_logp(N[k], mean, std)

print("rewards   ", R.numpy().round(2)); print("advantages", A.numpy().round(2))
print("old logp of traj 0, per step:", old_logp[:, 0].numpy().round(2))
print("replay ratio before update (should be 1):", torch.stack([(new_logp(k) - old_logp[k]).exp() for k in range(T)]).max().item())

params = [p for p in model.parameters()]
per_tr = []                                                  # per-transition gradient A_i * grad log p
for k in range(T):
    lp = new_logp(k)
    for i in range(2 * K):
        gr = torch.autograd.grad(A[i] * lp[i], params, retain_graph=True)
        per_tr.append(torch.cat([q.flatten() for q in gr]))
G = torch.stack(per_tr).view(T, 2 * K, -1)
mean_g = G.mean((0, 1))
print(f"|mean gradient| = {mean_g.norm():.3f}, mean |per-transition gradient| = {G.norm(dim=2).mean():.3f}")
print("per-step SNR |mean_i g_ik| / std_i:", [round((G[k].mean(0).norm() / G[k].std(0).norm()).item(), 2) for k in range(T)])

opt = torch.optim.SGD(model.parameters(), lr=1e-3)           # one clipped-PPO step
loss = 0
for k in range(T):
    ratio = (new_logp(k) - old_logp[k]).exp()
    loss = loss - torch.minimum(ratio * A, ratio.clamp(0.8, 1.2) * A).mean() / T
opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    for k in range(T):
        r = (new_logp(k) - old_logp[k]).exp()
        std = sde_step_params(S[k], torch.zeros_like(S[k]), ts[k].item(), ts[k + 1].item())[1]
        print(f"step {k} t={ts[k]:.1f} std={std:.3f} ratio min/max = {r.min():.4f}/{r.max():.4f}")
