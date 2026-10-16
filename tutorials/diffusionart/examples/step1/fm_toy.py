import math, torch
torch.manual_seed(0)

# ---- endpoint distribution p(x0): 2D mixture of isotropic Gaussians -------------
MEANS = torch.tensor([[-2.0, -1.5], [2.0, -1.5], [0.0, 2.0]])
PI    = torch.tensor([0.5, 0.3, 0.2])
S     = 0.35                                   # per-component std

def sample_data(n):
    k = torch.multinomial(PI, n, replacement=True)
    return MEANS[k] + S * torch.randn(n, 2)

def ideal_velocity(x, t, pi=PI):
    """v*(x,t) = E[x0 - eps | x_t = x] for x_t = (1-t) x0 + t eps (closed form)."""
    t = t.reshape(-1, 1, 1)                                  # [B,1,1]
    var = (1 - t) ** 2 * S**2 + t**2                         # Var(x_t | k), per dim
    diff = x[:, None, :] - (1 - t) * MEANS[None]             # [B,K,2]
    logw = torch.log(pi) - (diff**2).sum(-1) / (2 * var[..., 0]) - torch.log(var[..., 0])
    w = torch.softmax(logw, dim=1)                           # p(k | x_t)
    vk = MEANS[None] + ((1 - t) * S**2 - t) / var * diff     # E[u | x_t, k]
    return (w[..., None] * vk).sum(1)

# ---- velocity network v_theta(x, t) and flow-matching training ------------------
net = torch.nn.Sequential(torch.nn.Linear(3, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, 256), torch.nn.SiLU(),
                          torch.nn.Linear(256, 2))
v_theta = lambda x, t: net(torch.cat([x, t.reshape(-1, 1).expand(len(x), 1)], 1))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 6000)
for it in range(6001):
    x0, eps, t = sample_data(1024), torch.randn(1024, 2), torch.rand(1024)
    xt = (1 - t[:, None]) * x0 + t[:, None] * eps            # forward noising
    u = x0 - eps                                             # per-sample target
    loss = ((v_theta(xt, t) - u) ** 2).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if it % 2000 == 0:
        print(f"iter {it:5d}  flow-matching loss {loss.item():.3f}")

with torch.no_grad():   # irreducible floor E||u - v*||^2 and fit quality
    x0, eps, t = sample_data(200_000), torch.randn(200_000, 2), torch.rand(200_000)
    xt, u = (1 - t[:, None]) * x0 + t[:, None] * eps, x0 - eps
    vs = ideal_velocity(xt, t)
    print(f"loss of ideal v*   {((vs - u)**2).sum(1).mean():.3f}   (irreducible floor)")
    print(f"loss of learned v  {((v_theta(xt, t) - u)**2).sum(1).mean():.3f}")
    print(f"E||v_theta - v*||^2 {((v_theta(xt, t) - vs)**2).sum(1).mean():.4f}")

# ---- samplers: Euler ODE and Flow-GRPO SDE (both integrate t: 1 -> 0) -----------
def sample(v, n, steps, nu=0.0):
    ts = torch.linspace(1, 0, steps + 1)
    x, logps = torch.randn(n, 2), []
    for t, tn in zip(ts[:-1], ts[1:]):
        h = (t - tn).item(); vel = v(x, t.expand(n))
        if nu == 0:                                          # deterministic Euler
            x = x + h * vel; continue
        tg = ts[1].item() if t == 1 else t.item()            # repo guard at t=1
        g2 = nu**2 * t.item() / (1 - tg)                     # sigma_t^2
        mean = x + h * (vel + g2 / (2 * t.item()) * ((1 - t.item()) * vel - x))
        std = math.sqrt(g2 * h)
        x = mean + std * torch.randn_like(x)
        logps.append((-((x - mean)**2) / (2 * std**2) - math.log(std)
                      - 0.5 * math.log(2 * math.pi)).sum(1))
    return x, (torch.stack(logps).sum(0) if logps else None)

def report(name, x, logp=None):
    k = torch.cdist(x, MEANS).argmin(1)
    frac = torch.bincount(k, minlength=3).float() / len(x)
    spread = torch.tensor([(x[k == j] - MEANS[j]).std() if (k == j).sum() > 1
                          else float('nan') for j in range(3)])
    msg = f"{name:<22} mode frac {frac.numpy().round(3)}  within-mode std {spread.numpy().round(3)}"
    if logp is not None: msg += f"  mean traj log-prob {logp.mean():.1f}"
    print(msg)

torch.manual_seed(1)
with torch.no_grad():
    report("data", sample_data(20000))
    for steps in (1, 2, 8, 50):
        report(f"ODE learned, {steps:>2} steps", sample(v_theta, 20000, steps)[0])
    report("ODE ideal v*, 50 steps", sample(ideal_velocity, 20000, 50)[0])
    report("SDE learned nu=0.7, 50", *sample(v_theta, 20000, 50, nu=0.7))
    report("SDE learned nu=0.7, 10", *sample(v_theta, 20000, 10, nu=0.7))
    print("one-step Euler from t=1 lands at", sample(ideal_velocity, 4, 1)[0][0].numpy().round(3),
          " data mean", (PI @ MEANS).numpy().round(3))
