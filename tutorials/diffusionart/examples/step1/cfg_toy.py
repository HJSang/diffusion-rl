import math, torch
torch.manual_seed(0)
MEANS = torch.tensor([[-2.0, -1.5], [2.0, -1.5], [0.0, 2.0]])
PI, S = torch.tensor([0.5, 0.3, 0.2]), 0.35

def ideal_velocity(x, t, pi=PI):
    t = t.reshape(-1, 1, 1)
    var = (1 - t) ** 2 * S**2 + t**2
    diff = x[:, None, :] - (1 - t) * MEANS[None]
    logw = torch.log(pi) - (diff**2).sum(-1) / (2 * var[..., 0]) - torch.log(var[..., 0])
    w = torch.softmax(logw, dim=1)
    return (w[..., None] * (MEANS[None] + ((1 - t) * S**2 - t) / var * diff)).sum(1)

COND = torch.tensor([0.0, 0.0, 1.0])            # condition c = "component 2"
def v_cfg(x, t, w):
    vu, vc = ideal_velocity(x, t), ideal_velocity(x, t, COND)
    return vu + w * (vc - vu)

def ode(v, n, steps):
    ts, x = torch.linspace(1, 0, steps + 1), torch.randn(n, 2)
    for t, tn in zip(ts[:-1], ts[1:]):
        x = x + (t - tn) * v(x, t.expand(n))
    return x

# worked numbers at one state
x, t = torch.tensor([[0.5, 0.5]]), torch.tensor([0.5])
vu, vc = ideal_velocity(x, t), ideal_velocity(x, t, COND)
print("x=(0.5,0.5), t=0.5:  v_uncond", vu.numpy().round(3), " v_cond", vc.numpy().round(3))
for w in (1.0, 3.0):
    print(f"   w={w}: v_cfg", v_cfg(x, t, w).numpy().round(3))

for w in (0.0, 1.0, 2.0, 4.0):
    xs = ode(lambda x, t: v_cfg(x, t, w), 20000, 50)
    k = torch.cdist(xs, MEANS).argmin(1)
    inmode = xs[k == 2]
    print(f"CFG w={w}: frac in cond mode {float((k == 2).float().mean()):.3f}"
          f"  mean {inmode.mean(0).numpy().round(3)}  std {inmode.std(0).numpy().round(3)}")

# single-Gaussian endpoint: closed form vs Monte Carlo E[u | x_t ~= x]
m, s, t, xq = 2.0, 0.5, 0.5, 1.5
var = (1 - t) ** 2 * s**2 + t**2
print("closed form v*:", m + ((1 - t) * s**2 - t) / var * (xq - (1 - t) * m))
x0, eps = m + s * torch.randn(4_000_000), torch.randn(4_000_000)
xt = (1 - t) * x0 + t * eps
sel = (xt - xq).abs() < 0.01
print("Monte Carlo   :", float((x0 - eps)[sel].mean()), " (", int(sel.sum()), "samples )")

# SD3-style shifted schedule (shift=3, 10 steps), as in sampling.shifted_schedule
sig = torch.linspace(1, 0, 11)
print("uniform:", sig.numpy().round(3))
print("shift=3:", (3 * sig / (1 + 2 * sig)).numpy().round(3))
