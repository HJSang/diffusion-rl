"""Toy check: DiffusionNFT two-branch loss == beta^2 * single-target MSE + const.

Paper notation: t=1 noise, t=0 clean; x_t = (1-t) x0 + t eps; u_t = x0 - eps.
"""
import torch

torch.manual_seed(0)
D, N = 4, 64                                   # feature dim, batch of (x0, eps, t) items

net = torch.nn.Sequential(torch.nn.Linear(D + 1, 32), torch.nn.Tanh(), torch.nn.Linear(32, D))
old = torch.nn.Sequential(torch.nn.Linear(D + 1, 32), torch.nn.Tanh(), torch.nn.Linear(32, D))
old.load_state_dict(net.state_dict())
with torch.no_grad():                          # make theta != theta_old so the drift d is nonzero
    for p in net.parameters():
        p.add_(0.05 * torch.randn_like(p))

x0, eps = torch.randn(N, D), torch.randn(N, D)
t = torch.rand(N, 1)
xt = (1 - t) * x0 + t * eps
u = x0 - eps                                   # sample velocity target
r = torch.rand(N, 1)                           # goodness weights in [0, 1]

def v(model):
    return model(torch.cat([xt, t], dim=1))

def two_branch(beta):
    v_th, v_old = v(net), v(old).detach()
    v_pos = v_old + beta * (v_th - v_old)      # implicit positive policy
    v_neg = v_old - beta * (v_th - v_old)      # implicit negative policy
    per = r * (v_pos - u).pow(2).sum(1, keepdim=True) + (1 - r) * (v_neg - u).pow(2).sum(1, keepdim=True)
    return per.mean()

def single_target(beta):
    v_th, v_old = v(net), v(old).detach()
    a = (2 * r - 1) / beta
    v_tgt = (a * u + (1 - a) * v_old).detach()
    return (v_th - v_tgt).pow(2).sum(1, keepdim=True).mean()

def grads(loss):
    net.zero_grad()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in net.parameters()])

for beta in (1.0, 0.1):
    l2, l1 = two_branch(beta), single_target(beta)
    g2, g1 = grads(two_branch(beta)), grads(single_target(beta))
    with torch.no_grad():                      # the theta-independent constant from the proof
        h = u - v(old)
        const = (4 * r * (1 - r) * h.pow(2).sum(1, keepdim=True)).mean()
    print(f"beta={beta:4.2f}  L_NFT={l2.item():.6f}  beta^2*L_MSE+const={(beta**2 * l1 + const).item():.6f}"
          f"  max|g_NFT - beta^2 g_MSE|={(g2 - beta**2 * g1).abs().max().item():.2e}"
          f"  |g_NFT|={g2.norm().item():.4f}")

# Descent direction at the on-policy point v_theta = v_old (d = 0).
beta = 0.5
v_old = v(old).detach()
v_th = v_old.clone().requires_grad_(True)
v_pos, v_neg = v_old + beta * (v_th - v_old), v_old - beta * (v_th - v_old)
loss = (r * (v_pos - u).pow(2).sum(1, keepdim=True) + (1 - r) * (v_neg - u).pow(2).sum(1, keepdim=True)).sum()
loss.backward()
predicted = 2 * beta * (2 * r - 1) * (u - v_old)
print(f"on-policy: max|-grad - 2*beta*(2r-1)*(u - v_old)| = {(-v_th.grad - predicted).abs().max().item():.2e}")
