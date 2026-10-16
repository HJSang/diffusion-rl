"""Check: repo nft_loss (x0-space, per-branch detached weights) is a single-target MSE
in velocity with effective coefficient A' = (a-b)/(a+b), a = p/w+, b = (1-p)/w-."""
import sys, torch
sys.path.insert(0, sys.argv[1])
from visual_rl.algorithms.nft import nft_loss
torch.manual_seed(0)
N, D, beta = 5, 8, 0.3
x0, eps = torch.randn(N, D, dtype=torch.float64), torch.randn(N, D, dtype=torch.float64)
t = torch.rand(N, dtype=torch.float64) * 0.9 + 0.05
xt = (1 - t[:, None]) * x0 + t[:, None] * eps
old = torch.randn(N, D, dtype=torch.float64)            # SD3-convention old prediction (eps - x0 sign)
pred = (old + 0.1 * torch.randn(N, D, dtype=torch.float64)).requires_grad_(True)
adv = torch.randn(N, dtype=torch.float64) * 3
out = nft_loss(pred, old, x0, xt, t, adv, nft_beta=beta)
out["per_item_loss"].sum().backward()
p = out["positive_probability"][:, None]
wp, wn = out["positive_weight"][:, None], out["negative_weight"][:, None]
a, b = p / wp, (1 - p) / wn
A_eff = (a - b) / (a + b)
u_sd3 = eps - x0                                          # SD3 target = -u_t
d, h = pred.detach() - old, u_sd3 - old
# loss = 5/beta * t^2/D * [(a+b) beta^2 |d|^2 - 2 beta (a-b) <d,h> + ...] -> grad wrt pred:
pred_grad = 5 / beta * (t[:, None] ** 2) / D * (a + b) * 2 * beta**2 * (d - A_eff / beta * h)
print("max |autograd - single-target formula| =", (pred.grad - pred_grad).abs().max().item())
print("2p-1 :", (2 * p - 1).flatten().numpy().round(3))
print("A_eff:", A_eff.flatten().numpy().round(3))
