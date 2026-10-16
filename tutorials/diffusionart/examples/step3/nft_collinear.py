"""Exact check of the NFT collinearity lemma on a finite endpoint distribution.

p_old = uniform over K endpoints; x_t = (1-t) x0 + t eps, so p(x_t | x0) = N((1-t) x0, t^2 I).
Ideal forward velocity v_fwd^b(x_t) = E_{Q_b}[u_t | x_t] with u_t = x0 - eps = x0 - (x_t - (1-t) x0)/t.
"""
import torch

torch.manual_seed(1)
K, D, t = 6, 2, 0.6
x0 = torch.randn(K, D, dtype=torch.float64) * 2
R = torch.randn(K, dtype=torch.float64)
Z = 1.5 * R.std(unbiased=False)
r = 0.5 * (1 + ((R - R.mean()) / Z).clamp(-1, 1))   # goodness weights
xt = torch.tensor([0.3, -0.4], dtype=torch.float64)    # one fixed state s_t

log_lik = -((xt - (1 - t) * x0) ** 2).sum(1) / (2 * t**2)
post = torch.softmax(log_lik, 0)                      # Q_old(x0 | x_t), uniform prior
u_given_x0 = x0 - (xt - (1 - t) * x0) / t             # u_t is a function of (x0, x_t)

def v_fwd(weights):                                   # posterior under a reweighted prior
    w = post * weights
    return (w[:, None] * u_given_x0).sum(0) / w.sum()

v_old, v_pos, v_neg = v_fwd(torch.ones(K)), v_fwd(r), v_fwd(1 - r)
q = (post * r).sum()
g = (post[:, None] * (r[:, None] - q) * u_given_x0).sum(0)   # Cov(r, u_t | s_t)
print("q =", round(q.item(), 4))
print("q (v+ - v_old)     =", (q * (v_pos - v_old)).numpy().round(6))
print("(1-q)(v_old - v-)  =", ((1 - q) * (v_old - v_neg)).numpy().round(6))
print("g = Cov(r, u | s_t) =", g.numpy().round(6))
