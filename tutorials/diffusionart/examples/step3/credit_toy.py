"""Why per-transition likelihood gradients are noisy: a linear-Gaussian toy.

Generator: x0 = mu + sum_k s * xi_k (N independent Gaussian steps), reward R = <x0, w>.
For a Gaussian transition with mean mu_k, d log p / d mu_k = xi_k / s, so the
GRPO output-space push at step k is P_k = sum_i A_i xi_{i,k}. Compare its
alignment with the true improving direction w, against the aggregated
endpoint covariance sum_i A_i (x0_i - mean x0) that pools all steps first.
"""
import torch

torch.manual_seed(0)
K, s, trials = 8, 0.3, 2000

def cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))

print(" D    N   cos(per-step push, w)   cos(endpoint covariance, w)")
for D in (16, 1024):
    w = torch.nn.functional.normalize(torch.randn(D), dim=0)
    for N in (1, 10):
        xi = torch.randn(trials, K, N, D)
        x0 = s * xi.sum(2)                               # mu = 0 without loss of generality
        R = x0 @ w
        A = (R - R.mean(1, keepdim=True)) / R.std(1, keepdim=True)
        push = (A[:, :, None, None] * xi).sum(1)          # [trials, N, D]
        cov = (A[:, :, None] * (x0 - x0.mean(1, keepdim=True))).sum(1)
        print(f"{D:4d} {N:4d}   {cos(push, w).mean():.3f}                   {cos(cov, w).mean():.3f}")
