import math, torch
torch.manual_seed(0)
D, NU, K, M = 4, 0.8, 8, 40000          # dims, SDE noise, group size, Monte-Carlo trajectories

def v_star(x, t):                       # exact velocity for data x0 ~ N(0, I): E[x0 - eps | x_t]
    return (1 - 2 * t) / ((1 - t) ** 2 + t ** 2) * x

def sde(x, v, t, tn):                   # Flow-GRPO SDE step (contract sign), returns mean, std, dmean/dv
    h, g = t - tn, (tn if t == 1 else t)
    k = NU**2 / (2 * (1 - g))
    return x + h * ((1 + k * (1 - t)) * v - k * x), NU * math.sqrt(t * h / (1 - g)), h * (1 + k * (1 - t))

def run(T, theta):                      # policy v = v* + theta; reward R = x0[0]
    ts = torch.linspace(1, 0, T + 1).tolist()
    x1 = torch.randn(M, D); x = x1; scores = []
    for k in range(T):
        mean, std, dmu = sde(x, v_star(x, ts[k]) + theta, ts[k], ts[k + 1])
        xi = torch.randn(M, D)
        x = mean + std * xi
        scores.append(xi / std * dmu)    # grad_theta log p(x_next | x) = (x_next - mean)/std^2 * dmean/dtheta
    return x1, x, torch.stack(scores)    # scores: [T, M, D]

def group_adv(R):                       # GRPO advantage inside groups of K
    g = R.view(-1, K)
    return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + 1e-4)).view(-1)

print(" T | cos(E g, true) | SNR one transition | SNR one group (K*T terms) | Var(R) explained by x1 | SNR endpoint-cov")
for T in (1, 2, 5, 10, 20, 50):
    theta = torch.zeros(D, requires_grad=True)
    x1, x0, S = run(T, theta)
    true = torch.autograd.grad(x0[:, 0].mean(), theta)[0]          # pathwise gradient of E[R]
    R = x0[:, 0].detach(); A = group_adv(R)
    per_tr = A[None, :, None] * S                                   # [T, M, D]: one transition's estimate
    snr_tr = (per_tr.mean(1).norm(dim=1) / per_tr.std(1).norm(dim=1)).mean()
    grp = per_tr.mean(0).view(-1, K, D).mean(1)                     # one group's full estimate
    snr_grp = grp.mean(0).norm() / grp.std(0).norm()
    cos = torch.cosine_similarity(grp.mean(0), true, dim=0)
    beta = torch.linalg.lstsq(torch.cat([x1, torch.ones(M, 1)], 1), R[:, None]).solution
    r2 = 1 - (R - torch.cat([x1, torch.ones(M, 1)], 1) @ beta[:, 0]).var() / R.var()
    cov = ((R.view(-1, K, 1) - R.view(-1, K, 1).mean(1, keepdim=True)) *
           (x0.detach().view(-1, K, D) - x0.detach().view(-1, K, D).mean(1, keepdim=True))).mean(1)
    snr_cov = cov.mean(0).norm() / cov.std(0).norm()
    print(f"{T:2d} | {cos:14.3f} | {snr_tr:18.3f} | {snr_grp:25.3f} | {r2:22.3f} | {snr_cov:16.3f}")
