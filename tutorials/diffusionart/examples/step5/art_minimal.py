"""Minimal DiffusionART iteration in plain PyTorch, written in the paper's sign convention.

The network outputs the repo/SD3-style velocity eps - x0; we flip it once so that
v = x0 - eps points along denoising and the Euler step is x_{t-h} = x_t + h v.
"""
import torch

def rms(x):  # per-row root-mean-square over all feature axes
    return x.flatten(1).square().mean(1).sqrt()

def v_paper(net, x, t, c):  # paper velocity = -(repo velocity)
    return -net(x, t, c)

@torch.no_grad()
def ode_rollout(net, noise, sigmas, cond):
    """Euler ODE from t=1 to t=0; cache every visited state and its old velocity."""
    x, states, v_old = noise.clone(), [], []
    for i in range(len(sigmas) - 1):
        t = sigmas[i].expand(len(x))
        v = v_paper(net, x, t, cond)
        states.append(x.clone()); v_old.append(v.clone())
        x = x + (sigmas[i] - sigmas[i + 1]) * v           # h = t - t_next > 0
    return x, torch.stack(states, 1), torch.stack(v_old, 1)  # x0, [N,S,D], [N,S,D]

@torch.no_grad()
def guidance(x0, R, groups, eps=1e-5):
    """G(c) = Cov(R, x0 | c) / rms(Cov), one shared direction per prompt group."""
    G = torch.zeros(int(groups.max()) + 1, x0.shape[1])
    for g in groups.unique():
        r, x = R[groups == g], x0[groups == g]
        cov = ((r - r.mean())[:, None] * (x - x.mean(0))).sum(0) / (len(r) - 1)
        G[g] = cov / rms(cov[None]).clamp_min(eps) if (r != r[0]).any() else 0.0
    return G[groups]                                       # [N, D], row-aligned

def art_window_loss(net, states, v_old, sigmas, cond, G, rows, nu=0.8, delta=1e-5):
    """Masked, t-weighted MSE to the frozen target v_old + G over one window."""
    S = states.shape[1]
    t, t_next = sigmas[:S], sigmas[1:S + 1]
    h = (t - t_next)[None].expand(len(rows), S).reshape(-1)
    tt = t[None].expand(len(rows), S).reshape(-1)
    x = states[rows].reshape(-1, states.shape[-1])
    vo = v_old[rows].reshape(-1, states.shape[-1])
    g = G[rows][:, None].expand(-1, S, -1).reshape(-1, states.shape[-1])
    v = v_paper(net, x, tt, cond[rows].repeat_interleave(S))
    target = (vo + g).detach()                             # frozen ART target
    d = (v - vo).detach()                                  # drift from rollout policy
    # Flow-GRPO reference SDE step: mean gap = h(1+nu^2/2) d, var = nu^2 t/(1-t) h
    first = torch.isclose(tt, torch.ones_like(tt))
    var = torch.where(first, torch.full_like(tt, nu**2), nu**2 * tt / (1 - tt).clamp_min(1e-5) * h)
    coef = torch.where(first, h, h * (1 + nu**2 / 2))
    m = coef**2 * d.square().mean(1) / (2 * var)           # KL proxy per output dim
    outward = (d * (g - d)).mean(1) > 0
    w = tt * (~((m >= delta) & outward)).float()           # w = t [1 - 1{m>=delta} 1{outward}]
    w = w * (rms(g) > 0)                                   # zero-signal prompts: no loss
    per_item = w * (v - target).square().mean(1)
    return per_item.sum() / len(per_item), (w == 0).float().mean()

def art_iteration(net, opt, noise, sigmas, cond, groups, reward_fn, clip=1.0):
    x0, states, v_old = ode_rollout(net, noise, sigmas, cond)
    R = reward_fn(x0)
    G = guidance(x0, R, groups)
    n = len(x0)
    log = []
    for rows in (torch.arange(0, n // 2), torch.arange(n // 2, n)):   # B1, B2 disjoint
        loss, masked = art_window_loss(net, states, v_old, sigmas, cond, G, rows)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), clip)        # repo adds this
        opt.step()
        log.append((loss.item(), masked.item()))
    return R, G, log
