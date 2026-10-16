# DiffusionART vs on-policy DiffusionNFT on a 2D flow-matching toy.
# Notation: t=1 noise -> t=0 data, x_t=(1-t)x0+t*eps, u_t=x0-eps, Euler x_{t-h}=x_t+h*v.
import math, sys, torch, torch.nn as nn
torch.manual_seed(0); torch.set_num_threads(4)

MODES = torch.tensor([[2*math.cos(a), 2*math.sin(a)] for a in torch.arange(8)*math.pi/4])
def sample_data(n):                       # 8-Gaussian ring, radius 2, std 0.15
    return MODES[torch.randint(0, 8, (n,))] + 0.15*torch.randn(n, 2)
P = 8                                      # prompts; prompt c prefers ring mode c
def reward(x, c): return -(x - MODES[c]).square().sum(-1)

class Vel(nn.Module):                      # v_theta(x_t, t, c)
    def __init__(s, h=128):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(3 + P, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(),
                              nn.Linear(h, h), nn.SiLU(), nn.Linear(h, 2))
    def forward(s, x, t, c): return s.net(torch.cat([x, t[:, None], nn.functional.one_hot(c, P).float()], -1))

def pretrain(steps=3000):
    m = Vel(); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(steps):
        x0 = sample_data(512); eps = torch.randn_like(x0); t = torch.rand(512)
        xt = (1-t[:, None])*x0 + t[:, None]*eps
        c = torch.randint(0, P, (512,))                      # data ignores c: base model is prompt-agnostic
        loss = (m(xt, t, c) - (x0 - eps)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

N_STEPS = 10
@torch.no_grad()
def ode_rollout(m, c):                     # returns endpoints and cached (x_t, t, v_old)
    n = len(c); x = torch.randn(n, 2); ts = torch.linspace(1, 0, N_STEPS+1); xs, tt, vs = [], [], []
    for k in range(N_STEPS):
        t = ts[k].expand(n); v = m(x, t, c)
        xs.append(x); tt.append(t); vs.append(v)
        x = x + (ts[k] - ts[k+1])*v
    return x, torch.stack(xs, 1), torch.stack(tt, 1), torch.stack(vs, 1)

def prompts(groups, K):                    # `groups` prompt instances, K samples each
    return torch.arange(groups).remainder(P).repeat_interleave(K)

def group_cov_guidance(x0, R, K):          # Eq. (art_empirical_cov): per-prompt Cov(R, x0) / RMS
    X, R = x0.view(-1, K, 2), R.view(-1, K)
    cov = ((R - R.mean(1, keepdim=True))[..., None] * (X - X.mean(1, keepdim=True))).mean(1)
    return cov / cov.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)

def art_iteration(m, opt, groups=16, K=8, updates=2, nu=0.8, delta=1e-5):
    c = prompts(groups, K); x0, xs, ts, vs = ode_rollout(m, c)          # ODE rollout, cache v_old
    G = group_cov_guidance(x0, reward(x0, c), K).repeat_interleave(K, 0)  # one G(c) per prompt
    tgt = vs + G[:, None, :]                                             # frozen v_old + G(c)
    i = torch.arange(groups*K)
    batches = [i[i % K < K//2], i[i % K >= K//2]] if updates == 2 else [i]   # split within prompt
    h, log = 1.0/N_STEPS, {}
    for k, b in enumerate(batches):
        x, t, cb = xs[b].reshape(-1, 2), ts[b].reshape(-1), c[b].repeat_interleave(N_STEPS)
        v0, vt = vs[b].reshape(-1, 2), tgt[b].reshape(-1, 2)
        v = m(x, t, cb); d = (v - v0).detach()
        top = t > 1 - 1e-6                                               # Appendix C, t=1 convention
        sig2 = torch.where(top, torch.full_like(t, nu**2), nu**2 * t/(1-t).clamp_min(1e-6) * h)
        coef = torch.where(top, torch.full_like(t, h), h*(1 + nu**2/2))
        kl = coef**2 * d.square().mean(-1) / (2*sig2)                    # KL proxy m(s_t)
        outward = (d * (vt - v.detach())).sum(-1) > 0                    # <d, G - d> > 0
        masked = (kl >= delta) & outward
        w = t * (1 - masked.float())                                     # w(s_t)
        loss = (w * (v - vt).square().mean(-1)).mean()
        opt.zero_grad(); loss.backward()
        log[f'gn{k}'] = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item()
        log[f'drift{k}'] = d.square().mean().sqrt().item(); log[f'mask{k}'] = masked.float().mean().item()
        opt.step()
    return log

def nft_iteration(m, opt, groups=16, K=8, beta=1.0, updates=1, branch='both'):
    c = prompts(groups, K); x0, *_ = ode_rollout(m, c)
    R = reward(x0, c).view(-1, K)
    r = 0.5*(1 + ((R - R.mean(1, keepdim=True)) / R.std(1, keepdim=True).clamp_min(1e-6)).clamp(-1, 1))
    S = N_STEPS                                                          # S forward-noised states per endpoint
    x0r, rr, cr = x0.repeat_interleave(S, 0), r.view(-1).repeat_interleave(S, 0), c.repeat_interleave(S)
    t = torch.randint(1, N_STEPS+1, (len(x0r),)).float() / N_STEPS       # same t-grid as the sampler
    eps = torch.randn_like(x0r); xt = (1-t[:, None])*x0r + t[:, None]*eps; u = x0r - eps
    with torch.no_grad(): v_old = m(xt, t, cr)                           # on-policy: theta_old = theta
    a = ((2*rr - 1)/beta)[:, None]
    i = torch.randperm(len(xt)); log = {}
    for k, b in enumerate(i.chunk(updates)):
        v, vo = m(xt[b], t[b], cr[b]), v_old[b]
        if branch == 'both':   loss = beta**2 * (v - (a[b]*u[b] + (1-a[b])*vo)).square().sum(-1).mean()
        elif branch == 'pos':  loss = (rr[b] * (vo + beta*(v - vo) - u[b]).square().sum(-1)).mean()
        else:                  loss = ((1-rr[b]) * (vo - beta*(v - vo) - u[b]).square().sum(-1)).mean()
        opt.zero_grad(); loss.backward()
        log[f'gn{k}'] = torch.nn.utils.clip_grad_norm_(m.parameters(), 1e9).item(); opt.step()
    return log

@torch.no_grad()
def consistency_diag(m, cid=1, M=4096, n_states=1024):
    """Proposition 1 for prompt cid: F_NFT = 2g + (2q-1)e, with v_fwd from a kernel posterior."""
    cE = torch.full((M,), cid); E, *_ = ode_rollout(m, cE); RE = reward(E, cE)
    rE = 0.5*(1 + ((RE - RE.mean())/RE.std().clamp_min(1e-6)).clamp(-1, 1))   # goodness r(x0)
    x0 = E[torch.randint(0, M, (n_states,))]
    t = (torch.randint(1, N_STEPS+1, (n_states,)).float()/N_STEPS).clamp_min(0.2)  # kernel is poor near t=0
    xt = (1-t[:, None])*x0 + t[:, None]*torch.randn_like(x0)
    logw = -((xt[:, None, :] - (1-t[:, None, None])*E[None]).square().sum(-1)) / (2*t[:, None]**2)
    w = logw.softmax(-1)                                                 # Q_old(x0_j | s_t)
    uj = (E[None] - xt[:, None, :]) / t[:, None, None]                   # u_t implied by endpoint j
    v_fwd = (w[..., None]*uj).sum(1)
    q = (w*rE[None]).sum(1)
    g = (w[..., None]*rE[None, :, None]*uj).sum(1) - q[:, None]*v_fwd
    e = v_fwd - m(xt, t, torch.full((n_states,), cid))                   # forward-reverse gap
    rms = lambda z: z.square().sum(-1).mean().sqrt().item()
    hi, lo = t >= 0.7, t <= 0.4
    return dict(g2=rms(2*g), cons=rms((2*q - 1)[:, None]*e), e=rms(e),
                imb_hi=(2*q - 1)[hi].abs().mean().item(), imb_lo=(2*q - 1)[lo].abs().mean().item())

@torch.no_grad()
def evaluate(m, n_per=512):
    c = torch.arange(P).repeat_interleave(n_per); x0, *_ = ode_rollout(m, c)
    hit = ((x0 - MODES[c]).norm(dim=-1) < 0.5).float().mean().item()   # fraction landing on own mode
    return reward(x0, c).mean().item(), hit, x0.norm(dim=-1).max().item()

def run(name, base, step_fn, iters, lr, K, diag=True, seed=1, **kw):
    torch.manual_seed(seed); m = Vel(); m.load_state_dict(base.state_dict())
    opt = torch.optim.Adam(m.parameters(), lr); gmax = 0.0
    for it in range(iters + 1):
        if it % (iters//8) == 0:
            R, hit, rmax = evaluate(m); s = f'{name:12s} it={it:3d} R={R:9.3g} hit={hit:.2f} maxgrad={gmax:8.3g}'
            if diag:
                d = consistency_diag(m)
                s += (f' |2g|={d["g2"]:.3f} |(2q-1)e|={d["cons"]:.3f} |e|={d["e"]:.3f}'
                      f' |2q-1|: t>=.7 {d["imb_hi"]:.2f}, t<=.4 {d["imb_lo"]:.2f}')
            print(s, flush=True); gmax = 0.0
        if it < iters:
            log = step_fn(m, opt, groups=128//K, K=K, **kw)
            gmax = max(gmax, *(v for k, v in log.items() if k.startswith('gn')))

if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    iters, lr = 400, 1e-3
    base = pretrain()
    print(f'K={K}, 128 trajectories x {N_STEPS} ODE steps per iteration, Adam lr={lr}')
    run('ART 2 upd', base, art_iteration, iters, lr, K, updates=2)
    run('ART 1 upd', base, art_iteration, iters, lr, K, updates=1)
    run('NFT on-pol', base, nft_iteration, iters, lr, K)
    run('NFT 2 upd', base, nft_iteration, iters, lr, K, updates=2)
    run('NFT pos-only', base, nft_iteration, iters, lr, K, diag=False, branch='pos')
    run('NFT neg-only', base, nft_iteration, iters, lr, K, diag=False, branch='neg')
