# One DiffusionART iteration, paper convention (v points toward data; t runs 1 -> 0).
# Algorithm-level sketch: shapes, batching and distributed details are omitted.
def diffusionart_iteration(policy, prompts, K, reward_fn, opt, schedule, nu=0.8, delta=1e-5):
    # ---------------- Rollout stage (Algorithm 1, lines 1-3) ----------------
    rows = []                                   # one row per visited state s_t
    for c in prompts:
        X, rows_c = [], []
        for i in range(K):
            x = randn(D)
            for t, t_next in schedule:          # deterministic ODE, Euler shown
                v0 = policy(x, t, c).detach()   # cache v_theta^(0)(s_t) = v_old(s_t)
                rows_c.append(dict(x=x, t=t, h=t - t_next, c=c, i=i, v0=v0))
                x = x + (t - t_next) * v0
            X.append(x)                         # endpoint x_0^(i)
        X = stack(X)                            # (K, D)
        R = tensor([reward_fn(x0, c) for x0 in X])                 # black-box scalar rewards
        cov = ((R - R.mean())[:, None] * (X - X.mean(0))).mean(0)  # empirical Cov(R, x0 | c)
        G = cov / cov.square().mean().sqrt()                       # unit RMS: G(c)
        for row in rows_c:
            row['G'] = G
            row['target'] = row['v0'] + G                          # frozen guided target
        rows += rows_c
    # ---------------- Training stage (lines 4-5) ----------------
    B1 = [r for r in rows if r['i'] < K // 2]   # disjoint halves, split within each prompt
    B2 = [r for r in rows if r['i'] >= K // 2]
    for B in (B1, B2):                          # below, B.x, B.t, ... are B's rows stacked
        v = policy(B.x, B.t, B.c)                         # current prediction (with grad)
        d = (v - B.v0).detach()                           # displacement since the rollout
        sig2 = nu**2 * B.t / (1 - B.t) * B.h              # reference SDE variance (t < 1)
        m = (B.h * (1 + nu**2 / 2))**2 * d.square().mean(-1) / (2 * sig2)   # KL proxy
        outward = (d * (B.G - d)).sum(-1) > 0             # <d, G - d> > 0
        w = B.t * (1 - ((m >= delta) & outward).float())  # trust-region mask times t
        loss = (w * (v - B.target).square().mean(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
