import math, torch
torch.manual_seed(0)
nu, t, h, dv = 0.8, 0.5, 0.1, 0.01
s = nu * math.sqrt(t * h / (1 - t)); dmu = h * (1 + nu**2 / 2) * dv; delta = dmu / s
print(f"s={s:.4f} dmu={dmu:.5f} delta={delta:.5f}")
for D in (1, 64, 4096, 65536, 806400):
    xi = torch.randn(2000, D)
    # x' = mu_old + s*xi ; log rho = [-(x'-mu_new)^2 + (x'-mu_old)^2]/(2 s^2), mu_new = mu_old + dmu
    per = (-(s * xi - dmu) ** 2 + (s * xi) ** 2) / (2 * s * s)
    lsum, lmean = per.sum(1), per.mean(1)
    print(f"D={D:7d} sum: mean={lsum.mean():8.4f} std={lsum.std():7.4f} (theory {-D*delta**2/2:8.4f},{math.sqrt(D)*delta:7.4f}) "
          f"clip0.2 frac={((lsum.exp()-1).abs()>0.2).float().mean():.3f} | mean-reduced std={lmean.std():.2e} "
          f"clip1e-5 frac={((lmean.exp()-1).abs()>1e-5).float().mean():.3f}")
