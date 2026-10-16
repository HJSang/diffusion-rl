# Finite-difference check of d/d(eta) E_eta[x0] at eta=0 == Cov_old(R, x0).
import torch
torch.manual_seed(0)
modes = torch.tensor([[2., 0.], [0., 2.], [-2., 0.], [0., -2.]])
x0 = modes[torch.randint(0, 4, (200_000,))] + 0.3*torch.randn(200_000, 2)   # samples of p_old
x0 = x0.double()
R = -(x0 - torch.tensor([1.5, 1.5]).double()).square().sum(-1)             # any black-box reward

def tilted_mean(eta):                       # E_{p_eta}[x0] with p_eta ∝ p_old * exp(eta R), via self-normalized IS
    w = torch.softmax(eta * R, 0)
    return (w[:, None] * x0).sum(0)

cov = ((R - R.mean())[:, None] * (x0 - x0.mean(0))).mean(0)
for h in [1e-1, 1e-2, 1e-3, 1e-4]:
    fd = (tilted_mean(h) - tilted_mean(-h)) / (2*h)
    print(f'h={h:.0e}  finite diff = {fd.numpy().round(5)}  Cov(R,x0) = {cov.numpy().round(5)}  '
          f'max rel err = {((fd - cov).abs() / cov.abs()).max():.2e}')

# 1D Gaussian sanity check with a closed form: p_old = N(mu, s^2), R(x) = x  =>  E_eta[x] = mu + eta s^2
mu, s = 0.5, 1.3
z = mu + s*torch.randn(2_000_000).double()
print('1D Gaussian: Cov(R,x) =', round(((z - z.mean())**2).mean().item(), 4), ' closed form s^2 =', s**2,
      ' FD =', round(((torch.softmax(1e-3*z, 0)*z).sum() - (torch.softmax(-1e-3*z, 0)*z).sum()).item() / 2e-3, 4))
