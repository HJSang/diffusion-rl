import sys, torch
sys.path.insert(0, '.')
import toy_art_vs_nft as T
base = T.pretrain()
for K in [8, 4]:
  for lr in [1e-3, 3e-3]:
    for name, fn, kw in [('ART2', T.art_iteration, dict(updates=2)), ('ART1', T.art_iteration, dict(updates=1)),
                         ('NFT', T.nft_iteration, {}), ('NFT2', T.nft_iteration, dict(updates=2)),('POS', T.nft_iteration, dict(branch='pos'))]:
        torch.manual_seed(1); m = T.Vel(); m.load_state_dict(base.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr); out=[]
        for it in range(401):
            if it % 50 == 0:
                R, f, rm = T.evaluate(m, 256); out.append(f'{R:.2f}/{f:.2f}')
            if it < 400: log = fn(m, opt, K=K, groups=128//K, **kw)
        print(K, lr, name, ' '.join(out), flush=True)
