import sys, torch
sys.path.insert(0, '.')
import toy_art_vs_nft as T
torch.manual_seed(0); base = T.pretrain()
K, lr = int(sys.argv[1]), float(sys.argv[2])
for seed in [1, 2, 3]:
    for name, fn, kw in [('ART2', T.art_iteration, dict(updates=2)), ('ART1', T.art_iteration, dict(updates=1)),
                         ('NFT', T.nft_iteration, {}), ('NFT2', T.nft_iteration, dict(updates=2))]:
        torch.manual_seed(seed); m = T.Vel(); m.load_state_dict(base.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr); out=[]
        for it in range(401):
            if it % 50 == 0:
                R, f, rm = T.evaluate(m, 256); out.append(f'{R:.2f}')
            if it < 400: log = fn(m, opt, K=K, groups=128//K, **kw)
        print(K, lr, seed, name, ' '.join(out), flush=True)
