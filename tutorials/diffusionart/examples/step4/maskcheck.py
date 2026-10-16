import sys, torch
sys.path.insert(0, '.')
import toy_art_vs_nft as T
torch.manual_seed(0); base = T.pretrain()
for K in [2, 8]:
  for delta in [1e-5, 1e-3, float('inf')]:
    torch.manual_seed(1); m = T.Vel(); m.load_state_dict(base.state_dict())
    opt = torch.optim.Adam(m.parameters(), 1e-3); out=[]; mf=[]; dr=[]
    for it in range(401):
        if it % 100 == 0:
            R, f, rm = T.evaluate(m, 256); out.append(f'{R:.3f}/{f:.2f}')
        if it < 400:
            log = T.art_iteration(m, opt, K=K, groups=128//K, delta=delta); mf.append(log['mask1']); dr.append(log['drift1'])
    print(K, delta, ' '.join(out), 'mean mask frac', sum(mf)/len(mf), 'mean drift', sum(dr)/len(dr), flush=True)
