import html, sys
here = sys.argv[1]; repo = sys.argv[2]; out = sys.argv[3]
lines = lambda p: open(p).read().splitlines()
esc = lambda s: html.escape(s, quote=False)
toy = lines(f"{here}/toy_mdp.py")
samp = lines(f"{repo}/src/visual_rl/sampling.py")
pol = lines(f"{repo}/src/visual_rl/algorithms/policy.py")
sub = {
  "@@CODE_TOY1@@": "\n".join(toy[0:46]),
  "@@CODE_TOY2@@": "\n".join(toy[47:]),
  "@@OUT_TOY@@": open(f"{here}/toy_mdp.out").read().rstrip(),
  "@@CODE_SNR@@": open(f"{here}/snr_vs_T.py").read().rstrip(),
  "@@OUT_SNR@@": open(f"{here}/snr_vs_T.out").read().rstrip(),
  "@@REPO_SDE@@": "\n".join(samp[93:102] + ["    # ... L103-114: Dance and CPS transitions omitted ..."] + samp[114:125]),
  "@@REPO_CLIP@@": "\n".join(pol[32:46]),
}
s = open(f"{here}/step2.tmpl.html").read()
for k, v in sub.items():
    assert k in s, k
    s = s.replace(k, esc(v))
assert "@@" not in s
open(out, "w").write(s)
