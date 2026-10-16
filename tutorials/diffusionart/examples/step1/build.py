import html, textwrap, pathlib
here = pathlib.Path(__file__).parent
repo = here.parents[2] / "DiffusionART" / "src" / "visual_rl"
def lines(p, a, b): return textwrap.dedent("".join(open(p).readlines()[a-1:b])).rstrip("\n")
toy = (here / "fm_toy.py").read_text()
i = toy.index("# ---- samplers")
sub = {
 "TOY1": toy[:i].rstrip("\n"), "TOY2": toy[i:].rstrip("\n"),
 "TOY_OUT": (here / "fm_toy.out").read_text().rstrip("\n"),
 "CFG_OUT": "\n".join(l for l in (here / "cfg_toy.out").read_text().splitlines() if l.startswith("CFG")),
 "SHIFT_OUT": "\n".join(l for l in (here / "cfg_toy.out").read_text().splitlines() if l.startswith(("uniform", "shift"))),
 "REPO_ODE": lines(repo / "sampling.py", 74, 87),
 "REPO_SDE": lines(repo / "sampling.py", 94, 102),
 "REPO_LOGP": lines(repo / "sampling.py", 117, 125),
 "REPO_SHIFT": lines(repo / "sampling.py", 25, 30),
 "REPO_CFG": lines(repo / "models" / "sd3.py", 95, 97),
 "SD3_SKETCH": (here / "sd3_sketch.py").read_text().rstrip("\n"),
}
t = (here / "step1.tmpl.html").read_text()
for k, v in sub.items():
    assert f"%%{k}%%" in t, k
    t = t.replace(f"%%{k}%%", html.escape(v, quote=False))
assert "%%" not in t
(here.parents[1] / "fragments" / "step1.html").write_text(t)
print("ok", len(t))
