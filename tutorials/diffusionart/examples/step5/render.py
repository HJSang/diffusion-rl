"""Render step5.tpl.html into fragments/step5.html, inserting exact repo quotes and real outputs.

@@Q relpath start end@@  -> lines start..end (1-based, inclusive) of DiffusionART/relpath, HTML-escaped
@@F abspath@@            -> whole file, HTML-escaped (trailing newline stripped)
"""
import html
import re
import sys
from pathlib import Path

ROOT = Path("/Users/hejian.sang/Downloads/DiffusionART-paper")
REPO = ROOT / "DiffusionART"
TPL = ROOT / "tutorial/scratch/step5/step5.tpl.html"
OUT = ROOT / "tutorial/fragments/step5.html"


def quote(match):
    path, a, b = match.group(1), int(match.group(2)), int(match.group(3))
    lines = (REPO / path).read_text().splitlines()[a - 1 : b]
    if b - a + 1 > 60:
        sys.exit(f"quote too long: {path} {a}-{b}")
    return html.escape("\n".join(lines), quote=False)


def whole(match):
    return html.escape(Path(match.group(1)).read_text().rstrip("\n"), quote=False)


text = TPL.read_text()
text = re.sub(r"@@Q (\S+) (\d+) (\d+)@@", quote, text)
text = re.sub(r"@@F (\S+)@@", whole, text)
if "@@" in text:
    sys.exit("unrendered marker")
OUT.write_text(text)
prose = re.sub(r"<pre>.*?</pre>", " ", text, flags=re.S)
prose = re.sub(r"<svg.*?</svg>", " ", prose, flags=re.S)
prose = re.sub(r"<[^>]+>", " ", prose)
print("wrote", OUT, "prose words:", len(prose.split()), "total words:", len(re.sub(r"<[^>]+>", " ", text).split()))
