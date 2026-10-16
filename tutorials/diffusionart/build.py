"""Assemble the DiffusionART tutorial from HTML fragments.

Outputs
  site/index.html          full tutorial, published as the Artifact page (no doctype: the host wraps it)
  site/step-N.html         one standalone page per step (full documents)
  DiffusionART-tutorial.html  full tutorial as a standalone document for opening locally

Usage: python3 build.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRAG = ROOT / "fragments"
SITE = ROOT / "site"

STEPS = [
    (1, "Flow matching and sampling"),
    (2, "From LLM RL to diffusion RL"),
    (3, "The baselines"),
    (4, "The DiffusionART paper"),
    (5, "Reading and running the code"),
]

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    "family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&"
    "family=JetBrains+Mono:wght@400;600&"
    'family=Literata:opsz,wght@7..72,500;7..72,650&display=swap">'
)

MATHJAX = """<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['\\\\[', '\\\\]'], ['$$', '$$']],
    macros: { E: '\\\\mathbb{E}', rms: '\\\\operatorname{rms}', Cov: '\\\\operatorname{Cov}' }
  },
  svg: { fontCache: 'global' },
  options: { skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'] }
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-svg.js" async></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>"""

FOOT_SCRIPT = """<script>
(() => {
  try { if (window.hljs) document.querySelectorAll('pre code').forEach(el => window.hljs.highlightElement(el)); } catch (e) {}
  // Copy buttons on code blocks.
  document.querySelectorAll('figure.code pre:first-of-type').forEach(pre => {
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'copy'; btn.textContent = 'Copy';
    btn.addEventListener('click', () => {
      const text = pre.innerText;
      const done = () => { btn.textContent = 'Copied'; setTimeout(() => btn.textContent = 'Copy', 1400); };
      const fallback = () => { const r = document.createRange(); r.selectNodeContents(pre);
        const s = getSelection(); s.removeAllRanges(); s.addRange(r); btn.textContent = 'Selected'; };
      try { navigator.clipboard.writeText(text).then(done, fallback); } catch (e) { fallback(); }
    });
    const wrap = document.createElement('div');
    wrap.className = 'pre-wrap';
    pre.parentElement.insertBefore(wrap, pre);
    wrap.appendChild(pre); wrap.appendChild(btn);
  });
  // Highlight the current section in the sidebar.
  const links = [...document.querySelectorAll('.toc a[href^="#"]')];
  const map = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const targets = [...map.keys()].map(id => document.getElementById(id)).filter(Boolean);
  if ('IntersectionObserver' in window && targets.length) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          links.forEach(a => a.classList.remove('active'));
          const a = map.get(e.target.id); if (a) a.classList.add('active');
        }
      });
    }, { rootMargin: '0px 0px -75% 0px' });
    targets.forEach(t => io.observe(t));
  }
})();
</script>"""


def read(name: str) -> str:
    return (FRAG / name).read_text(encoding="utf-8")


def h3s(fragment: str):
    """(id, text) for each h3 in a fragment."""
    out = []
    for m in re.finditer(r'<h3[^>]*id="([^"]+)"[^>]*>(.*?)</h3>', fragment, re.S):
        text = re.sub(r"<[^>]+>", "", m.group(2))
        text = re.sub(r"\s+", " ", text).strip()
        out.append((m.group(1), html.unescape(text)))
    return out


def toc(steps_frags, *, full: bool, current: int | None = None) -> str:
    items = []
    if full:
        items.append('<li class="toc-step"><a href="#overview">Overview</a></li>')
    for n, title in STEPS:
        frag = steps_frags[n]
        href = f"#step-{n}" if full else f"step-{n}.html"
        cls = "toc-step" + (" current" if current == n else "")
        sub = ""
        if full or current == n:
            prefix = "" if full or current == n else f"step-{n}.html"
            sub = "<ol>" + "".join(
                f'<li><a href="{prefix}#{i}">{html.escape(t)}</a></li>' for i, t in h3s(frag)
            ) + "</ol>"
        items.append(
            f'<li class="{cls}"><a href="{href}"><span class="num">{n}</span>{html.escape(title)}</a>{sub}</li>'
        )
    if full:
        items.append('<li class="toc-step"><a href="#closing">Putting it together</a></li>')
    pages = ""
    if full:
        pages = '<p class="toc-alt">Prefer one step at a time? ' + " · ".join(
            f'<a href="step-{n}.html">Step {n}</a>' for n, _ in STEPS
        ) + "</p>"
    else:
        pages = '<p class="toc-alt"><a href="index.html">Full tutorial on one page</a></p>'
    return (
        '<nav class="toc" aria-label="Contents"><p class="toc-brand">DiffusionART<br><span>a guided path</span></p>'
        f'<ol class="toc-list">{"".join(items)}</ol>{pages}</nav>'
    )


def page(title: str, nav: str, main: str, *, doctype: bool) -> str:
    css = (ROOT / "tutorial.css").read_text(encoding="utf-8")
    head = (
        f'<title>{html.escape(title)}</title>\n{FONTS}\n<style>\n{css}\n</style>\n{MATHJAX}\n'
    )
    body = f'<div class="shell">{nav}<main class="content">{main}</main></div>\n{FOOT_SCRIPT}\n'
    if doctype:
        return (
            '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head}</head>\n<body>\n{body}</body>\n</html>\n"
        )
    return head + body


def pager(n: int) -> str:
    prev_ = f'<a class="prev" href="step-{n-1}.html">← Step {n-1}: {html.escape(STEPS[n-2][1])}</a>' if n > 1 else '<a class="prev" href="index.html">← Overview</a>'
    next_ = f'<a class="next" href="step-{n+1}.html">Step {n+1}: {html.escape(STEPS[n][1])} →</a>' if n < len(STEPS) else '<a class="next" href="index.html#closing">Putting it together →</a>'
    return f'<nav class="pager">{prev_}{next_}</nav>'


def main() -> None:
    frags = {n: read(f"step{n}.html") for n, _ in STEPS if (FRAG / f"step{n}.html").exists()}
    missing = [n for n, _ in STEPS if n not in frags]
    for n in missing:
        frags[n] = (
            f'<section class="step" id="step-{n}"><header class="step-head"><p class="eyebrow">Step {n}</p>'
            f"<h2>{html.escape(STEPS[n-1][1])}</h2><p class=\"lede\">Being written.</p></header></section>"
        )
    intro, closing = read("intro.html"), read("closing.html")

    SITE.mkdir(exist_ok=True)
    full_main = intro + "".join(frags[n] for n, _ in STEPS) + closing
    full_nav = toc(frags, full=True)
    title = "DiffusionART Study Guide"
    (SITE / "index.html").write_text(page(title, full_nav, full_main, doctype=False), encoding="utf-8")
    (ROOT / "DiffusionART-tutorial.html").write_text(page(title, full_nav, full_main, doctype=True), encoding="utf-8")

    for n, name in STEPS:
        main_html = frags[n] + pager(n)
        (SITE / f"step-{n}.html").write_text(
            page(f"Step {n}: {name}", toc(frags, full=False, current=n), main_html, doctype=True),
            encoding="utf-8",
        )
    print("built", sorted(p.name for p in SITE.iterdir()), "missing:", missing)


if __name__ == "__main__":
    main()
