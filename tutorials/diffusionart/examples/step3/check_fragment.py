import re, html, sys, pathlib
root = pathlib.Path(sys.argv[1])
s = (root/'tutorial/fragments/step3.html').read_text()
blocks = re.findall(r'<pre><code class="language-(\w+)">(.*?)</code></pre>', s, re.S)
print(len(blocks), "code blocks")
repo = root/'DiffusionART/src/visual_rl'
srcs = [p.read_text() for p in repo.rglob('*.py')]
scratch = [p.read_text() for p in (root/'tutorial/scratch/step3').glob('*.py')]
for lang, b in blocks:
    code = html.unescape(b)
    if lang != 'python': continue
    chunks = [c for c in re.split(r'\n# ---- .*? ----\n|\n    # \.\.\. .*?\.\.\.\n', code)]
    for c in chunks:
        lines = [l for l in c.split('\n') if l.strip()]
        ok_repo = all(any(l in src for src in srcs) for l in lines)
        ok_scr = all(any(l in src for src in scratch) for l in lines)
        ok_block = any(c.strip() in src for src in srcs + scratch)
        print(lang, len(lines), 'lines  contiguous-match:', ok_block, ' all-lines-in-repo:', ok_repo, ' in-scratch:', ok_scr)
ids = re.findall(r'id="([^"]+)"', s)
print("bad ids:", [i for i in ids if not (i.startswith('s3-') or i.startswith('w3-') or i=='step-3')])
print("dup ids:", {i for i in ids if ids.count(i)>1})
# math: check stray < > inside $...$ (excluding &lt;)
text = re.sub(r'<script>.*?</script>', '', s, flags=re.S)
text = re.sub(r'<pre>.*?</pre>', '', text, flags=re.S)
for m in re.finditer(r'\$[^$]+\$', text):
    if '<' in m.group(0) or '>' in m.group(0): print("raw <> in math:", m.group(0)[:80])
for m in re.finditer(r'\\\[(.*?)\\\]', text, re.S):
    if '<' in m.group(1) or '>' in m.group(1): print("raw <> in display:", m.group(1)[:80])
print("dollar count even:", text.count('$') % 2 == 0)
words = re.sub(r'<[^>]+>', ' ', text)
print("prose words approx:", len(words.split()))
