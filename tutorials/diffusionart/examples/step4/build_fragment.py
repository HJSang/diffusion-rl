import html, re
src = open('toy_art_vs_nft.py').read().splitlines()
def block(a, b): return '\n'.join(src[a-1:b])
def sel(path, keep):
    lines = open(path).read().splitlines(); out = [lines[0]]; prev = None
    for l in lines[1:]:
        m = re.match(r'(.{12}) it=\s*(\d+)', l)
        if not m: continue
        name, it = m.group(1).strip(), int(m.group(2))
        if it in keep.get(name, ()):
            if prev is not None and name == prev[0] and keep[name].index(it) != keep[name].index(prev[1]) + 1:
                pass
            out.append(l); prev = (name, it)
        elif out[-1] != '...': out.append('...')
    return '\n'.join(out)
K2 = sel('out_K2b.txt', {'ART 2 upd': [0, 50, 100, 400], 'ART 1 upd': [50, 100, 400],
                          'NFT on-pol': [0, 50, 100, 150, 200, 300, 350], 'NFT 2 upd': [50, 100, 150, 200],
                          'NFT pos-only': [0, 200, 400], 'NFT neg-only': [50, 100]})
K8 = sel('out_K8b.txt', {'ART 2 upd': [100, 200, 400], 'ART 1 upd': [50, 100, 400],
                          'NFT on-pol': [100, 200, 400], 'NFT 2 upd': [100, 400],
                          'NFT pos-only': [400], 'NFT neg-only': [100]})
fills = {
    'CODE:toy1': block(1, 45), 'CODE:toy2': block(47, 90), 'CODE:toy3': block(92, 143),
    'CODE:tilt': open('tilt_check.py').read().rstrip(), 'OUT:tilt': open('tilt_check.out').read().rstrip(),
    'CODE:pseudo': open('pseudocode.py').read().rstrip(), 'CODE:maskcheck': open('maskcheck.py').read().rstrip(),
    'OUT:mask': open('maskcheck.out').read().rstrip(), 'OUT:K2': K2, 'OUT:K8': K8,
}
t = open('step4.tmpl.html').read()
for k, v in fills.items():
    assert '{{' + k + '}}' in t, k
    t = t.replace('{{' + k + '}}', html.escape(v, quote=False))
assert '{{' not in t
open('../../fragments/step4.html', 'w').write(t)
print(K2); print(K8)
