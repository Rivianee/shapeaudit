"""Acha f-strings com expressão quebrada em mais de uma linha.
Isso só é válido a partir do Python 3.12 (PEP 701); em 3.9–3.11 dá
SyntaxError. O pacote declara suportar 3.9+, então isso é um bug."""
import glob, io, sys, tokenize

def problemas(caminho):
    out = []
    with open(caminho, 'rb') as fh:
        toks = list(tokenize.tokenize(fh.readline))
    abertos = []
    for t in toks:
        if t.type == tokenize.FSTRING_START:
            abertos.append(t)
        elif t.type == tokenize.FSTRING_END and abertos:
            ini = abertos.pop()
            triplo = ini.string.endswith(('"""', "'''"))
            if not triplo and t.end[0] > ini.start[0]:
                out.append((ini.start[0], t.end[0]))
    return out

if __name__ == "__main__":
    alvo = sys.argv[1:] or (glob.glob('modelcheck/*.py') + glob.glob('shapeaudit/*.py')
                            + glob.glob('modelguard/*.py') + glob.glob('tests/*.py'))
    total = 0
    for f in sorted(alvo):
        for ini, fim in problemas(f):
            print(f'{f}:{ini}-{fim}  f-string quebrada em {fim - ini + 1} linhas')
            total += 1
    print(f'total: {total}')
    sys.exit(1 if total else 0)
