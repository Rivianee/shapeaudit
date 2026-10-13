"""Acha f-strings com expressão quebrada em mais de uma linha.

Isso só é válido a partir do Python 3.12 (PEP 701); em 3.9–3.11 dá
SyntaxError. O pacote declara suportar 3.9+, então isso é um bug.

A checagem muda conforme a versão que está rodando, e isso é de propósito:

* **Python < 3.12** — `compile()` é a verdade. A sintaxe é um erro de fato
  nessa versão, então basta compilar cada arquivo. Pega a f-string
  multilinha e qualquer outra sintaxe que não exista na versão.
* **Python >= 3.12** — `compile()` aceita, então não serve. Aí usamos o
  tokenizador, que a partir do 3.12 emite `FSTRING_START`/`FSTRING_END` e
  deixa ver em quantas linhas a f-string se espalha.

A versão anterior deste arquivo usava só o caminho do tokenizador. Como
`tokenize.FSTRING_START` só existe no 3.12+, ela levantava `AttributeError`
em 3.11 — ou seja, era cega exatamente na versão que o teste protege.
"""
import glob
import sys
import tokenize

#: True quando o interpretador atual aceita a sintaxe do PEP 701.
PEP701 = sys.version_info >= (3, 12)


def _por_compilacao(caminho):
    """Caminho para Python < 3.12: a sintaxe é SyntaxError de verdade."""
    with open(caminho, 'rb') as fh:
        fonte = fh.read()
    try:
        compile(fonte, caminho, 'exec')
    except SyntaxError as e:
        return [(e.lineno or 0, e.lineno or 0, e.msg)]
    return []


def _por_tokenizacao(caminho):
    """Caminho para Python >= 3.12: compila, então inspecionamos os tokens."""
    with open(caminho, 'rb') as fh:
        toks = list(tokenize.tokenize(fh.readline))
    ini_tok, fim_tok = tokenize.FSTRING_START, tokenize.FSTRING_END
    out, abertos = [], []
    for t in toks:
        if t.type == ini_tok:
            abertos.append(t)
        elif t.type == fim_tok and abertos:
            ini = abertos.pop()
            triplo = ini.string.endswith(('"""', "'''"))
            if not triplo and t.end[0] > ini.start[0]:
                n = t.end[0] - ini.start[0] + 1
                out.append((ini.start[0], t.end[0],
                            f'f-string quebrada em {n} linhas'))
    return out


def problemas(caminho):
    """Devolve [(linha_inicial, linha_final, motivo)] — vazio se estiver ok."""
    return _por_tokenizacao(caminho) if PEP701 else _por_compilacao(caminho)


if __name__ == "__main__":
    alvo = sys.argv[1:] or (glob.glob('modelcheck/*.py') + glob.glob('shapeaudit/*.py')
                            + glob.glob('modelguard/*.py') + glob.glob('tests/*.py'))
    via = 'tokenizador (3.12+)' if PEP701 else 'compilacao (< 3.12)'
    print(f'checando {len(alvo)} arquivos em Python '
          f'{sys.version_info.major}.{sys.version_info.minor} via {via}')
    total = 0
    for f in sorted(alvo):
        for ini, fim, motivo in problemas(f):
            print(f'{f}:{ini}-{fim}  {motivo}')
            total += 1
    print(f'total: {total}')
    sys.exit(1 if total else 0)
