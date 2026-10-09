"""
Guardas por CLASSE de defeito (v1.73).

POR QUE ESTE ARQUIVO EXISTE
---------------------------
Corrigir bug a bug não escala: o bug 31 (`KeyError: 'means'`) é o bug 16
(`KeyError: 'monotonic'`) quinze versões depois, em outro módulo. Os dois
vieram da mesma causa — código assumindo a forma de um retorno que tem dois
caminhos.

Os testes aqui não verificam casos. Verificam CLASSES, varrendo a árvore
sintática do pacote inteiro. Nenhum executa modelo: falham em segundos, na
CI, antes de qualquer treino.

AS CLASSES CATALOGADAS
----------------------
  1. resposta plausível onde não havia o que medir   18 de 31 defeitos
  2. limiar absoluto sobre quantidade de escala livre  6 defeitos
  3. chave assumida em retorno de forma variável       bugs 16, 31
  4. implementações paralelas divergindo               bug 28
  5. exceção engolida sem registro                     bug 30

As classes 4 e 5 já têm guarda em test_consistency.py. Este arquivo cobre
a 3 (estrutural) e documenta 1 e 2.
"""
import ast
import importlib
import inspect
import os
import pkgutil
import warnings
import pytest
warnings.filterwarnings('ignore')

import modelcheck

_RAIZ = os.path.dirname(inspect.getfile(modelcheck))


def _modulos():
    """Todos os .py do pacote, exceto experimentais e rascunhos."""
    for arq in sorted(os.listdir(_RAIZ)):
        if not arq.endswith('.py') or arq.startswith('_'):
            continue
        yield arq, os.path.join(_RAIZ, arq)


# ══════════════════════════════════════════════════════════════════════
# CLASSE 3 — chave assumida em retorno de forma variável
# ══════════════════════════════════════════════════════════════════════
#
# `_shape_profile` devolve {shape, means, monotonic}.
# `_shape_profile_small_n` devolve {shape, confidence, method, n, votes}.
#
# Só `shape` existe nos dois. Qualquer outra chave acessada com `[...]`
# quebra quando o perfil vem do caminho de amostra pequena — que foi
# exatamente o bug 31.

CHAVES_NAO_GARANTIDAS = {'means', 'monotonic', 'confidence', 'method',
                         'votes'}
CHAVES_SEGURAS = {'shape'}


def _acessos_arriscados(caminho):
    src = open(caminho, encoding='utf-8').read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    linhas = src.split('\n')
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript)
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            continue
        chave = node.slice.value
        if chave not in CHAVES_NAO_GARANTIDAS:
            continue
        alvo = ast.unparse(node.value)
        # só interessa quando o alvo parece um perfil de forma
        if not any(alvo.startswith(p) for p in ('p', 'perfil', 'prof')):
            continue
        linha = linhas[node.lineno - 1]
        if 'chave-garantida' in linha:      # marcador com justificativa
            continue
        # acesso protegido por verificação explícita nas linhas anteriores
        # (`if 'means' in p:`) não é risco — é o padrão correto.
        contexto = '\n'.join(linhas[max(0, node.lineno - 4):node.lineno])
        if f"'{chave}' in " in contexto or f'"{chave}" in ' in contexto:
            continue
        out.append((node.lineno, f"{alvo}['{chave}']", linha.strip()[:60]))
    return out


@pytest.mark.parametrize('arq,caminho', list(_modulos()))
def test_classe3_chave_de_perfil_nao_e_assumida(arq, caminho):
    """
    Falha se alguém acessar `perfil['means']` (ou outra chave não garantida)
    sem `.get`. O perfil de amostra pequena não tem essas chaves.

    Para acessar mesmo assim, marque a linha com o comentário
    `# chave-garantida: <motivo>` — exigir a justificativa por escrito é o
    que impede o marcador de virar válvula de escape.
    """
    riscos = _acessos_arriscados(caminho)
    assert not riscos, (
        f"{arq}: {len(riscos)} acesso(s) a chave não garantida do perfil de "
        f"forma — quebra quando o perfil vem de `_shape_profile_small_n`, "
        f"como no bug 31. Use `.get()` ou justifique com "
        f"`# chave-garantida: <motivo>`. Ocorrências: {riscos[:3]}")


def test_os_dois_perfis_compartilham_apenas_shape():
    """
    Trava a premissa da guarda acima. Se um dos caminhos passar a devolver
    mais chaves em comum, a lista `CHAVES_SEGURAS` deve ser atualizada —
    e este teste obriga a revisão.
    """
    from modelcheck import shape as _s
    src = inspect.getsource(_s)
    tree = ast.parse(src)
    chaves = {}
    for fn in ('_shape_profile', '_shape_profile_small_n'):
        k = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == fn:
                for r in ast.walk(node):
                    if isinstance(r, ast.Return) and isinstance(r.value, ast.Dict):
                        k |= {c.value for c in r.value.keys
                              if isinstance(c, ast.Constant)}
        chaves[fn] = k
    comuns = chaves['_shape_profile'] & chaves['_shape_profile_small_n']
    assert comuns == CHAVES_SEGURAS, (
        f"as chaves comuns aos dois perfis mudaram: {sorted(comuns)}. "
        f"Atualize CHAVES_SEGURAS e CHAVES_NAO_GARANTIDAS.")


# ══════════════════════════════════════════════════════════════════════
# CLASSE 1 — resposta plausível onde não havia o que medir
# ══════════════════════════════════════════════════════════════════════
#
# 18 dos 31 defeitos. Score 100 para tabela vazia, `nan` atravessando
# comparação, threshold padrão disfarçado de calibrado, aprovação de
# fairness sobre coluna inexistente.
#
# Não dá para detectar por AST — a forma varia demais. O que dá é testar
# a invariante no comportamento: entrada sem informação não pode produzir
# veredito de aprovação.

@pytest.mark.parametrize('caso', ['vazio', 'uma_linha', 'sem_features'])
def test_classe1_entrada_sem_informacao_nao_aprova(caso):
    """
    Entrada degenerada deve levantar erro ou declarar que nada foi medido —
    nunca devolver APROVADO, que o leitor interpreta como "verifiquei e está
    bem".
    """
    import numpy as np, pandas as pd
    import modelcheck as M
    from conftest import RF

    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 300), 'b': rng.uniform(0, 1, 300)})
    y = pd.Series(3 * X.a + rng.randn(300) * .2)
    m1, m2 = RF(0, n=30).fit(X, y), RF(7, n=30).fit(X, y)

    alvo = {'vazio': X.head(0), 'uma_linha': X.head(1),
            'sem_features': X[[]]}[caso]
    try:
        r = M.Guard().audit(m1, m2, alvo, y.head(len(alvo)), list(alvo.columns))
    except (ValueError, TypeError, IndexError):
        return                                  # recusar é a resposta certa
    assert r.verdict != 'APROVADO' or r.not_evaluated, (
        f"entrada '{caso}' produziu APROVADO sem declarar lacuna — é a "
        f"classe de defeito mais frequente do projeto (18 de 31)")


# ══════════════════════════════════════════════════════════════════════
# CLASSE 2 — limiar absoluto sobre quantidade de escala livre
# ══════════════════════════════════════════════════════════════════════
#
# 6 defeitos: PSI com limiar da indústria, cosseno entre médias, MMD sobre
# TF-IDF, limiar de contexto não invariante ao tamanho.
#
# A invariante testável: multiplicar o alvo por uma constante não pode
# mudar o veredito.

def test_classe2_veredito_e_invariante_a_escala_do_alvo():
    """
    Trava a classe: se um limiar absoluto voltar a ser comparado com uma
    quantidade que escala com o alvo, este teste falha.
    """
    import numpy as np, pandas as pd
    import modelcheck as M
    from conftest import RF

    rng = np.random.RandomState(0)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, 600) for i in range(4)})
    base = 3 * X.x0 + 2 * X.x1 + rng.randn(600) * .2

    vereditos = {}
    for k in (1e-3, 1.0, 1e3):
        y = pd.Series(base * k)
        m1 = RF(0, n=40).fit(X, y)
        Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
        m2 = RF(0, n=40).fit(Xc, y)
        vereditos[k] = M.Guard().audit(m1, m2, X, y, list(X.columns)).verdict

    assert len(set(vereditos.values())) == 1, (
        f"o veredito mudou com a escala do alvo: {vereditos}. Algum limiar "
        f"absoluto está sendo comparado com quantidade que escala.")


# ═══ BUG 38: sintaxe que só funciona no Python 3.12 ══════════════════
def test_nenhuma_fstring_quebrada_em_varias_linhas():
    """
    Quebrar uma expressão dentro de f-string em duas linhas só é válido a
    partir do 3.12 (PEP 701). O pacote declara suportar 3.9+, e uma usuária
    no 3.11 recebeu SyntaxError ao importar. O erro não aparece aqui porque
    o container roda 3.12 — por isso o teste inspeciona o código.
    """
    import glob
    import os
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from checa_fstring import problemas
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruins = []
    for pasta in ('modelcheck', 'shapeaudit', 'modelguard'):
        for f in glob.glob(os.path.join(raiz, pasta, '*.py')):
            for ini, fim in problemas(f):
                ruins.append(f'{os.path.basename(f)}:{ini}')
    assert not ruins, f'f-string multilinha (quebra em Python < 3.12): {ruins}'


# ═══ BUG 39: binária rara virava "mudança de escala" ═════════════════
def test_binaria_rara_sem_positivos_na_amostra_nao_alarma():
    """
    No AI4I, `RNF` (0,15% de positivos) aparecia como "dados mudaram" em TODA
    auditoria: a amostra de inferência não pegava nenhum positivo, a variância
    ia a zero e a razão de desvios virava 0 — lido como mudança de escala.
    """
    import numpy as np
    import pandas as pd
    from modelcheck.validation import TrainServingValidator
    r = np.random.RandomState(0); n = 6000
    tr = pd.DataFrame({'rara': (r.rand(n) < .002).astype(float),
                       'x': r.uniform(0, 1, n)})
    sv = pd.DataFrame({'rara': np.zeros(400), 'x': r.uniform(0, 1, 400)})
    assert 'rara' not in TrainServingValidator().check(tr, sv, ['rara', 'x'])['flagged']


def test_mudanca_real_de_taxa_continua_alarmando():
    import numpy as np
    import pandas as pd
    from modelcheck.validation import TrainServingValidator
    r = np.random.RandomState(0); n = 6000
    tr = pd.DataFrame({'rara': (r.rand(n) < .002).astype(float)})
    sv = pd.DataFrame({'rara': (r.rand(2000) < .05).astype(float)})
    assert 'rara' in TrainServingValidator().check(tr, sv, ['rara'])['flagged']


def test_nivel_novo_na_inferencia_alarma():
    """Treino com {0, 3} e inferência com {0, 1}: recodificação, não raridade."""
    import numpy as np
    import pandas as pd
    from modelcheck.validation import TrainServingValidator
    r = np.random.RandomState(0); n = 4000
    a = (r.rand(n) < .02).astype(float)
    tr = pd.DataFrame({'b': a * 3})
    sv = pd.DataFrame({'b': (r.rand(600) < .02).astype(float)})
    assert 'b' in TrainServingValidator().check(tr, sv, ['b'])['flagged']
