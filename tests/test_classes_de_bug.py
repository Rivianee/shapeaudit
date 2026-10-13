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
    no 3.11 recebeu SyntaxError ao importar.

    O detector tem dois caminhos (ver `checa_fstring`): em Python < 3.12 ele
    compila cada arquivo, porque ali a sintaxe é erro de fato; em 3.12+ ele
    tokeniza, porque ali compila sem reclamar. Rodar este teste na versão
    mais antiga que o pacote declara suportar é o que dá a garantia real —
    em 3.12+ a checagem é indicador, não prova.
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
            for ini, _fim, motivo in problemas(f):
                ruins.append(f'{os.path.basename(f)}:{ini} {motivo}')
    assert not ruins, f'sintaxe invalida em Python < 3.12: {ruins}'


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


# ═══ BUG 40: coluna constante virava número complexo ════════════════
def test_coluna_binaria_virando_constante_fora_de_escala():
    """
    O teste de proporção da binária rara era aplicado a qualquer coluna de
    dois níveis, mesmo quando os valores não eram uma codificação 0/1. Uma
    coluna {0,1} que passa a chegar constante em 3.0 fazia `pool` sair de
    [0,1]; `pool * (1 - pool)` ficava negativo e a raiz devolvia um COMPLEXO,
    derrubando a checagem treino × uso com

        TypeError: '>' not supported between 'complex' and 'float'

    Era o pior lugar possível para quebrar: coluna constante é exatamente o
    sintoma de pipeline quebrado que a checagem existe para pegar.
    """
    import numpy as np
    import pandas as pd
    from modelcheck.validation import TrainServingValidator

    rng = np.random.default_rng(0)
    n = 1000
    fe = ['a', 'flag']
    treino = pd.DataFrame({'a': rng.normal(0, 1, n),
                           'flag': (rng.random(n) < 0.02).astype(float)})

    # fora de [0,1]: é recodificação, não mudança de taxa
    for valor in (3.0, 6.5, -1.0, 12.0):
        uso = treino.assign(flag=valor)
        r = TrainServingValidator().check(treino, uso, fe)      # não deve levantar
        assert 'flag' in r['flagged'], f'constante em {valor} passou batido'
        d = [x for x in r['features'] if x['feature'] == 'flag'][0]
        assert 'NEW_LEVEL' in (d.get('alerts') or []), \
            f'constante em {valor} devia virar NEW_LEVEL, veio {d.get("alerts")}'
        assert d.get('z') is None, 'z não deve ser calculado fora da escala 0/1'

    # dentro de [0,1]: aí sim é mudança de taxa
    for valor in (0.0, 1.0):
        r = TrainServingValidator().check(treino, treino.assign(flag=valor), fe)
        assert 'flag' in r['flagged']
        d = [x for x in r['features'] if x['feature'] == 'flag'][0]
        assert isinstance(d.get('z'), float), 'z devia ser numérico aqui'

    # e a correção do bug 39 continua de pé: binária rara saudável não acusa
    uso_ok = treino.assign(flag=(rng.random(n) < 0.02).astype(float))
    assert TrainServingValidator().check(treino, uso_ok, fe)['flagged'] == [], \
        'binária rara saudável voltou a dar falso positivo (regressão do bug 39)'


# ═══ BUG 41: o laudo afirmava um check que não rodou ════════════════
def test_laudo_nao_afirma_check_que_nao_rodou():
    """
    O resumo do laudo mostrava "✓ dados de treino e de uso consistentes"
    mesmo quando a comparação treino × uso não havia acontecido — porque
    `skew_features` vazio significava tanto "checou e está ok" quanto "não
    checou". Num instrumento de decisão isso é o pior tipo de erro: a pessoa
    promove o modelo confiando num visto que não existe.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from modelcheck.api import Guard

    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({'a': rng.normal(0, 1, n), 'b': rng.normal(0, 1, n),
                      'c': rng.integers(1, 13, n).astype(float)})
    y = X.a * 2 + X.c * 0.3 + rng.normal(0, .3, n)
    mk = lambda d: RandomForestRegressor(n_estimators=30, max_depth=5,
                                         random_state=0, n_jobs=1).fit(d, y)
    quebrado = X.assign(c=X.c.mean())
    v1, v2 = mk(X), mk(quebrado)
    fe = list(X.columns)

    # sem X_train a checagem não roda: o laudo tem de dizer isso
    sem = Guard().audit(v1, v2, X, y, fe)
    assert sem.skew_status == 'ausente', sem.skew_status
    # a frase do RESUMO, nao o "nao verificado" da lista de checks
    # individuais, que ja existia e e legitimo
    FRASE = 'dados de treino e de uso: n\u00e3o verificado'
    html = sem.to_html()
    assert FRASE in html, \
        'o resumo devia dizer "nao verificado" quando a checagem nao rodou'
    assert 'dados de treino e de uso consistentes' not in html, \
        'o resumo afirmou "consistentes" sem ter rodado a checagem'

    # com X_train ela roda, e ai pode falar
    com = Guard().audit(v1, v2, X, y, fe, X_train=quebrado)
    assert com.skew_status == 'ok', com.skew_status
    assert FRASE not in com.to_html()
