import pytest
"""Detector por distância de Fréchet (v0.10.0)."""
import inspect, warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
import modelcheck as M
from modelcheck.frechet import distancia, perfil, divergencias_forma


def _par(forma2, seed=0, n=1500):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, n) for i in range(4)})
    z = lambda v: (v - v.mean()) / v.std()
    f = {'lin': lambda x: x, 'u_inv': lambda x: -(x - .5) ** 2}
    y1 = pd.Series(z(X.x0) + .5 * z(X.x1) + rng.randn(n) * .2)
    y2 = pd.Series(z(f[forma2](X.x0)) + .5 * z(X.x1) + rng.randn(n) * .2)
    return X, y1, y2


def test_distancia_invariante_a_escala():
    a = np.array([0, 1, 2, 3, 2, 1.])
    assert distancia(a, a * 1000) == pytest.approx(0, abs=1e-9)
    assert distancia(a, a + 50) == pytest.approx(0, abs=1e-9)


def test_detecta_forma_com_importancia_preservada():
    X, y1, y2 = _par('u_inv')
    r = M.Guard().audit(RF(0, n=60).fit(X, y1), RF(0, n=60).fit(X, y2), X, y1,
                        list(X.columns))
    divs = [d['feature'] for d in r.raw['layer1']['sdd']['divs']]
    assert 'x0' in divs


def test_retreino_saudavel_nao_dispara():
    X, y1, _ = _par('lin')
    r = M.Guard().audit(RF(0, n=60).fit(X, y1), RF(7, n=60).fit(X, y1), X, y1,
                        list(X.columns))
    assert r.raw['layer1']['sdd']['divs'] == []


def test_variavel_fraca_nao_e_avaliada():
    """
    Peso < 5%: medido em retreinos saudáveis, 77% das variáveis com peso < 2%
    passavam do limiar só por acaso.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 500)})
    sv1 = np.c_[rng.randn(500)]; sv2 = np.c_[rng.randn(500)]
    Xf = X.assign(b=rng.uniform(0, 1, 500))
    s1 = np.c_[sv1 * .01, X.a * 10]; s2 = np.c_[-sv2 * .01, X.a * 10]
    assert divergencias_forma(Xf[['a', 'b']].rename(columns={'a': 'fraca', 'b': 'forte'}),
                              s1, s2, ['fraca', 'forte']) == []


def test_uma_so_implementacao():
    """Classe 4 do catálogo: o orquestrador não pode reimplementar o cálculo."""
    from modelcheck import orchestrator
    import ast
    src = inspect.getsource(orchestrator._run_layer1)
    assert 'divergencias_forma' in src
    codigo = ast.unparse(ast.parse(src))        # sem comentários
    assert 'spread >= threshold' not in codigo


def test_mesma_categoria_e_reportada():
    X, y1, _ = _par('lin')
    rng = np.random.RandomState(3)
    y2 = pd.Series((1 - np.exp(-4 * X.x0)) * 3 + .5 * X.x1 + rng.randn(len(X)) * .1)
    m1, m2 = RF(0, n=60).fit(X, y1), RF(0, n=60).fit(X, y2)
    r = M.Guard().audit(m1, m2, X, y1, list(X.columns))
    for d in r.raw['layer1']['sdd']['divs']:
        assert 'mesma_categoria' in d and 'frechet' in d


def test_calibrar_limiar_no_dominio():
    """v0.12.0: o limiar padrão veio de uma bancada sintética; medido em três
    domínios, o calibrado varia de 0.217 a 0.471."""
    from modelcheck.frechet import calibrar_limiar
    X, y1, _ = _par('lin')
    pares = [(RF(s, n=40).fit(X, y1), RF(s + 50, n=40).fit(X, y1)) for s in range(5)]
    r = calibrar_limiar(pares, X, list(X.columns))
    assert r['limiar'] is not None and 0 < r['limiar'] < 3
    assert r['n_distancias'] >= 5


def test_calibrar_com_poucos_pares_declara():
    from modelcheck.frechet import calibrar_limiar
    X, y1, _ = _par('lin')
    # BUG 90 (v1.0.4): devolvia limiar=None, e o fluxo do README
    # (Guard(frechet_threshold=cal['limiar'])) caía no padrão sem avisar.
    with pytest.raises(ValueError, match="poucos pares"):
        calibrar_limiar([], X, list(X.columns))
