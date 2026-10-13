"""
Etapa 2 — Robustez a dados malformados ("teste do caos").

Critério: zero exceção não tratada. Toda condição malformada deve gerar
aviso claro, erro com instrução acionável, ou resultado com cautela.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
from modelcheck.audit import audit
from modelcheck.api import Guard as ModelGuard


@pytest.fixture
def base():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.randn(400), 'b': rng.randn(400),
                      'c': rng.randn(400)})
    y = X.a * 3 + rng.randn(400) * .3
    m = RF(0).fit(X, y)
    return X, y, m


# ── NaN em proporções variadas ───────────────────────────────────────
@pytest.mark.parametrize('frac', [0.01, 0.05, 0.20])
def test_nan_gera_aviso_e_nao_quebra(base, frac):
    X, y, m = base
    Xn = X.copy()
    idx = np.random.RandomState(1).choice(len(X), int(len(X) * frac), replace=False)
    Xn.loc[Xn.index[idx], 'b'] = np.nan
    r = audit(m, m, Xn, verbose=False)
    assert any('NAN_PRESENT' in w for w in r['warnings']), \
        f"NaN {frac:.0%} deve gerar aviso NAN_PRESENT"
    assert r['health_score'] is not None


# ── Infinito ─────────────────────────────────────────────────────────
def test_inf_gera_aviso(base):
    X, y, m = base
    Xi = X.copy(); Xi.loc[Xi.index[0], 'a'] = np.inf
    r = audit(m, m, Xi, verbose=False)
    assert any('INF_PRESENT' in w for w in r['warnings'])


# ── Tipos mistos / string em coluna numérica ─────────────────────────
def test_coluna_string_levanta_erro_acionavel(base):
    X, y, m = base
    Xs = X.copy(); Xs['b'] = Xs['b'].astype(str)
    with pytest.raises(TypeError) as e:
        audit(m, m, Xs, verbose=False)
    msg = str(e.value)
    assert 'não numéricas' in msg and 'pd.to_numeric' in msg, \
        "erro deve instruir como corrigir"


# ── Outliers extremos (10× desvio) ───────────────────────────────────
def test_outliers_extremos_nao_quebram(base):
    X, y, m = base
    Xo = X.copy()
    Xo.loc[Xo.index[:5], 'a'] = Xo['a'].std() * 10
    r = audit(m, m, Xo, verbose=False)
    assert isinstance(r['health_score'], (int, float))


# ── n pequeno ────────────────────────────────────────────────────────
@pytest.mark.parametrize('n', [10, 30])
def test_n_pequeno_gera_low_n(base, n):
    X, y, m = base
    r = audit(m, m, X.head(n), verbose=False)
    if n < 30:
        assert any('LOW_N' in w for w in r['warnings']), \
            f"n={n} deve gerar aviso LOW_N"
    assert r['health_score'] is not None


# ── Feature constante ────────────────────────────────────────────────
def test_feature_constante_nao_quebra():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.randn(300), 'const': 1.0})
    y = X.a * 2 + rng.randn(300) * .2
    m = RF(0).fit(X, y)
    r = audit(m, m, X, verbose=False)
    assert r['health_score'] == 100


# ── Feature binária e categórica ─────────────────────────────────────
def test_binaria_e_categorica_nao_quebram():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.randn(400),
                      'bin': (rng.randn(400) > 0).astype(int),
                      'cat': rng.randint(0, 4, 400)})
    y = X.a * 2 + X['bin'] + rng.randn(400) * .2
    m = RF(0).fit(X, y)
    r = audit(m, m, X, verbose=False)
    assert r['health_score'] is not None


# ── Nível categórico novo, não visto no treino ───────────────────────
def test_nivel_categorico_novo():
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame({'a': rng.randn(400), 'cat': rng.randint(0, 3, 400)})
    ytr = Xtr.a * 2 + rng.randn(400) * .2
    m = RF(0).fit(Xtr, ytr)
    Xte = Xtr.copy(); Xte.loc[Xte.index[:20], 'cat'] = 9   # nível inédito
    r = audit(m, m, Xte, verbose=False)
    assert r['health_score'] is not None


# ── Modelos incompatíveis ────────────────────────────────────────────
def test_modelos_incompativeis_levantam_erro(base):
    X, y, m = base
    m1f = RF(0).fit(X[['a']], y)
    with pytest.raises(ValueError) as e:
        audit(m, m1f, X, verbose=False)
    msg = str(e.value).lower()
    # A mensagem mudou na v1.44 (passou a comparar conjunto de nomes).
    # O contrato é: erro explícito mencionando features, não silêncio.
    assert 'features' in msg, f"a mensagem deve citar features: {msg}"
    assert 'não é possível comparar' in msg, \
        f"a mensagem deve dizer que a comparação é impossível: {msg}" 


# ── representation inválida ──────────────────────────────────────────
def test_representation_invalida(base):
    X, y, m = base
    with pytest.raises(ValueError):
        audit(m, m, X, representation='xyz', verbose=False)


# ── check_features com feature inexistente ───────────────────────────
def test_check_features_feature_inexistente(base):
    X, y, m = base
    g = ModelGuard()
    # BUG 89 (v1.0.4): antes devolvia alert=False e o texto "continuam
    # influenciando o modelo" — um visto sobre o que nem foi medido. Agora
    # é erro com a lista de colunas disponíveis.
    with pytest.raises(ValueError, match="não existem"):
        g.check_features(m, m, X, monitored=['nao_existe'],
                         feature_names=list(X.columns))


# ── audit_versions com menos de 2 modelos ────────────────────────────
def test_audit_versions_um_modelo(base):
    X, y, m = base
    g = ModelGuard()
    with pytest.raises(ValueError):
        g.audit_versions([m], X, y, list(X.columns))
