"""
Testes de PROPRIEDADE (v1.44).

Não testam cenários — testam invariantes que todo detector deve respeitar.
Foi essa classe de teste que encontrou os bugs 9, 11, 12, 13 e 14.

Quatro categorias que não estavam cobertas antes desta versão:
  1. invariância a transformação monotônica das FEATURES
  2. colinearidade — o detector aponta a feature certa?
  3. modelo degenerado (constante) — diagnóstico específico, não genérico
  4. alinhamento por NOME e não por índice de coluna
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
from modelcheck.audit import audit
from modelcheck.importance import ImportanceCollapseDetector


# ═══ 1. Transformação monotônica das features ════════════════════════
@pytest.mark.parametrize('nome,tf', [
    ('identidade', lambda v: v),
    ('log',        np.log),
    ('sqrt',       np.sqrt),
    ('x100',       lambda v: v * 100),
])
def test_invariancia_a_transformacao_monotonica_da_feature(nome, tf):
    """
    Se os dois modelos treinam na MESMA transformação da feature, não há
    divergência real — o score deve ser 100 em qualquer transformação
    monotônica. Se variasse, teríamos falso positivo por unidade de feature.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(1, 10, 500),
                      'b': rng.uniform(1, 10, 500)})
    y = pd.Series(3 * np.log(X.a) + 2 * X.b + rng.randn(500) * 0.2)
    Xt = X.copy(); Xt['a'] = tf(Xt['a'].values)
    m1 = RF(0, n=40).fit(Xt, y)
    r = audit(m1, m1, Xt, verbose=False)
    assert r['health_score'] == 100, \
        f"transformação '{nome}' gerou divergência falsa: {r['divergences']}"


# ═══ 2. Colinearidade ════════════════════════════════════════════════
def test_colinearidade_aponta_a_feature_corrompida():
    """
    x1 e x1_copia com rho≈0.998. Corrompemos APENAS x1. O detector deve
    apontar x1, não a cópia — senão o engenheiro investiga o lugar errado.
    """
    rng = np.random.RandomState(0)
    x1 = rng.uniform(0, 1, 600)
    X = pd.DataFrame({'x1': x1,
                      'x1_copia': x1 + rng.randn(600) * 0.02,
                      'x2': rng.uniform(0, 1, 600)})
    y = pd.Series(3 * X.x1 + 1.5 * X.x2 + rng.randn(600) * 0.15)
    rho = abs(np.corrcoef(X.x1, X.x1_copia)[0, 1])
    assert rho > 0.99, f"o setup precisa de alta correlação, obtido {rho:.3f}"

    m1 = RF(0, n=40).fit(X, y)
    Xn = X.copy(); Xn['x1'] = Xn['x1'].mean()
    m2 = RF(0, n=40).fit(Xn, y)

    icd = ImportanceCollapseDetector(n_boot=5, n_sample=200)
    r = icd.compare(m1, m2, X, list(X.columns))
    hits = [x['feature'] for x in r['features']
            if x['alert'] == 'IMPORTANCE_COLLAPSE']
    assert 'x1' in hits, f"deve apontar x1 (a corrompida): {hits}"
    assert 'x1_copia' not in hits, \
        f"não deve apontar a cópia como colapsada: {hits}"


# ═══ 3. Modelo degenerado ════════════════════════════════════════════
class _ModeloConstante:
    """Modelo que sempre prevê o mesmo valor — colapso total."""

    def __init__(self, valor, n_features):
        self.valor = float(valor)
        self.n_features_in_ = int(n_features)

    def predict(self, X):
        return np.full(len(X), self.valor)


def test_modelo_degenerado_gera_diagnostico_especifico():
    """
    Um modelo que virou constante é COLAPSO, não mudança de forma. Antes da
    v1.44 o diagnóstico era "divergência de forma em a e b" — o engenheiro
    investigaria as features quando o problema é o modelo inteiro.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    y = pd.Series(4 * X.a + rng.randn(400) * 0.2)
    m1 = RF(0, n=30).fit(X, y)
    m_const = _ModeloConstante(y.mean(), X.shape[1])

    r = audit(m1, m_const, X, verbose=False)
    assert r['model_collapsed'] is True
    assert any('MODEL_COLLAPSED' in w for w in r['warnings']), \
        f"deve avisar sobre colapso: {r['warnings']}"


def test_modelo_normal_nao_e_marcado_como_degenerado():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    y = pd.Series(4 * X.a + rng.randn(400) * 0.2)
    m1 = RF(0, n=30).fit(X, y)
    r = audit(m1, m1, X, verbose=False)
    assert r['model_collapsed'] is False
    assert r['health_score'] == 100


def test_ambos_degenerados():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 300), 'b': rng.uniform(0, 1, 300)})
    c1 = _ModeloConstante(1.0, 2); c2 = _ModeloConstante(2.0, 2)
    r = audit(c1, c2, X, verbose=False)
    assert any('BOTH_MODELS_COLLAPSED' in w for w in r['warnings'])


# ═══ 4. Alinhamento por nome, não por índice ══════════════════════════
@pytest.mark.parametrize('ordem', [
    ['col1', 'col2', 'col3'],
    ['col3', 'col2', 'col1'],
    ['col2', 'col3', 'col1'],
])
def test_alinhamento_por_nome_nao_por_indice(ordem):
    """
    Se o detector comparasse por POSIÇÃO, reordenar as colunas de X_eval
    faria 'col1' de v1 ser comparada com 'col3' de v2 — divergência
    inteiramente artificial. Em produção, pipelines de feature engineering
    reordenam colunas com frequência.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'col1': rng.uniform(0, 1, 500),
                      'col2': rng.uniform(0, 1, 500),
                      'col3': rng.uniform(0, 1, 500)})
    y = pd.Series(3 * X.col1 + 2 * X.col2 + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X[['col1', 'col2', 'col3']], y)
    r = audit(m1, m1, X[ordem], verbose=False)
    assert r['health_score'] == 100, \
        (f"ordem {ordem} gerou divergência falsa — possível comparação por "
         f"índice: {r['divergences']}")
