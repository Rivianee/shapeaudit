"""
Cobertura do classificador de formas — o componente mais central (v1.46).

`shape.py` tinha 34% de cobertura: dois terços do classificador nunca
exercitados. Os bugs 9 e 13 estavam exatamente nesse tipo de trecho.

Estes testes cobrem, com formas de ground truth conhecido:
  - as 5 classes da taxonomia (crescente, decrescente, u, u_invertido, irregular)
  - os 3 métodos do ensemble small_n (regressão, bootstrap, spearman split)
  - `_adaptive_bins` em toda a faixa
  - fronteiras: n=0, 1, 2, min_n, feature constante, SHAP constante
"""
import warnings, numpy as np, pytest
warnings.filterwarnings('ignore')
from modelcheck.shape import (_adaptive_bins, _shape_profile,
                              _shape_profile_small_n)

FORMAS = ('crescente', 'decrescente', 'u', 'u_invertido', 'irregular')


# ═══ _adaptive_bins — faixa completa ══════════════════════════════════
@pytest.mark.parametrize('n,esperado', [
    (0, 5), (39, 5), (40, 5), (200, 5),
    (240, 6), (280, 7), (10_000, 7),
])
def test_adaptive_bins_respeita_limites(n, esperado):
    """Deve ficar sempre em [5, 7], independente do n."""
    b = _adaptive_bins(n)
    assert 5 <= b <= 7
    assert b == esperado, f"n={n}: esperado {esperado}, obtido {b}"


# ═══ Taxonomia: as 5 classes com ground truth ═════════════════════════
def _curva(tipo, n=200, ruido=0.02, seed=0):
    rng = np.random.RandomState(seed)
    x = np.linspace(-1, 1, n)
    base = {
        'crescente':   3 * x,
        'decrescente': -3 * x,
        'u':           3 * x ** 2,
        'u_invertido': -3 * x ** 2,
    }[tipo]
    return x, base + rng.randn(n) * ruido


@pytest.mark.parametrize('tipo', ['crescente', 'decrescente', 'u', 'u_invertido'])
def test_shape_profile_classifica_as_4_formas_inequivocas(tipo):
    x, y = _curva(tipo)
    p = _shape_profile(x, y)
    assert p is not None, f"{tipo}: devolveu None"
    assert p['shape'] == tipo, f"esperado '{tipo}', obtido '{p['shape']}'"


@pytest.mark.parametrize('tipo', ['crescente', 'decrescente', 'u', 'u_invertido'])
def test_small_n_classifica_as_4_formas(tipo):
    """O ensemble de 3 métodos deve acertar formas claras com n=20."""
    x, y = _curva(tipo, n=20, ruido=0.01)
    p = _shape_profile_small_n(x, y, min_n=8)
    assert p is not None
    assert p['shape'] == tipo, f"esperado '{tipo}', obtido '{p['shape']}'"
    assert 0.0 < p['confidence'] <= 1.0
    assert p['method'] == 'small_n_ensemble'
    assert len(p['votes']) >= 1


def test_shape_profile_ruido_puro_nao_e_monotonico():
    """Ruído sem estrutura não deve ser classificado como monotônico."""
    rng = np.random.RandomState(0)
    x = np.linspace(0, 1, 200)
    p = _shape_profile(x, rng.randn(200))
    assert p is not None
    assert p['shape'] in FORMAS
    assert p['shape'] == 'irregular' or not p.get('monotonic', False), \
        f"ruído classificado como '{p['shape']}' com monotonic={p.get('monotonic')}"


def test_shape_profile_retorna_estrutura_completa():
    x, y = _curva('crescente')
    p = _shape_profile(x, y)
    for k in ('means', 'monotonic', 'shape'):
        assert k in p, f"chave '{k}' ausente"
    assert isinstance(p['means'], (list, np.ndarray))
    assert isinstance(p['monotonic'], (bool, np.bool_))


# ═══ Fronteiras extremas: n=0, 1, 2 ═══════════════════════════════════
@pytest.mark.parametrize('n', [0, 1, 2, 3, 5, 7])
def test_shape_profile_n_extremo_devolve_none_sem_quebrar(n):
    """n abaixo do mínimo deve devolver None, nunca levantar exceção."""
    rng = np.random.RandomState(0)
    x = rng.randn(n) if n else np.array([])
    y = rng.randn(n) if n else np.array([])
    assert _shape_profile(x, y) is None


@pytest.mark.parametrize('n', [0, 1, 2, 7])
def test_small_n_abaixo_do_minimo_devolve_none(n):
    rng = np.random.RandomState(0)
    x = rng.randn(n) if n else np.array([])
    y = rng.randn(n) if n else np.array([])
    assert _shape_profile_small_n(x, y, min_n=8) is None


def test_small_n_exatamente_no_minimo():
    """n == min_n deve funcionar (fronteira inclusiva)."""
    x = np.linspace(0, 1, 8)
    p = _shape_profile_small_n(x, 3 * x, min_n=8)
    assert p is not None and p['n'] == 8


# ═══ Degenerados ══════════════════════════════════════════════════════
def test_feature_constante_devolve_none():
    x = np.ones(100)
    y = np.random.RandomState(0).randn(100)
    assert _shape_profile(x, y) is None
    assert _shape_profile_small_n(x, y, min_n=8) is None


def test_shap_constante_nao_quebra():
    """SHAP todo zero (modelo colapsado) não deve levantar exceção."""
    x = np.linspace(0, 1, 100)
    y = np.zeros(100)
    p = _shape_profile(x, y)
    assert p is None or p['shape'] in FORMAS


def test_nan_nos_dados_nao_quebra():
    x = np.linspace(0, 1, 100)
    y = np.random.RandomState(0).randn(100)
    y[:10] = np.nan
    p = _shape_profile(x, y)
    assert p is None or p['shape'] in FORMAS


def test_inf_nos_dados_nao_quebra():
    x = np.linspace(0, 1, 100)
    y = np.random.RandomState(0).randn(100)
    y[0] = np.inf
    p = _shape_profile(x, y)
    assert p is None or p['shape'] in FORMAS


def test_valores_duplicados_em_x():
    """Muitos valores repetidos quebram bins quantílicos."""
    x = np.repeat([0.0, 1.0], 50)
    y = np.concatenate([np.zeros(50), np.ones(50)])
    p = _shape_profile(x, y)
    assert p is None or p['shape'] in FORMAS


# ═══ Propriedades do classificador ════════════════════════════════════
@pytest.mark.parametrize('tipo', ['crescente', 'u_invertido'])
def test_classificacao_invariante_a_escala_do_shap(tipo):
    """Multiplicar os SHAP por constante não muda a forma."""
    x, y = _curva(tipo)
    formas = {_shape_profile(x, y * k)['shape'] for k in (1e-3, 1.0, 1e3)}
    assert len(formas) == 1, f"{tipo}: forma varia com escala do SHAP: {formas}"


@pytest.mark.parametrize('tipo', ['crescente', 'u'])
def test_classificacao_invariante_a_escala_da_feature(tipo):
    """Multiplicar x por constante não muda a forma."""
    x, y = _curva(tipo)
    formas = {_shape_profile(x * k, y)['shape'] for k in (1e-3, 1.0, 1e3)}
    assert len(formas) == 1, f"{tipo}: forma varia com escala de x: {formas}"


def test_inverter_o_sinal_do_shap_inverte_a_forma():
    """crescente ↔ decrescente e u ↔ u_invertido sob negação."""
    for a, b in [('crescente', 'decrescente'), ('u', 'u_invertido')]:
        x, y = _curva(a)
        assert _shape_profile(x, y)['shape'] == a
        assert _shape_profile(x, -y)['shape'] == b, \
            f"negar '{a}' deveria dar '{b}'"


def test_classificacao_deterministica():
    x, y = _curva('u_invertido')
    formas = {_shape_profile(x, y)['shape'] for _ in range(5)}
    assert len(formas) == 1


def test_small_n_deterministico():
    """O ensemble usa bootstrap — precisa ser determinístico."""
    x, y = _curva('crescente', n=20)
    saidas = {(_shape_profile_small_n(x, y, min_n=8)['shape'],
               _shape_profile_small_n(x, y, min_n=8)['confidence'])
              for _ in range(5)}
    assert len(saidas) == 1, f"small_n não determinístico: {saidas}"


def test_n_bins_explicito_respeitado():
    x, y = _curva('crescente', n=300)
    for nb in (3, 5, 8):
        p = _shape_profile(x, y, n_bins=nb)
        assert p is not None
        assert len(p['means']) <= nb


def test_confianca_maior_para_forma_mais_clara():
    """Curva limpa deve ter confiança >= curva ruidosa."""
    x, y_limpo = _curva('crescente', n=20, ruido=0.005)
    _, y_ruido = _curva('crescente', n=20, ruido=1.5)
    c_limpo = _shape_profile_small_n(x, y_limpo, min_n=8)['confidence']
    p_ruido = _shape_profile_small_n(x, y_ruido, min_n=8)
    if p_ruido is not None:
        assert c_limpo >= p_ruido['confidence']


# ═══ ShapeDivergenceDetector — a classe pública ════════════════════════
import pandas as pd
from modelcheck.shape import ShapeDivergenceDetector


def _shap_de(tipo, n=200, seed=0):
    """Matriz SHAP sintética: coluna 0 com a forma pedida, coluna 1 ruído."""
    rng = np.random.RandomState(seed)
    x = np.linspace(-1, 1, n)
    base = {'crescente': 3 * x, 'decrescente': -3 * x,
            'u': 3 * x ** 2, 'u_invertido': -3 * x ** 2}[tipo]
    sv = np.column_stack([base + rng.randn(n) * 0.02, rng.randn(n) * 0.01])
    X = pd.DataFrame({'f0': x, 'f1': rng.randn(n)})
    return sv, X


def test_detector_modelos_identicos_sem_divergencia():
    sv, X = _shap_de('crescente')
    r = ShapeDivergenceDetector().check({'v1': sv, 'v2': sv.copy()}, X)
    assert isinstance(r, (list, dict))
    txt = str(r)
    assert 'f0' not in txt or 'SHAPE_DIV' not in txt, \
        f"modelos idênticos não devem divergir: {txt[:200]}"


def test_detector_formas_opostas_geram_divergencia():
    """
    crescente vs decrescente na mesma feature: divergência inequívoca.

    Nota (v1.60): o filtro tau, alinhado com o `audit()` pelo bug 28, exige
    espalhamento de importância. Aqui as duas curvas têm magnitude idêntica
    (só o sinal muda), então tau=0.30 bloqueia. Este é precisamente o caso
    de Instabilidade de Forma pura documentado no Benchmark B, onde o
    critério tau detecta 6%. Testamos com tau desativado, que é a
    operacionalização da Definição 3.
    """
    sv1, X = _shap_de('crescente')
    sv2, _ = _shap_de('decrescente')
    r = ShapeDivergenceDetector(tau=None).check({'v1': sv1, 'v2': sv2}, X)
    assert 'f0' in str(r), f"f0 com formas opostas deve aparecer: {str(r)[:200]}"


def test_tau_bloqueia_forma_pura_comportamento_documentado():
    """
    Trava o comportamento medido: com importância preservada, tau bloqueia.
    Se isso mudar, o Benchmark B do artigo precisa ser refeito.
    """
    sv1, X = _shap_de('crescente')
    sv2, _ = _shap_de('decrescente')
    com_tau = ShapeDivergenceDetector(tau=0.30).check({'v1': sv1, 'v2': sv2}, X)
    sem_tau = ShapeDivergenceDetector(tau=None).check({'v1': sv1, 'v2': sv2}, X)
    assert 'f0' not in str(com_tau)
    assert 'f0' in str(sem_tau)


def test_detector_ignora_feature_de_peso_baixo():
    """min_weight deve filtrar features com atribuição negligível."""
    rng = np.random.RandomState(0)
    x = np.linspace(-1, 1, 200)
    # f1 tem SHAP 1000x menor que f0
    sv1 = np.column_stack([3 * x, rng.randn(200) * 1e-4])
    sv2 = np.column_stack([3 * x, -rng.randn(200) * 1e-4])
    X = pd.DataFrame({'f0': x, 'f1': rng.randn(200)})
    r = ShapeDivergenceDetector(min_weight=0.03).check({'v1': sv1, 'v2': sv2}, X)
    assert 'f1' not in str(r), f"feature de peso ~0 não deve divergir: {str(r)[:200]}"


def test_detector_com_tres_modelos():
    sv1, X = _shap_de('crescente')
    sv2, _ = _shap_de('decrescente')
    sv3, _ = _shap_de('u')
    r = ShapeDivergenceDetector().check({'a': sv1, 'b': sv2, 'c': sv3}, X)
    assert r is not None


def test_detector_um_unico_modelo_nao_quebra():
    sv, X = _shap_de('crescente')
    r = ShapeDivergenceDetector().check({'só_um': sv}, X)
    assert r is not None


def test_detector_dict_vazio_nao_quebra():
    _, X = _shap_de('crescente')
    try:
        r = ShapeDivergenceDetector().check({}, X)
        assert r is not None or r == [] or r == {}
    except (ValueError, KeyError):
        pass          # erro explícito também é aceitável


def test_detector_per_bin_test_ativado():
    """O caminho com teste estatístico por bin nunca era exercitado."""
    sv1, X = _shap_de('crescente')
    sv2, _ = _shap_de('decrescente')
    r = ShapeDivergenceDetector(per_bin_test=True, alpha=0.05).check(
        {'v1': sv1, 'v2': sv2}, X)
    assert r is not None


def test_detector_aceita_numpy_e_dataframe():
    sv, X = _shap_de('crescente')
    sv2, _ = _shap_de('u')
    r_df = ShapeDivergenceDetector().check({'v1': sv, 'v2': sv2}, X)
    r_np = ShapeDivergenceDetector().check({'v1': sv, 'v2': sv2}, X.values)
    assert r_df is not None and r_np is not None


def test_detector_deterministico():
    sv1, X = _shap_de('crescente')
    sv2, _ = _shap_de('u_invertido')
    saidas = {str(ShapeDivergenceDetector().check({'v1': sv1, 'v2': sv2}, X))
              for _ in range(3)}
    assert len(saidas) == 1, "detector não determinístico"


@pytest.mark.parametrize('n', [10, 30, 100])
def test_detector_com_n_pequeno(n):
    sv1, X = _shap_de('crescente', n=n)
    sv2, _ = _shap_de('decrescente', n=n)
    r = ShapeDivergenceDetector().check({'v1': sv1, 'v2': sv2}, X)
    assert r is not None
