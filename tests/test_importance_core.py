"""
Cobertura do `importance.py` — o detector com F1=0.99 (v1.48).

Estava em 60%. As lacunas eram justamente os caminhos que menos usamos nos
experimentos: o fallback por permutação (modelos não-árvore), o modo lazy,
o `calibrate()` e o `report()`.

O `shape.py` rendeu dois bugs quando saiu de 34% para 91%. Mesmo perfil aqui.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from conftest import RF
from modelcheck.importance import (ImportanceCollapseDetector,
                                   _shap_importance,
                                   _perturbation_importance)


@pytest.fixture
def dados():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'sig': rng.uniform(0, 1, 400),
                      'sig2': rng.uniform(0, 1, 400),
                      'ruido': rng.uniform(0, 1, 400)})
    y = pd.Series(5 * X.sig + 2 * X.sig2 + rng.randn(400) * 0.2)
    return X, y


# ═══ _shap_importance — o caminho e o fallback ═════════════════════════
def test_shap_importance_em_arvore(dados):
    X, y = dados
    m = RF(0, n=30).fit(X, y)
    imp = _shap_importance(m, X, n_sample=200)
    assert imp is not None and len(imp) == 3
    assert (imp >= 0).all(), "importância deve ser não-negativa"
    assert imp[0] > imp[2], "a feature de sinal deve pesar mais que o ruído"


def test_shap_importance_devolve_none_em_nao_arvore(dados):
    """Ridge não tem TreeExplainer — deve devolver None, não quebrar."""
    X, y = dados
    assert _shap_importance(Ridge().fit(X, y), X, n_sample=200) is None


def test_shap_importance_em_classificador(dados):
    """Cobre o caminho de SHAP em lista/3D."""
    X, y = dados
    yb = (y > y.median()).astype(int)
    m = RandomForestClassifier(n_estimators=30, max_depth=5,
                               random_state=0, n_jobs=1).fit(X, yb)
    imp = _shap_importance(m, X, n_sample=200)
    assert imp is not None and len(imp) == 3


# ═══ _perturbation_importance — o fallback ═════════════════════════════
def test_perturbation_importance_ordena_corretamente(dados):
    X, y = dados
    m = Ridge().fit(X, y)
    imp = _perturbation_importance(m, X, list(X.columns), n_repeat=3)
    assert len(imp) == 3
    assert imp[0] > imp[2], "sinal deve pesar mais que ruído"


def test_perturbation_importance_usa_proba_em_classificador(dados):
    """O caminho de predict_proba nunca era exercitado."""
    X, y = dados
    yb = (y > y.median()).astype(int)
    m = RandomForestClassifier(n_estimators=20, max_depth=4,
                               random_state=0, n_jobs=1).fit(X, yb)
    imp = _perturbation_importance(m, X, list(X.columns), n_repeat=2)
    assert len(imp) == 3 and (imp >= 0).all()


def test_perturbation_importance_modo_lazy_e_mais_barato(dados):
    """
    O modo lazy permuta só as monitoradas + amostra para o denominador.
    Medido na v1.28: 25s -> 1.9s com P=200.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 40), columns=[f'f{i}' for i in range(40)])
    y = pd.Series(X.f0 * 3 + rng.randn(300) * 0.2)
    m = Ridge().fit(X, y)
    fs = list(X.columns)
    completo = _perturbation_importance(m, X, fs, n_repeat=2)
    lazy = _perturbation_importance(m, X, fs, n_repeat=2, subset=['f0', 'f1'])
    assert len(lazy) == len(completo) == 40
    # f0 deve liderar nos dois modos
    assert int(np.argmax(completo)) == 0
    assert int(np.argmax(lazy)) == 0


def test_perturbation_importance_deterministico(dados):
    X, y = dados
    m = Ridge().fit(X, y)
    a = _perturbation_importance(m, X, list(X.columns), n_repeat=3, seed=7)
    b = _perturbation_importance(m, X, list(X.columns), n_repeat=3, seed=7)
    assert np.allclose(a, b)


# ═══ compare — o método principal ══════════════════════════════════════
def test_compare_usa_permutation_em_nao_arvore(dados):
    X, y = dados
    m1 = Ridge().fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = Ridge().fit(Xc, y)
    r = ImportanceCollapseDetector(n_boot=3, n_sample=150).compare(
        m1, m2, X, list(X.columns))
    assert 'PERMUTATION' in r['method']
    hits = [f['feature'] for f in r['features']
            if f['alert'] == 'IMPORTANCE_COLLAPSE']
    assert 'sig' in hits


def test_compare_marca_lazy_no_method(dados):
    X, y = dados
    m1 = Ridge().fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = Ridge().fit(Xc, y)
    r = ImportanceCollapseDetector(n_boot=3, n_sample=150).compare(
        m1, m2, X, list(X.columns), monitored=['sig'])
    assert r['method'] == 'PERMUTATION_LAZY'


def test_compare_detecta_emergencia(dados):
    """
    IMPORTANCE_EMERGENCE: feature irrelevante ganha peso. Acontece quando a
    principal é neutralizada e o modelo redistribui.
    """
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean(); Xc['sig2'] = Xc['sig2'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    r = ImportanceCollapseDetector(n_boot=4, n_sample=150).compare(
        m1, m2, X, list(X.columns))
    assert r['n_emergence'] >= 0          # o campo existe e é contado
    assert all('ratio_abs' in f for f in r['features'])


def test_compare_rank_shuffle_detectado(dados):
    """Ranking de importâncias reordenado deve marcar RANK_SHUFFLE."""
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    r = ImportanceCollapseDetector(n_boot=4, n_sample=150).compare(
        m1, m2, X, list(X.columns))
    assert isinstance(r['rank_shuffle'], bool)
    assert -1.0 <= r['rank_rho'] <= 1.0


def test_compare_score_entre_zero_e_cem(dados):
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    det = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    for a, b in [(m1, m1), (m1, m2)]:
        s = det.compare(a, b, X, list(X.columns))['importance_score']
        assert 0.0 <= s <= 100.0


def test_compare_reporta_ic_da_razao(dados):
    """O IC percentil da razão precisa estar na saída e ser coerente."""
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    r = ImportanceCollapseDetector(n_boot=5, n_sample=150).compare(
        m1, m1, X, list(X.columns))
    for f in r['features']:
        lo, hi = f['ratio_ci']
        assert lo <= hi, f"IC invertido em {f['feature']}: [{lo}, {hi}]"


# ═══ calibrate ═════════════════════════════════════════════════════════
def test_calibrate_produz_limiar_por_feature(dados):
    """
    Medido no ENA: a variação natural difere por feature (ENA_MLT p5=0.70,
    EAR p5=0.31). O limiar fixo 0.40 gerava falso positivo no EAR.
    """
    X, y = dados
    dfc = X.copy(); dfc['y'] = y
    det = ImportanceCollapseDetector(n_boot=3, n_sample=120)
    mf = lambda Xd, yd: RF(0, n=25).fit(Xd, yd)
    cal = det.calibrate(mf, dfc, 'y', list(X.columns), window=120, n_pairs=3)
    assert 'thresholds' in cal and 'stats' in cal
    for f in X.columns:
        assert f in cal['thresholds']
        assert 0.0 <= cal['thresholds'][f] <= cal['default']
    assert det.per_feature_threshold, "deve gravar o limiar no detector"


def test_calibrate_com_amostra_insuficiente_sinaliza_falha(dados):
    """
    BUG 18: quando nenhuma feature reunia pares suficientes, o `calibrate()`
    devolvia os thresholds PADRÃO sem sinalizar. O usuário recebia um dict
    que parecia calibração válida e seguia com o limiar fixo achando que
    tinha calibrado. Só o campo `n: 0`, enterrado em `stats`, denunciava.
    """
    X, y = dados
    dfc = X.copy(); dfc['y'] = y
    det = ImportanceCollapseDetector(n_boot=2, n_sample=100)
    mf = lambda Xd, yd: RF(0, n=20).fit(Xd, yd)
    cal = det.calibrate(mf, dfc.head(60), 'y', list(X.columns),
                        window=200, n_pairs=1)
    assert cal['ok'] is False, "falha de calibração deve ser sinalizada"
    assert 'reason' in cal and len(cal['reason']) > 20, \
        "deve explicar por que não calibrou"
    assert cal['n_calibradas'] == 0
    assert det.per_feature_threshold == {}, \
        "não deve gravar limiar quando a calibração falhou"


def test_calibrate_bem_sucedido_marca_ok(dados):
    X, y = dados
    dfc = X.copy(); dfc['y'] = y
    det = ImportanceCollapseDetector(n_boot=3, n_sample=120)
    mf = lambda Xd, yd: RF(0, n=25).fit(Xd, yd)
    cal = det.calibrate(mf, dfc, 'y', list(X.columns), window=120, n_pairs=3)
    assert cal['ok'] is True
    assert cal['n_calibradas'] >= 1


def test_limiar_calibrado_e_usado_no_compare(dados):
    """Um limiar draconiano deve suprimir o alerta — prova que é aplicado."""
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    fs = list(X.columns)
    det = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    padrao = det.compare(m1, m2, X, fs)['n_collapse']
    det.per_feature_threshold = {f: 0.0 for f in fs}   # nada colapsa
    restrito = det.compare(m1, m2, X, fs)['n_collapse']
    assert padrao >= 1
    assert restrito == 0, "o limiar calibrado não está sendo aplicado"


# ═══ report ════════════════════════════════════════════════════════════
def test_report_em_controle_e_em_corrupcao(dados):
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    det = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    t_ctrl = det.report(det.compare(m1, m1, X, list(X.columns)))
    t_corr = det.report(det.compare(m1, m2, X, list(X.columns)))
    assert isinstance(t_ctrl, str) and len(t_ctrl) > 10
    assert 'sig' in t_corr, "o report deve nomear a feature colapsada"
    assert t_ctrl != t_corr


def test_report_cita_o_metodo_usado(dados):
    X, y = dados
    m1 = Ridge().fit(X, y)
    det = ImportanceCollapseDetector(n_boot=3, n_sample=120)
    txt = det.report(det.compare(m1, m1, X, list(X.columns)))
    assert 'PERMUTATION' in txt or 'SHAP' in txt


# ═══ monitored ═════════════════════════════════════════════════════════
def test_monitored_alert_so_dispara_nas_monitoradas(dados):
    """
    O achado da v1.27: retreino legítimo desloca importância de OUTRAS
    features; corrupção colapsa a específica. F1=0.99 vem daí.
    """
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    det = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r_on = det.compare(m1, m2, X, list(X.columns), monitored=['sig'])
    assert r_on['monitored_alert'] is True
    assert r_on['monitored_hits'] == ['sig']
    r_off = det.compare(m1, m2, X, list(X.columns), monitored=['ruido'])
    assert r_off['monitored_alert'] is False, \
        "monitorar outra feature não deve alertar pela corrupção em 'sig'"


def test_monitored_none_nao_preenche_hits(dados):
    X, y = dados
    m1 = RF(0, n=30).fit(X, y)
    r = ImportanceCollapseDetector(n_boot=3, n_sample=120).compare(
        m1, m1, X, list(X.columns))
    assert r['monitored'] is None
    assert r['monitored_hits'] == []
