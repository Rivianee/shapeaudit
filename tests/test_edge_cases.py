"""
Cobertura de casos de borda nos módulos críticos (v1.40).

Foco nos caminhos nunca exercitados: n mínimo, features constantes,
todas as features irrelevantes, grupos degenerados, entradas vazias.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF, _split
from modelcheck.audit import audit


# ── shape.py ─────────────────────────────────────────────────────────
def test_shape_profile_feature_constante():
    from modelcheck.shape import _shape_profile
    x = np.ones(50); v = np.random.RandomState(0).randn(50)
    assert _shape_profile(x, v) is None, "feature constante deve dar None"


def test_shape_profile_n_insuficiente():
    from modelcheck.shape import _shape_profile
    rng = np.random.RandomState(0)
    assert _shape_profile(rng.randn(5), rng.randn(5)) is None


def test_shape_profile_small_n_funciona_com_poucos_pontos():
    from modelcheck.shape import _shape_profile_small_n
    x = np.linspace(0, 1, 10); v = 3 * x        # crescente inequívoco
    p = _shape_profile_small_n(x, v, min_n=3)
    assert p is not None and p['shape'] == 'crescente'


def test_shape_profile_small_n_rejeita_menos_que_min():
    from modelcheck.shape import _shape_profile_small_n
    x = np.array([0.0, 1.0]); v = np.array([0.0, 1.0])
    assert _shape_profile_small_n(x, v, min_n=3) is None


# ── slope.py ─────────────────────────────────────────────────────────
def test_slope_com_todas_as_features_constantes():
    """Nenhuma feature tem variância: não deve quebrar."""
    import shap
    from modelcheck.slope import ShapeSlopeDetector
    X = pd.DataFrame({'a': np.ones(200), 'b': np.ones(200) * 2})
    y = pd.Series(np.random.RandomState(0).randn(200))
    m = RF(0, n=20).fit(X, y)
    sv = shap.TreeExplainer(m).shap_values(X)
    sv = sv[:, :, 1] if sv.ndim == 3 else sv
    res = ShapeSlopeDetector().check(sv, sv, X, list(X.columns))
    assert isinstance(res, list)


def test_slope_ratio_zero_gera_collapsed():
    """Reforça a correção do bug 10 diretamente na estrutura de saída."""
    import shap
    from modelcheck.slope import ShapeSlopeDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    y = pd.Series(5 * X.a + rng.randn(400) * 0.1)
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=30).fit(Xc, y)

    def sv(m):
        s = shap.TreeExplainer(m).shap_values(X)
        return s[:, :, 1] if s.ndim == 3 else s

    res = ShapeSlopeDetector(threshold_ratio=3.0).check(
        sv(m1), sv(m2), X, list(X.columns))
    assert all('ratio' in r for r in res)


# ── importance.py ────────────────────────────────────────────────────
def test_icd_todas_as_features_irrelevantes():
    """Target puro ruído: nenhuma feature importa, não deve alertar."""
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(400, 4), columns=list('abcd'))
    y = pd.Series(rng.randn(400))          # sem relação nenhuma
    m1 = RF(0, n=30).fit(X, y); m2 = RF(1, n=30).fit(X, y)
    icd = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r = icd.compare(m1, m2, X, list(X.columns))
    assert r['n_collapse'] == 0, \
        "sem sinal real, não deve haver colapso"


def test_icd_monitored_vazio():
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(300, 3), columns=list('abc'))
    y = pd.Series(X.a * 2 + rng.randn(300) * 0.2)
    m = RF(0, n=20).fit(X, y)
    icd = ImportanceCollapseDetector(n_boot=3, n_sample=120)
    r = icd.compare(m, m, X, list(X.columns), monitored=[])
    assert r['monitored_alert'] is False
    assert r['monitored_hits'] == []


def test_icd_uma_unica_feature():
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 300)})
    y = pd.Series(3 * X.a + rng.randn(300) * 0.1)
    m1 = RF(0, n=20).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=20).fit(Xc, y)
    icd = ImportanceCollapseDetector(n_boot=3, n_sample=120)
    r = icd.compare(m1, m2, X, ['a'])
    assert len(r['features']) == 1


# ── paired.py ────────────────────────────────────────────────────────
def test_paired_modelos_identicos_sem_regras():
    from modelcheck.paired import PairedDivergenceAnalyzer
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.uniform(0, 1, (400, 3)), columns=list('abc'))
    y = pd.Series(3 * X.a + rng.randn(400) * 0.1)
    m = RF(0, n=30).fit(X, y)
    r = PairedDivergenceAnalyzer(max_depth=2, min_samples=20).analyze(
        m, m, X, list(X.columns))
    assert r['mean_divergence'] == 0.0
    assert len(r['rules']) == 0


def test_paired_com_ruido_puro_nao_aponta():
    """Lente de ruído: feature irrelevante não deve dominar as regras."""
    from modelcheck.paired import PairedDivergenceAnalyzer
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'sig': rng.uniform(0, 1, 500),
                      'noise': rng.uniform(0, 1, 500)})
    y = pd.Series(5 * X.sig + rng.randn(500) * 0.1)
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['sig'] = Xc['sig'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    r = PairedDivergenceAnalyzer(max_depth=2, min_samples=25).analyze(
        m1, m2, X, list(X.columns))
    top = [f for f, _ in r['top_features']]
    if top:
        assert top[0] == 'sig', f"a feature de sinal deve liderar: {top}"


# ── validation.py (4ª camada, cobertura era 8%) ──────────────────────
def test_validation_identico_nao_alerta():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(500, 3), columns=list('abc'))
    r = TrainServingValidator().check(X, X, list(X.columns))
    assert r['flagged'] == [], f"mesmos dados não devem alertar: {r['alerts']}"
    assert r['skew_score'] == 100.0


def test_validation_detecta_mudanca_de_escala():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame(rng.randn(500, 3), columns=list('abc'))
    Xsv = Xtr.copy(); Xsv['a'] = Xsv['a'] * 1000
    r = TrainServingValidator().check(Xtr, Xsv, list(Xtr.columns))
    assert 'a' in r['flagged']
    assert any(x in r['alerts'] for x in ('SCALE_CHANGE', 'PSI_ALERT',
                                          'MEAN_SHIFT', 'OUT_OF_RANGE'))


def test_validation_detecta_coluna_ausente():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame(rng.randn(300, 3), columns=list('abc'))
    Xsv = Xtr.drop(columns=['b'])
    r = TrainServingValidator().check(Xtr, Xsv, list(Xtr.columns))
    assert 'SCHEMA_MISMATCH' in r['alerts']


def test_validation_detecta_nivel_categorico_novo():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame({'num': rng.randn(400),
                        'cat': pd.Categorical(rng.choice(list('xy'), 400))})
    Xsv = Xtr.copy()
    Xsv['cat'] = Xsv['cat'].astype(str)
    Xsv.loc[Xsv.index[:50], 'cat'] = 'z'
    r = TrainServingValidator().check(Xtr, Xsv, ['num', 'cat'])
    assert 'NEW_LEVELS' in r['alerts'] or 'LEVEL_SHIFT' in r['alerts']


def test_validation_detecta_ausentes_novos():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame(rng.randn(400, 3), columns=list('abc'))
    Xsv = Xtr.copy(); Xsv.loc[Xsv.index[:120], 'b'] = np.nan
    r = TrainServingValidator().check(Xtr, Xsv, list(Xtr.columns))
    assert 'MISSING_SHIFT' in r['alerts']


def test_validation_colunas_extras():
    from modelcheck.validation import TrainServingValidator
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame(rng.randn(300, 2), columns=list('ab'))
    Xsv = Xtr.copy(); Xsv['nova'] = rng.randn(300)
    r = TrainServingValidator().check(Xtr, Xsv)
    assert 'nova' in r['extra_columns']


def test_validation_report_e_plain_nao_quebram():
    from modelcheck.validation import TrainServingValidator, plain_report
    rng = np.random.RandomState(0)
    Xtr = pd.DataFrame(rng.randn(300, 2), columns=list('ab'))
    Xsv = Xtr.copy(); Xsv['a'] = Xsv['a'] * 500
    v = TrainServingValidator()
    r = v.check(Xtr, Xsv, list(Xtr.columns))
    assert isinstance(v.report(r), str) and len(v.report(r)) > 0
    assert isinstance(plain_report(r), str) and len(plain_report(r)) > 0


# ── shape_extensions.py — bug 12 (v1.41) ─────────────────────────────
def test_categorical_divergence_ignora_categorica_irrelevante():
    """
    BUG 12: `threshold_magnitude` é ABSOLUTO, então qualquer flutuação de
    uma feature irrelevante passava o corte. Medido: 'cat' com 0.17% da
    importância apontada como divergente em controle puro (mesmo modelo,
    seed diferente).
    """
    import shap
    from modelcheck.shape_extensions import CategoricalShapeDivergence
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'sig': rng.uniform(0, 1, 600),
                      'noise': rng.uniform(0, 1, 600),
                      'cat': rng.randint(0, 3, 600)})
    y = pd.Series(5 * X.sig + rng.randn(600) * 0.2)   # cat irrelevante

    def sv(seed):
        m = RF(seed, n=30).fit(X, y)
        s = shap.TreeExplainer(m).shap_values(X)
        return s[:, :, 1] if s.ndim == 3 else s

    r = CategoricalShapeDivergence().check(sv(0), sv(1), X, list(X.columns))
    alertas = [x['feature'] for x in r
               if isinstance(x, dict) and x.get('alert')]
    assert alertas == [], f"controle não deve alertar: {alertas}"


def test_categorical_divergence_detecta_categorica_relevante():
    """O filtro não pode esconder uma categórica que de fato importa."""
    import shap
    from modelcheck.shape_extensions import CategoricalShapeDivergence
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'sig': rng.uniform(0, 1, 500),
                      'cat': rng.randint(0, 3, 500)})
    y = pd.Series(5 * X.sig + 3 * (X['cat'] == 2) + rng.randn(500) * 0.2)
    m1 = RF(0, n=25).fit(X, y)
    m2 = RF(0, n=25).fit(X.assign(cat=0), y)

    def sv(m):
        s = shap.TreeExplainer(m).shap_values(X)
        return s[:, :, 1] if s.ndim == 3 else s

    r = CategoricalShapeDivergence().check(sv(m1), sv(m2), X, list(X.columns))
    alertas = [x['feature'] for x in r
               if isinstance(x, dict) and x.get('alert')]
    assert 'cat' in alertas, f"categórica relevante deve alertar: {alertas}"


def test_todos_os_modulos_importam():
    """
    Teste de fumaça: 53 módulos no pacote. Um módulo que não importa
    quebra o `from modelcheck import *` de qualquer usuário.
    """
    import importlib, os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg = os.path.join(base, 'modelcheck')
    mods = sorted(f[:-3] for f in os.listdir(pkg)
                  if f.endswith('.py') and f != '__init__.py')
    falhas = []
    for m in mods:
        try:
            importlib.import_module(f'modelcheck.{m}')
        except Exception as e:
            falhas.append((m, f"{type(e).__name__}: {e}"))
    assert not falhas, f"módulos que não importam: {falhas}"


# ── slope.py — bug 13: invariância de escala (v1.42) ─────────────────
def test_slope_invariante_a_escala_do_target():
    """
    BUG 13: `min_effect` era um limiar ABSOLUTO sobre o slope, que está em
    unidades do target. Medido: com o target em escala 0.001 o detector dava
    ZERO alertas; em escala 1, 1e3 e 1e6 dava 2. Mesmos dados, mesma
    corrupção, diagnóstico diferente só por mudança de unidade — um usuário
    com target em probabilidade receberia silêncio.
    """
    import shap
    from modelcheck.slope import ShapeSlopeDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    base = 5 * X.a + rng.randn(400) * 0.2

    def alerta_em(escala):
        y = pd.Series(base.values * escala)
        m1 = RF(0, n=25).fit(X, y)
        Xc = X.copy(); Xc['a'] = Xc['a'].mean()
        m2 = RF(0, n=25).fit(Xc, y)

        def sv(m):
            s = shap.TreeExplainer(m).shap_values(X)
            return s[:, :, 1] if s.ndim == 3 else s

        r = ShapeSlopeDetector(threshold_ratio=3.0).check(
            sv(m1), sv(m2), X, list(X.columns))
        return {x['feature'] for x in r if x.get('divergence')}

    # a feature corrompida deve ser detectada em TODAS as escalas
    for escala in (0.001, 1.0, 1000.0):
        assert 'a' in alerta_em(escala), \
            f"escala {escala}: 'a' (corrompida) deveria ser detectada"


def test_slope_tem_filtro_de_importancia():
    """
    BUG 14: o slope não tinha filtro de importância relativa. Agora tem, e
    features irrelevantes recebem alert_type='IRRELEVANT_FEATURE'.
    """
    import shap
    from modelcheck.slope import ShapeSlopeDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'sig': rng.uniform(0, 1, 500),
                      'ruido': rng.uniform(0, 1, 500)})
    y = pd.Series(5 * X.sig + rng.randn(500) * 0.2)
    # controle puro: mesma config, só seed diferente
    m1 = RF(0, n=25).fit(X, y); m2 = RF(1, n=25).fit(X, y)

    def sv(m):
        s = shap.TreeExplainer(m).shap_values(X)
        return s[:, :, 1] if s.ndim == 3 else s

    det = ShapeSlopeDetector(threshold_ratio=3.0)
    assert hasattr(det, 'min_rel_importance')
    r = det.check(sv(m1), sv(m2), X, list(X.columns))
    div = [x['feature'] for x in r if x.get('divergence')]
    assert 'ruido' not in div, f"ruído não deve divergir em controle: {div}"


# ── Invariância de escala em TODOS os detectores (v1.43) ──────────────
def _setup_escala(esc, n=400, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({'a': rng.uniform(0, 1, n), 'b': rng.uniform(0, 1, n),
                      'c': rng.uniform(0, 1, n)})
    y = pd.Series((5 * X.a + 2 * X.b + rng.randn(n) * 0.2).values * esc)
    m1 = RF(0, n=25).fit(X, y)
    m1b = RF(1, n=25).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=25).fit(Xc, y)
    return m1, m1b, m2, X.sample(min(200, n), random_state=1), list(X.columns)


@pytest.mark.parametrize('detector', ['canary', 'geometry', 'psi', 'sdd', 'icd'])
def test_invariancia_de_escala(detector):
    """
    Multiplicar o target por 10³ e 10⁻³ não deve mudar o DIAGNÓSTICO.

    Bugs 13, 15 e 16 foram exatamente isso: limiar absoluto sobre quantidade
    em unidades do target. O bug 13 (slope) causava SILÊNCIO TOTAL com target
    em escala 0.001 — um usuário com target em probabilidade não receberia
    alerta nenhum num modelo quebrado.
    """
    def alertas(esc, corrompido):
        m1, m1b, m2, Xe, fs = _setup_escala(esc)
        mb = m2 if corrompido else m1b
        if detector == 'canary':
            from modelcheck.canary import GeometricCanaryDetector
            return len(GeometricCanaryDetector(n_sample=120).compare(
                m1, mb, Xe, fs)['alerts'])
        if detector == 'geometry':
            from modelcheck.geometry import GeometryDivergenceDetector
            return GeometryDivergenceDetector(n_sample=120).compare(
                m1, mb, Xe, fs)['n_alerts']
        if detector == 'psi':
            from modelcheck.psi_shap import PSIShapDetector
            d = PSIShapDetector(); d._calibrated = True
            return d.compare(m1, mb, Xe, fs)['n_alert']
        if detector == 'sdd':
            from modelcheck.audit import audit as _a
            return len(_a(m1, mb, Xe, verbose=False)['divergences'])
        from modelcheck.importance import ImportanceCollapseDetector
        return ImportanceCollapseDetector(n_boot=4, n_sample=120).compare(
            m1, mb, Xe, fs)['n_collapse']

    for corrompido in (False, True):
        vals = [alertas(e, corrompido) for e in (0.001, 1.0, 1000.0)]
        assert len(set(vals)) == 1, (
            f"{detector} não invariante a escala "
            f"({'corrompido' if corrompido else 'controle'}): {vals}")


def test_slope_invariancia_no_controle():
    """Complementa o teste do bug 13 pelo lado do controle."""
    import shap
    from modelcheck.slope import ShapeSlopeDetector

    def n_div(esc):
        m1, m1b, _, Xe, fs = _setup_escala(esc)

        def sv(m):
            s = shap.TreeExplainer(m).shap_values(Xe)
            return s[:, :, 1] if s.ndim == 3 else s

        r = ShapeSlopeDetector(threshold_ratio=3.0).check(
            sv(m1), sv(m1b), Xe, fs)
        return sum(1 for x in r if x.get('divergence'))

    vals = [n_div(e) for e in (0.001, 1.0, 1000.0)]
    assert len(set(vals)) == 1, f"slope não invariante no controle: {vals}"


# ── Invariância de escala nos detectores geométricos (v1.43) ──────────
@pytest.mark.parametrize('detector', ['canary', 'geometry', 'psi', 'universal'])
def test_invariancia_de_escala_geometricos(detector):
    """
    A família de bugs 9/11/12/13/14 é "limiar absoluto sobre quantidade
    dependente de escala". O bug 13 (slope) causava SILÊNCIO TOTAL com
    target em escala 0.001. Este teste varre os detectores geométricos
    exigindo o MESMO número de alertas em 3 ordens de magnitude.

    Medido: todos os 5 são invariantes (nº de alertas idêntico).
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400),
                      'b': rng.uniform(0, 1, 400),
                      'c': rng.uniform(0, 1, 400)})
    base = 5 * X.a + 2 * X.b + rng.randn(400) * 0.2
    Xe = X.sample(200, random_state=1)
    fs = list(X.columns)

    def n_alertas(escala):
        y = pd.Series(base.values * escala)
        m1 = RF(0, n=30).fit(X, y)
        Xc = X.copy(); Xc['a'] = Xc['a'].mean()
        m2 = RF(0, n=30).fit(Xc, y)
        if detector == 'canary':
            from modelcheck.canary import GeometricCanaryDetector
            r = GeometricCanaryDetector(n_sample=150).compare(m1, m2, Xe, fs)
            return len(r['alerts'])
        if detector == 'geometry':
            from modelcheck.geometry import GeometryDivergenceDetector
            r = GeometryDivergenceDetector(n_sample=150).compare(m1, m2, Xe, fs)
            return r['n_alerts']
        if detector == 'psi':
            from modelcheck.psi_shap import PSIShapDetector
            d = PSIShapDetector(); d._calibrated = True
            return d.compare(m1, m2, Xe, fs)['n_alert']
        from modelcheck.universal import UniversalCanaryDetector
        r = UniversalCanaryDetector(n_sample=150).compare(m1, m2, Xe, fs)
        return len(r['alerts'])

    vals = [n_alertas(e) for e in (1e-3, 1.0, 1e3)]
    assert len(set(vals)) == 1, \
        (f"{detector} não é invariante a escala: {vals} para "
         f"target × 1e-3, 1, 1e3")


def test_modulos_nao_validados_fora_do_namespace():
    """
    llm, llm_audit e vision têm 1.246 linhas e zero teste — incluindo um
    EUAIActChecker (claim regulatório). Movidos para experimental/ na v1.43.
    Este teste impede que voltem ao namespace principal sem validação.
    """
    import modelcheck
    for nome in ('LLMGuard', 'VisionGuard', 'EUAIActChecker',
                 'HallucinationDetector', 'VisionFairness'):
        assert not hasattr(modelcheck, nome), \
            (f"'{nome}' voltou ao namespace principal sem validação de "
             f"ground truth — ver experimental/__init__.py")


# ── Propriedades novas (v1.44): alinhamento, degenerado, monotônica ───
def test_alinhamento_por_nome_nao_por_indice():
    """
    BUG 15: o SHAP devolve colunas na ordem de TREINO de cada modelo. Se v1
    foi treinado com (a,b,c) e v2 com (c,b,a), indexar sv1[:,i] e sv2[:,i]
    com o mesmo i compara 'a' de v1 contra 'c' de v2.

    Medido antes da correção: score 10 e 2 divergências espúrias com os
    MESMOS dados, só reordenando as colunas de treino. Cenário real:
    pipeline de feature engineering que reordena colunas.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0.1, 1, 500),
                      'b': rng.uniform(0.1, 1, 500),
                      'c': rng.uniform(0.1, 1, 500)})
    y = pd.Series(5 * X.a + 2 * X.b + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X, y)
    m2 = RF(0, n=40).fit(X[['c', 'b', 'a']], y)      # mesma info, outra ordem
    r = audit(m1, m2, X, verbose=False)
    assert r['health_score'] == 100, \
        (f"reordenar colunas de treino não pode gerar divergência: "
         f"{[d['feature'] for d in r['divergences']]}")
    assert any('FEATURE_ORDER_DIFFERS' in w for w in r['warnings']), \
        "a diferença de ordem deve ser reportada como aviso"


def test_alinhamento_preserva_deteccao_real():
    """O alinhamento não pode esconder corrupção verdadeira."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0.1, 1, 500),
                      'b': rng.uniform(0.1, 1, 500),
                      'c': rng.uniform(0.1, 1, 500)})
    y = pd.Series(5 * X.a + 2 * X.b + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=40).fit(Xc[['c', 'b', 'a']], y)     # ordem trocada + corrompida
    r = audit(m1, m2, X, verbose=False)
    feats = [d['feature'] for d in r['divergences']]
    assert 'a' in feats, f"'a' corrompida deve ser detectada: {feats}"


def test_conjuntos_de_features_diferentes_levantam_erro():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 300), 'b': rng.uniform(0, 1, 300),
                      'c': rng.uniform(0, 1, 300)})
    y = pd.Series(3 * X.a + rng.randn(300) * 0.2)
    m1 = RF(0, n=25).fit(X, y)
    m2 = RF(0, n=25).fit(X[['a', 'b']], y)
    with pytest.raises(ValueError, match="features diferentes"):
        audit(m1, m2, X, verbose=False)


def test_modelo_degenerado_constante():
    """
    Modelo que sempre prevê a média: SHAP é zero em todas as features.
    Deve reportar colapso de importância, não erro nem silêncio.
    """
    from sklearn.dummy import DummyRegressor
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 500), 'b': rng.uniform(0, 1, 500)})
    y = pd.Series(5 * X.a + 2 * X.b + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X, y)
    dm = DummyRegressor(strategy='mean').fit(X, y)
    icd = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r = icd.compare(m1, dm, X, list(X.columns))
    colapsos = [x['feature'] for x in r['features']
                if x['alert'] == 'IMPORTANCE_COLLAPSE']
    assert set(colapsos) == {'a', 'b'}, \
        f"modelo constante deve colapsar TODAS as features: {colapsos}"


def test_treino_transformado_avaliado_cru_e_detectado():
    """
    EXPECTATIVA CORRIGIDA na v1.44.

    A versão anterior deste teste exigia ZERO divergência quando v2 é
    treinado com log(a) e avaliado em `a` cru. Isso estava ERRADO: árvores
    são invariantes a transformação monotônica apenas quando treino E
    inferência usam a MESMA transformação. Treinar em log(a) e prever em
    `a` cru coloca os limiares de split na escala errada — é train/serving
    skew, e o detector deve alertar.

    Medido: 'a' vai de crescente para irregular, spread 0.453. Detecção
    correta.

    A invariância legítima (mesma transformação nos dois lados) está
    testada em tests/test_properties.py.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0.1, 1, 500),
                      'b': rng.uniform(0.1, 1, 500)})
    y = pd.Series(5 * X.a + 2 * X.b + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X, y)
    Xl = X.copy(); Xl['a'] = np.log(Xl['a'])
    m2 = RF(0, n=40).fit(Xl, y)
    r = audit(m1, m2, X, verbose=False)
    feats = [d['feature'] for d in r['divergences']]
    assert 'a' in feats, \
        (f"treinar em log(a) e avaliar em 'a' cru é skew e deve ser "
         f"detectado; divergências: {feats}")


def test_colinearidade_aponta_a_feature_certa():
    """
    Com ρ=0.998, corromper x1 não deve fazer o detector acusar x1_copia.
    Medido: aponta apenas x1.
    """
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    x1 = rng.uniform(0, 1, 500)
    X = pd.DataFrame({'x1': x1, 'x1_copia': x1 + rng.randn(500) * 0.02,
                      'x2': rng.uniform(0, 1, 500)})
    y = pd.Series(3 * X.x1 + 1.5 * X.x2 + rng.randn(500) * 0.2)
    m1 = RF(0, n=40).fit(X, y)
    Xz = X.copy(); Xz['x1'] = Xz['x1'].mean()
    m2 = RF(0, n=40).fit(Xz, y)
    icd = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r = icd.compare(m1, m2, X, list(X.columns))
    hits = [x['feature'] for x in r['features']
            if x['alert'] == 'IMPORTANCE_COLLAPSE']
    assert 'x1' in hits, f"x1 (corrompida) deve ser apontada: {hits}"
    assert 'x1_copia' not in hits, \
        f"x1_copia (íntegra) não deve ser apontada: {hits}"
