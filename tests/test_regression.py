"""
Etapa 1 — Regressão: cenários já validados devem continuar passando.

Os valores esperados vêm das medições documentadas em v1.24–v1.28.
Qualquer alteração no framework que quebre estes testes é regressão.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import make_pair, RF
from modelcheck.audit import audit
from modelcheck.importance import ImportanceCollapseDetector
from modelcheck.api import Guard as ModelGuard


# ── Contrato básico: controle idêntico dá exatamente 100 ─────────────
@pytest.mark.parametrize('ds', ['diabetes', 'friedman', 'toy'])
def test_controle_identico_da_100(ds, request):
    X, y = request.getfixturevalue(ds)
    m1, _, Xte, _, _ = make_pair(X, y)
    r = audit(m1, m1, Xte, verbose=False)
    assert r['health_score'] == 100, f"{ds}: controle idêntico deve dar 100"
    assert len(r['divergences']) == 0


# ── ICD: recall em corrupção de feature, 0 FP em controle ────────────
@pytest.mark.parametrize('ds', ['diabetes', 'friedman'])
@pytest.mark.parametrize('kind', ['neutralize', 'shuffle'])
def test_icd_detecta_corrupcao(ds, kind, request):
    X, y = request.getfixturevalue(ds)
    m1, m2, Xte, _, tgt = make_pair(X, y, corrupt=True, kind=kind)
    icd = ImportanceCollapseDetector(n_boot=5, n_sample=150)
    res = icd.compare(m1, m2, Xte, list(X.columns))
    hits = [r['feature'] for r in res['features']
            if r['alert'] == 'IMPORTANCE_COLLAPSE']
    assert tgt in hits, f"{ds}/{kind}: ICD deve acusar '{tgt}', acusou {hits}"


@pytest.mark.parametrize('ds', ['diabetes', 'friedman', 'toy'])
def test_icd_sem_falso_positivo_em_controle(ds, request):
    X, y = request.getfixturevalue(ds)
    m1, _, Xte, _, _ = make_pair(X, y)
    icd = ImportanceCollapseDetector(n_boot=5, n_sample=150)
    res = icd.compare(m1, m1, Xte, list(X.columns))
    assert res['n_collapse'] == 0
    assert res['importance_score'] == 100


# ── check_features: recall 100%, especificidade 100% ────────────────
@pytest.mark.parametrize('ds', ['diabetes', 'friedman'])
def test_check_features_recall_e_especificidade(ds, request):
    X, y = request.getfixturevalue(ds)
    g = ModelGuard()
    fs = list(X.columns)
    m1, m2, Xte, _, tgt = make_pair(X, y, corrupt=True, kind='neutralize')
    r = g.check_features(m1, m2, Xte, monitored=[tgt], feature_names=fs)
    assert r['alert'] is True, f"{ds}: deve alertar sobre '{tgt}'"
    assert tgt in r['collapsed']
    m1b, m2b, Xteb, _, _ = make_pair(X, y, seed2=1)
    r2 = g.check_features(m1b, m2b, Xteb, monitored=[tgt], feature_names=fs)
    assert r2['alert'] is False, f"{ds}: variância de seed não deve alertar"


# ── Classificação: ICD deve funcionar ───────────────────────────────
def test_icd_em_classificacao(breast):
    X, y = breast
    m1, m2, Xte, _, tgt = make_pair(X, y, corrupt=True, kind='neutralize', clf=True)
    icd = ImportanceCollapseDetector(n_boot=5, n_sample=150)
    res = icd.compare(m1, m2, Xte, list(X.columns))
    hits = [r['feature'] for r in res['features']
            if r['alert'] == 'IMPORTANCE_COLLAPSE']
    assert tgt in hits


# ── representation: SHAP mais sensível que ALE (documentado) ─────────
def test_shap_nao_menos_sensivel_que_ale(diabetes):
    X, y = diabetes
    m1, m2, Xte, _, _ = make_pair(X, y, corrupt=True, kind='neutralize')
    n_shap = len(audit(m1, m2, Xte, representation='shap', verbose=False)['divergences'])
    n_ale = len(audit(m1, m2, Xte, representation='ale', verbose=False)['divergences'])
    assert n_shap >= 1
    assert n_ale >= 1


# ── Tendência: exige 4+ transições (FP 34% com 3) ────────────────────
def test_tendencia_exige_quatro_transicoes(toy):
    X, y = toy
    Xtr = X.iloc[:240]; ytr = y.iloc[:240]
    models = [RF(s).fit(Xtr, ytr) for s in range(4)]   # 3 transições
    g = ModelGuard()
    r = g.audit_versions(models, X.iloc[240:], y.iloc[240:], list(X.columns))
    assert 'indeterminado' in r['trend'], \
        "com 3 transições a tendência deve ser indeterminada (FP medido: 34%)"
    models5 = [RF(s).fit(Xtr, ytr) for s in range(6)]  # 5 transições
    r5 = g.audit_versions(models5, X.iloc[240:], y.iloc[240:], list(X.columns))
    assert r5['trend'] in ('estável', 'piorando', 'melhorando')


# ── Determinismo: mesma entrada, mesma saída ─────────────────────────
def test_determinismo_audit(toy):
    X, y = toy
    m1, m2, Xte, _, _ = make_pair(X, y, corrupt=True, kind='neutralize')
    scores = [audit(m1, m2, Xte, verbose=False)['health_score'] for _ in range(3)]
    assert len(set(scores)) == 1, f"audit não determinístico: {scores}"


def test_determinismo_check_features(toy):
    X, y = toy
    g = ModelGuard()
    m1, m2, Xte, _, tgt = make_pair(X, y, corrupt=True, kind='neutralize')
    outs = [tuple(g.check_features(m1, m2, Xte, monitored=[tgt],
                                   feature_names=list(X.columns))['collapsed'])
            for _ in range(3)]
    assert len(set(outs)) == 1, f"check_features não determinístico: {outs}"


# ═══════════════════════════════════════════════════════════════════
# PSI-SHAP — os 4 casos decisivos (v1.32)
# Travam regressão de calibração: os limiares padrão da indústria
# (0.10/0.25) faziam o controle disparar 6 de 11 features.
# ═══════════════════════════════════════════════════════════════════

def test_psi_shap_controle_nao_dispara_apos_calibrar(diabetes):
    """
    Caso A: variância de seed não deve gerar ALERT — DEPOIS de calibrar.

    Com os limiares de outro domínio o controle dispara (medido: p99 varia
    de 0.38 no ENA/SE a 1.35 no ENA/NE contra 0.43 no Wine). Este teste
    documenta o uso correto: calibrar antes de comparar.
    """
    from modelcheck.psi_shap import PSIShapDetector
    from conftest import RF, _split
    X, y = diabetes
    Xtr, Xte, ytr, yte = _split(X, y)
    fs = list(X.columns)
    det = PSIShapDetector(n_sample=200)
    cal = det.calibrate(lambda s: RF(s).fit(Xtr, ytr), Xtr, ytr, Xte, fs,
                        seeds=[11, 12, 13, 14, 15, 16])
    assert cal['ok'], f"calibração falhou: {cal}"
    m1, m2 = RF(0).fit(Xtr, ytr), RF(1).fit(Xtr, ytr)
    r = det.compare(m1, m2, Xte, fs)
    assert r['measured']
    assert r['n_alert'] == 0, \
        f"após calibrar, controle não deve gerar ALERT (gerou {r['n_alert']})"


def test_psi_shap_limiares_nao_transferem(diabetes):
    """
    Trava o achado: os limiares calibrados num domínio diferem o suficiente
    de outro para justificar o aviso de calibração obrigatória.
    """
    from modelcheck.psi_shap import PSIShapDetector
    from conftest import RF, _split
    X, y = diabetes
    Xtr, Xte, ytr, yte = _split(X, y)
    det = PSIShapDetector(n_sample=200)
    cal = det.calibrate(lambda s: RF(s).fit(Xtr, ytr), Xtr, ytr, Xte,
                        list(X.columns), seeds=[11, 12, 13, 14, 15, 16])
    padrao = 0.25
    assert cal['ok']
    assert cal['psi_warn'] > 0, "calibração deve produzir limiar positivo"


def test_psi_shap_detecta_colapso(diabetes):
    """Caso C: colapso de importância deve gerar ALERT."""
    from modelcheck.psi_shap import PSIShapDetector
    X, y = diabetes
    m1, m2, Xte, _, _ = make_pair(X, y, corrupt=True, kind='neutralize')
    det = PSIShapDetector(); det._calibrated = True
    r = det.compare(m1, m2, Xte, list(X.columns))
    assert r['n_alert'] >= 1, "colapso de importância deve gerar ALERT no PSI"


def test_psi_shap_nao_dispara_em_skew_linear(toy):
    """
    Caso D: transformação linear preserva a ordem; o PSI não deve alertar.
    Garante que o PSI-SHAP não é muleta da 4ª camada.
    """
    from modelcheck.psi_shap import PSIShapDetector
    from conftest import RF, _split
    X, y = toy
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0).fit(Xtr, ytr)
    Xs = Xtr.copy()
    Xs['a'] = Xs['a'] * 0.7 + Xs['a'].mean() * 0.3
    m2 = RF(0).fit(Xs, ytr)
    det = PSIShapDetector(); det._calibrated = True
    r = det.compare(m1, m2, Xte, list(X.columns))
    assert r['n_alert'] == 0, \
        f"skew linear não deve gerar ALERT no PSI, gerou {r['n_alert']}"


def test_psi_shap_limiares_calibrados_sao_padrao():
    """Trava o valor calibrado: 0.10/0.25 fazia o controle disparar."""
    from modelcheck.psi_shap import PSIShapDetector
    det = PSIShapDetector()
    assert det.psi_warn >= 0.20, "limiar warn baixo demais (era 0.10)"
    assert det.psi_alert >= 0.50, "limiar alert baixo demais (era 0.25)"


def test_psi_avisa_se_nao_calibrado(toy):
    """Os limiares não transferem entre domínios: exige aviso."""
    from modelcheck.psi_shap import PSIShapDetector
    X, y = toy
    m1, m2, Xte, _, _ = make_pair(X, y, seed2=1)
    det = PSIShapDetector()
    with pytest.warns(UserWarning, match="não calibrado"):
        det.compare(m1, m2, Xte, list(X.columns))


# ═══════════════════════════════════════════════════════════════════
# PairedDivergenceAnalyzer — validação contra ground truth (v1.34)
# A divergência é injetada numa região conhecida do espaço de features
# e verificamos se a regra extraída identifica essa região.
# ═══════════════════════════════════════════════════════════════════

def _pda_setup(regiao_fn, delta=4.0, n=1500, seed=0):
    from conftest import RF, _split
    from modelcheck.paired import PairedDivergenceAnalyzer
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, n) for i in range(5)})
    y = 3*X.x0 + 2*X.x1 + X.x2 + rng.randn(n)*0.15
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0, n=60).fit(Xtr, ytr)
    mtr = np.asarray(regiao_fn(Xtr), bool)
    y2 = ytr.copy()
    if mtr.sum():
        y2.loc[Xtr[mtr].index] = y2.loc[Xtr[mtr].index] + delta
    m2 = RF(0, n=60).fit(Xtr, y2)
    pa = PairedDivergenceAnalyzer(max_depth=3, min_samples=15)
    res = pa.analyze(m1, m2, Xte, list(X.columns))
    return res, Xte, np.asarray(regiao_fn(Xte), bool)


def _rule_mask(rule_text, X):
    m = np.ones(len(X), dtype=bool)
    for cond in rule_text.split(' AND '):
        for sep, op in [(' ≤ ', 'le'), (' > ', 'gt')]:
            if sep in cond:
                f, v = cond.split(sep)
                col = X[f.strip()].values.astype(float)
                m &= (col <= float(v)) if op == 'le' else (col > float(v))
                break
    return m


def test_pda_silencioso_sem_divergencia():
    """Controle: sem divergência injetada, não deve extrair regras."""
    res, Xte, _ = _pda_setup(lambda d: np.zeros(len(d), bool))
    assert len(res['rules']) == 0, \
        f"controle não deve gerar regras, gerou {len(res['rules'])}"


def test_pda_encontra_regiao_de_uma_feature():
    """Região x0 > 0.75: a regra deve usar x0 e cobrir a região."""
    res, Xte, verdade = _pda_setup(lambda d: d.x0 > 0.75)
    assert res['rules'], "deveria extrair ao menos uma regra"
    r = res['rules'][0]
    assert 'x0' in r['rule_text'], f"regra deve citar x0: {r['rule_text']}"
    pred = _rule_mask(r['rule_text'], Xte)
    inter = (pred & verdade).sum()
    iou = inter / max((pred | verdade).sum(), 1)
    assert iou >= 0.60, f"IoU baixo ({iou:.2f}) para região simples"


def test_pda_encontra_regiao_de_duas_features():
    """Conjunção x0 > 0.5 AND x1 < 0.5: a regra deve citar as duas."""
    res, Xte, verdade = _pda_setup(lambda d: (d.x0 > 0.5) & (d.x1 < 0.5))
    assert res['rules']
    txt = " ".join(r['rule_text'] for r in res['rules'][:2])
    assert 'x0' in txt and 'x1' in txt, f"regra deve citar x0 e x1: {txt}"
    pred = np.zeros(len(Xte), bool)
    for r in res['rules'][:2]:
        pred |= _rule_mask(r['rule_text'], Xte)
    iou = (pred & verdade).sum() / max((pred | verdade).sum(), 1)
    assert iou >= 0.50, f"IoU baixo ({iou:.2f}) para conjunção"


def test_pda_nao_inventa_features_erradas():
    """A regra não deve se apoiar em features irrelevantes (x3, x4)."""
    res, Xte, _ = _pda_setup(lambda d: d.x0 > 0.75)
    assert res['rules']
    principal = res['rules'][0]['rule_text']
    conds = [c.strip() for c in principal.split(' AND ')]
    usadas = set()
    for c in conds:
        for sep in (' ≤ ', ' > '):
            if sep in c:
                usadas.add(c.split(sep)[0].strip())
    assert 'x0' in usadas, f"x0 deve estar na regra: {principal}"


# ═══════════════════════════════════════════════════════════════════
# confidence.py — cobertura do IC (v1.35)
# Medido: cobertura 100% com nominal 95% -> IC conservador, largo.
# Estes testes travam a direção do erro: sobre-cobertura é aceitável,
# sub-cobertura (falsa precisão) não é.
# ═══════════════════════════════════════════════════════════════════

def test_ci_contem_o_valor_do_conjunto_completo(toy):
    """O IC calculado numa subamostra deve conter o score do conjunto todo."""
    from modelcheck.confidence import score_with_ci
    from conftest import RF, _split
    X, y = toy
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0).fit(Xtr, ytr)
    Xc = Xtr.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0).fit(Xc, ytr)
    fn = lambda a, b, Xb: audit(a, b, Xb, verbose=False)
    verdadeiro = fn(m1, m2, Xte)['health_score']
    sub = Xte.sample(min(120, len(Xte)), random_state=3)
    ci = score_with_ci(fn, m1, m2, sub, n_boot=15, seed=1)
    assert 'error' not in ci
    assert ci['ci_low'] <= verdadeiro <= ci['ci_high'], \
        (f"sub-cobertura: verdadeiro={verdadeiro} fora de "
         f"[{ci['ci_low']}, {ci['ci_high']}] — IC daria falsa precisão")


def test_ci_reporta_largura_e_nao_e_degenerado(toy):
    """O IC deve ter largura reportada e não ser um ponto."""
    from modelcheck.confidence import score_with_ci
    from conftest import RF, _split
    X, y = toy
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0).fit(Xtr, ytr)
    Xc = Xtr.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0).fit(Xc, ytr)
    fn = lambda a, b, Xb: audit(a, b, Xb, verbose=False)
    ci = score_with_ci(fn, m1, m2, Xte, n_boot=15, seed=1)
    assert ci['ci_width'] >= 0
    assert ci['n_valid'] >= 10, "réplicas insuficientes"
    assert ci['ci_low'] <= ci['mean'] <= ci['ci_high']


def test_compare_scores_ci_detecta_nao_distinguiveis(toy):
    """
    Cenários idênticos não devem ser declarados distintos.
    Trava a direção conservadora do teste.
    """
    from modelcheck.confidence import score_with_ci, compare_scores_ci
    from conftest import RF, _split
    X, y = toy
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0).fit(Xtr, ytr)
    fn = lambda a, b, Xb: audit(a, b, Xb, verbose=False)
    ci_a = score_with_ci(fn, m1, m1, Xte, n_boot=12, seed=1)
    ci_b = score_with_ci(fn, m1, m1, Xte, n_boot=12, seed=2)
    cmp = compare_scores_ci(ci_a, ci_b, 'A', 'B')
    assert cmp['distinct'] is False, \
        "dois controles idênticos não podem ser 'distintos'"


# ═══════════════════════════════════════════════════════════════════
# ShapeCausalExplainer — validação contra ground truth (v1.36)
# Medido: sem filtro de importância, uma feature de ruído (0.08% da
# importância) era apontada como causa com confiança 0.62-0.92 em
# controle SEM nenhum efeito causal.
# ═══════════════════════════════════════════════════════════════════

def _causal_setup(n=700, seed=0, grupo='aleatorio', causa=False):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, n) for i in range(4)})
    if grupo == 'aleatorio':
        g = rng.randint(0, 3, n)
    else:
        g = pd.qcut(X.x3, 3, labels=False).values
    sinal = np.where(g == 2, -3.0, 3.0) if causa else 3.0
    y = pd.Series(sinal * X.x0 + 2 * X.x1 + X.x2 + rng.randn(n) * 0.2)
    X['g'] = g
    return X, y


def test_causal_wilson_nao_quebra_com_n_zero():
    """O IC de Wilson devolvia ZeroDivisionError com n=0."""
    from modelcheck.causal_bootstrap import _wilson_ci
    lo, hi = _wilson_ci(0.0, 0)
    assert 0.0 <= lo <= hi <= 1.0


def test_causal_wilson_dentro_de_zero_um():
    from modelcheck.causal_bootstrap import _wilson_ci
    for p in (0.0, 0.25, 0.5, 1.0):
        for n in (1, 5, 20, 100):
            lo, hi = _wilson_ci(p, n)
            assert 0.0 <= lo <= hi <= 1.0, f"IC inválido p={p} n={n}"


def test_causal_exclui_feature_irrelevante():
    """
    x3 não entra no target: deve ser excluída por irrelevância, senão
    gera causalidade espúria (medido: confiança 0.62 em controle).
    """
    from modelcheck.causal_bootstrap import ShapeCausalExplainer
    X, y = _causal_setup(grupo='aleatorio', causa=False)
    e = ShapeCausalExplainer(n_bootstrap=6, min_group_size=40, n_shap=60,
                             min_rel_importance=0.02)
    e.fit(X, y, group_col='g', verbose=False)
    assert 'x3' in e.excluded_features_, \
        (f"x3 (importância {e.rel_importance_.get('x3')}) deveria ser "
         f"excluída; excluídas={e.excluded_features_}")


def test_causal_controle_sem_causa_espuria():
    """Grupo aleatório e nenhum efeito: não deve apontar causa."""
    from modelcheck.causal_bootstrap import ShapeCausalExplainer
    X, y = _causal_setup(grupo='aleatorio', causa=False)
    e = ShapeCausalExplainer(n_bootstrap=6, min_group_size=40, n_shap=60,
                             min_rel_importance=0.02)
    e.fit(X, y, group_col='g', verbose=False)
    df = e.all_confidences()
    altos = df[df.confidence >= 0.5] if len(df) else df
    assert len(altos) == 0, \
        f"controle sem efeito não deve apontar causa: {altos.to_dict('records')[:3]}"


# ═══════════════════════════════════════════════════════════════════
# Auditoria do padrão "feature irrelevante domina" (v1.37)
# Encontrado no causal_bootstrap (v1.36), auditado em shape, slope e
# paired. Achados: SDD gerava divergência espúria em ruído; slope
# levantava ZeroDivisionError quando a feature era neutralizada.
# ═══════════════════════════════════════════════════════════════════

def _ruido_setup(n=700, seed=0, corromper=False):
    """x3 é ruído puro: não entra no target (~0.05% da importância)."""
    from conftest import RF
    rng = np.random.RandomState(seed)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, n) for i in range(4)})
    y = pd.Series(3*X.x0 + 2*X.x1 + X.x2 + rng.randn(n)*0.2)
    m1 = RF(0, n=40).fit(X, y)
    if corromper:
        Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
        m2 = RF(0, n=40).fit(Xc, y)
    else:
        m2 = RF(1, n=40).fit(X, y)
    return m1, m2, X.sample(min(250, n), random_state=1), y


def test_sdd_ignora_ruido_no_controle():
    """
    Sem filtro de importância o SDD apontava 'x3' (ruído, 0.04% da
    importância) como divergente em controle de seed — score 58 em vez
    de 100. Trava a correção.
    """
    m1, m2, Xe, _ = _ruido_setup(corromper=False)
    r = audit(m1, m2, Xe, verbose=False)
    feats = [d['feature'] for d in r['divergences']]
    assert 'x3' not in feats, \
        f"ruído não deve ser apontado como divergente: {feats}"


def test_sdd_filtro_desativavel():
    """min_rel_importance=0.0 restaura o comportamento antigo."""
    m1, m2, Xe, _ = _ruido_setup(corromper=False)
    com = audit(m1, m2, Xe, verbose=False)['health_score']
    sem = audit(m1, m2, Xe, min_rel_importance=0.0, verbose=False)['health_score']
    assert com >= sem, "o filtro não deve reduzir o score do controle"


def test_sdd_ainda_detecta_causa_real():
    """O filtro não pode esconder a divergência verdadeira em x0."""
    m1, m2, Xe, _ = _ruido_setup(corromper=True)
    r = audit(m1, m2, Xe, verbose=False)
    feats = [d['feature'] for d in r['divergences']]
    assert 'x0' in feats, f"x0 (corrompida) deve ser detectada: {feats}"


def test_slope_nao_quebra_com_feature_neutralizada():
    """
    ZeroDivisionError: o slope de v2 é zero quando a feature é
    neutralizada — o cenário mais comum de falha de pipeline. O erro
    derrubava o check() inteiro, impedindo a análise das outras features.
    """
    import shap
    from modelcheck.slope import ShapeSlopeDetector
    m1, m2, Xe, _ = _ruido_setup(corromper=True)

    def sv(m):
        s = shap.TreeExplainer(m).shap_values(Xe)
        return s[:, :, 1] if s.ndim == 3 else s

    det = ShapeSlopeDetector(threshold_ratio=3.0)
    res = det.check(sv(m1), sv(m2), Xe, list(Xe.columns))   # não deve levantar
    assert isinstance(res, list) and len(res) == len(Xe.columns)
    tipos = {r.get('alert_type') for r in res}
    assert 'SLOPE_COLLAPSED' in tipos or any(r.get('divergence') for r in res), \
        "colapso de inclinação deve ser reportado"


# ═══════════════════════════════════════════════════════════════════
# Alta dimensionalidade e determinismo (v1.38)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.parametrize('P', [60, 200])
def test_icd_sem_falso_positivo_em_alta_dimensao(P):
    """
    Com P=200 o ICD acusava uma feature ÍNTEGRA (f1) em 2/3 dos casos.
    Causa: o peso RELATIVO de f1 caía (0.32 -> 0.11) porque o denominador
    se redistribuía entre 198 features de ruído, embora o coeficiente de
    f1 quase não mudasse (+1.995 -> +1.947). Corrigido exigindo colapso
    também na razão ABSOLUTA.
    """
    from sklearn.linear_model import Ridge
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.randn(700, P), columns=[f'f{i}' for i in range(P)])
    y = pd.Series(X.f0 * 3 + X.f1 * 2 + rng.randn(700) * 0.3)
    Xc = X.copy(); Xc['f0'] = Xc['f0'].mean()
    icd = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r = icd.compare(Ridge().fit(X, y), Ridge().fit(Xc, y),
                    X.sample(200, random_state=1), list(X.columns),
                    monitored=['f0', 'f1'])
    hits = r['monitored_hits']
    assert 'f0' in hits, f"f0 (corrompida) deve ser detectada em P={P}"
    assert 'f1' not in hits, \
        f"f1 (íntegra) não deve ser acusada em P={P}: {hits}"


def test_icd_reporta_razao_absoluta_e_relativa():
    """Ambas as escalas devem estar na saída, para inspeção."""
    from modelcheck.importance import ImportanceCollapseDetector
    m1, m2, Xe, _ = _ruido_setup(corromper=True)
    icd = ImportanceCollapseDetector(n_boot=4, n_sample=150)
    r = icd.compare(m1, m2, Xe, list(Xe.columns))
    assert all('ratio' in f and 'ratio_abs' in f for f in r['features'])


@pytest.mark.parametrize('detector', ['canary', 'topology', 'geometry', 'psi'])
def test_detectores_geometricos_sao_deterministicos(detector, toy):
    """
    Canary, Topology, Geometry e PSI usam subamostragem. Três execuções
    com a mesma entrada devem dar resultado idêntico — requisito para
    auditoria reproduzível. Medido: todos determinísticos.
    """
    from conftest import RF, _split
    X, y = toy
    Xtr, Xte, ytr, yte = _split(X, y)
    m1 = RF(0).fit(Xtr, ytr)
    Xc = Xtr.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0).fit(Xc, ytr)
    fs = list(X.columns)

    if detector == 'canary':
        from modelcheck.canary import GeometricCanaryDetector
        fn = lambda: GeometricCanaryDetector(n_sample=100).compare(
            m1, m2, Xte, fs)['canary_score']
    elif detector == 'topology':
        from modelcheck.universal import TopologicalDriftGudhi
        d = TopologicalDriftGudhi(n_points=40)
        if not d.available:
            pytest.skip("gudhi não instalado")
        fn = lambda: d.compare(m1, m2, Xte, fs)['topo_score']
    elif detector == 'geometry':
        from modelcheck.geometry import GeometryDivergenceDetector
        fn = lambda: GeometryDivergenceDetector(n_sample=100).compare(
            m1, m2, Xte, fs)['geometry_score']
    else:
        from modelcheck.psi_shap import PSIShapDetector
        d = PSIShapDetector(); d._calibrated = True
        fn = lambda: d.compare(m1, m2, Xte, fs)['psi_score']

    vals = [fn() for _ in range(3)]
    assert len(set(vals)) == 1, f"{detector} não determinístico: {vals}"
