"""
orchestrator.py — ModelGuard Enterprise Orchestrator

Sistema hierárquico que integra todos os detectores em 3 camadas:

  CAMADA 1 — FORMA LOCAL
    SDD, ShapeSlopeDetector, RHALEDetector, PairedDivergenceAnalyzer

  CAMADA 2 — GEOMÉTRICA
    GeometricCanaryDetector, GTSF, TopologicalDriftDetector

  CAMADA 3 — TEMPORAL / EXTREMOS
    ShapeTrendingDetector (se histórico), OODBehavioralDetector

Saída: diagnóstico hierárquico com:
  - Health Score Global ponderado por camada
  - Alertas organizados por camada e severidade
  - Interpretação em linguagem natural
  - Recomendação de ação específica

Uso:
    from modelcheck.orchestrator import ModelGuardOrchestrator

    orc = ModelGuardOrchestrator()
    report = orc.diagnose(model_v1, model_v2, X_eval, y_eval, features)
    print(report['report_text'])
"""
import numpy as np
import pandas as pd
from typing import List, Optional
import concurrent.futures


def _is_tree(model) -> bool:
    cls = type(model).__name__.lower()
    return any(k in cls for k in
               ['forest','tree','boost','xgb','lgb','catboost','extra'])


def _registrar_falha(results, camada, exc, consequencia):
    """
    POLÍTICA (v1.66): em código de auditoria, toda exceção capturada vira
    uma lacuna declarada. Um `except` que não registra esconde uma medição
    que deixou de acontecer — foi assim que o bug 30 sobreviveu: a camada
    de forma falhava com NameError, o veredito saía certo pelo detector de
    integridade, e nenhum teste de comportamento notou.

    Se uma exceção não merece virar lacuna, ela não deveria ser capturada.
    """
    results.setdefault('_failures', []).append({
        'layer': camada,
        'reason': f'{type(exc).__name__}: {exc}'[:160],
        'consequence': consequencia,
    })


def _modelo_degenerado(model_v1, model_v2, X_eval) -> bool:
    """
    v2 prevê praticamente o mesmo valor para todos os casos, e v1 não.

    Usa a saída contínua (probabilidade em classificador) — com `predict`
    duro, um classificador desbalanceado que prevê sempre a maioria pareceria
    degenerado sem ser.
    """
    def _saida(m):
        if hasattr(m, 'predict_proba'):
            try:
                return np.asarray(m.predict_proba(X_eval), dtype=float)
            except Exception:
                pass    # noqa: lacuna-nao-aplicavel — usa predict()
        return np.asarray(m.predict(X_eval), dtype=float)
    try:
        a, b = _saida(model_v1), _saida(model_v2)
        amp = lambda z: float(np.nanmax(np.ptp(z.reshape(len(z), -1), axis=0)))
        a1, a2 = amp(a), amp(b)
        return bool(a1 > 1e-9 and a2 <= 1e-6 * max(a1, 1e-12) + 1e-12)
    except Exception:
        return False    # noqa: lacuna-nao-aplicavel — sinal complementar


def _run_layer1(model_v1, model_v2, X_eval, y_eval,
                feature_names, threshold, use_shap, run_rhale=False,
                min_rel_importance=0.01, frechet_threshold=None):
    """Camada 1: Forma Local."""
    results = {}

    # SDD via SHAP ou ALE
    if use_shap:
        import shap
        from modelcheck.shape import _shape_profile
        from modelcheck.slope import ShapeSlopeDetector
        try:
            from modelcheck.frechet import como_array_shap, fatias_por_classe
            # BUG 68 (v1.0.4): multiclasse mantém todas as classes; shap
            # antigo (lista) também é normalizado em vez de quebrar em .ndim.
            sv1 = como_array_shap(shap.TreeExplainer(model_v1).shap_values(X_eval))
            sv2 = como_array_shap(shap.TreeExplainer(model_v2).shap_values(X_eval))

            # v0.10.0: detecção por distância de Fréchet, classe como nome.
            # O critério anterior (spread >= threshold) exigia que a
            # IMPORTÂNCIA mudasse junto com a forma e detectava 0% no
            # Benchmark B. Implementação única em modelcheck/frechet.py.
            from modelcheck.frechet import divergencias_forma, diff_influencia
            sem_forma = []
            from modelcheck.frechet import LIMIAR_PADRAO
            shape_divs = divergencias_forma(
                X_eval, sv1, sv2, feature_names,
                min_rel_importance=min_rel_importance,
                limiar=(LIMIAR_PADRAO if frechet_threshold is None
                        else float(frechet_threshold)),
                _pulados=sem_forma)
            results['influencia'] = diff_influencia(sv1, sv2, feature_names,
                                                    X_eval=X_eval)

            # slope é sinal de depuração, fora do veredito: usa a fatia
            # principal (classe 1 no binário, classe 0 no multiclasse)
            slope_det = ShapeSlopeDetector(threshold_ratio=3.0)
            slope_res = slope_det.check(fatias_por_classe(sv1)[0],
                                        fatias_por_classe(sv2)[0],
                                        X_eval, feature_names)
            slope_alerts = [r for r in slope_res if r.get('divergence')]

            results['sdd'] = {'divs': shape_divs, 'method': 'SHAP',
                              'sem_forma': sem_forma}
            results['slope'] = {'alerts': slope_alerts}
            results['sv1'] = sv1; results['sv2'] = sv2
        except Exception as e:
            from modelcheck.frechet import explicar_erro_shap
            results['sdd'] = {'divs': [], 'method':
                              f'SHAP_ERROR:{explicar_erro_shap(e, model_v1, model_v2)}'}
            results['slope'] = {'alerts': []}
    else:
        try:
            from modelcheck.ale_detector import ALEShapeDetector
            det = ALEShapeDetector(n_bins=5, n_boot=20, min_confidence=0.6)
            ale_res = det.compare(model_v1, model_v2, X_eval, feature_names)
            shape_divs = [{'feature':r['feature'],'shape_v1':r['shape1'],
                           'shape_v2':r['shape2'],'spread':round((r['conf1']+r['conf2'])/2,3)}
                          for r in ale_res if r['divergence']]
            results['sdd'] = {'divs': shape_divs, 'method': 'ALE'}
        except Exception as e:
            # vira lacuna pelo método com ERROR (AuditResult.not_evaluated)
            results['sdd'] = {'divs': [], 'method': f'ALE_ERROR:{e}'}
        results['slope'] = {'alerts': []}

    # RHALE — DESLIGADO POR PADRÃO (v1.61)
    # Medido: consome 51 dos 55 segundos do diagnóstico completo (93% do
    # tempo), por causa do bootstrap de forma em cada bin de cada variável.
    # É um detector complementar, não faz parte do núcleo validado, e sua
    # contribuição ao veredito é marginal. Ligue com run_rhale=True se
    # precisar dele.
    try:
        if not run_rhale:
            raise RuntimeError('rhale desligado por padrão (custo: ~50s)')
        from modelcheck.rhale import RHALEDetector
        rhale = RHALEDetector(max_bins=5, min_bin_n=5, n_boot=5, min_conf=0.60, hetero_threshold=6.0)
        rhale_res = rhale.compare(model_v1, model_v2, X_eval, feature_names)
        results['rhale'] = rhale_res
    except Exception as e:
        results['rhale'] = []
        # desligamento intencional nao e lacuna: so registra se o
        # usuario pediu a camada e ela falhou de verdade.
        if run_rhale:
            _registrar_falha(results, 'RHALE heterogeneity', e,
                             'heterogeneidade do efeito nao foi medida')

    # Importance Collapse (detecta o que o SDD não detecta)
    try:
        from modelcheck.importance import ImportanceCollapseDetector
        icd = ImportanceCollapseDetector(n_boot=6, n_sample=150)
        results['importance'] = icd.compare(model_v1, model_v2,
                                             X_eval, feature_names)
    except Exception as e:
        results['importance'] = {'importance_score': None, 'alerts': [],
                                  'n_collapse': 0, 'n_emergence': 0,
                                  'measured': False, 'error': str(e)}
        # BUG 71 (v1.0.4): a falha da camada de integridade era guardada e
        # esquecida. Sem causa raiz e sem lacuna declarada, o veredito caía
        # em APROVADO com uma coluna inteiramente quebrada. É a camada que
        # decide BLOQUEADO: a falha dela tem de aparecer.
        _registrar_falha(results, 'feature integrity', e,
                         'não foi possível verificar se alguma variável '
                         'deixou de influenciar o modelo')

    # Paired Divergence
    try:
        from modelcheck.paired import PairedDivergenceAnalyzer
        pa = PairedDivergenceAnalyzer(max_depth=3, min_samples=8)
        paired_res = pa.analyze(model_v1, model_v2, X_eval, feature_names)
        results['paired'] = paired_res
    except Exception as e:
        results['paired'] = {}
        _registrar_falha(results, 'paired divergence', e,
                         'divergencia pareada por subgrupo nao foi medida')

    # Score camada 1
    n_shape = len(results['sdd']['divs'])
    n_slope = len(results['slope']['alerts'])
    n_hetero = sum(1 for r in results['rhale']
                   if isinstance(r, dict) and r.get('hetero_alert'))
    # Hetero só penaliza se há evidência de shape_div ou slope também
    n_hetero_eff = n_hetero if (n_shape > 0 or n_slope > 0) else 0
    imp = results.get('importance', {})
    n_collapse = imp.get('n_collapse', 0) or 0
    n_emerge   = imp.get('n_emergence', 0) or 0
    score1 = max(0, 100 - n_shape*15 - n_slope*10
                 - min(n_hetero_eff,3)*5
                 - n_collapse*20 - n_emerge*8)
    results['layer_score'] = round(score1, 1)
    results['n_alerts'] = n_shape + n_slope + n_hetero

    return results


def _run_layer2(model_v1, model_v2, X_eval, feature_names):
    """Camada 2: Geométrica."""
    results = {}

    # Geometric Canary
    try:
        from modelcheck.canary import GeometricCanaryDetector
        can = GeometricCanaryDetector(n_sample=100, frechet_thresh=3.0,
                                      topology_thresh=0.90, isotropy_thresh=0.10)
        results['canary'] = can.compare(model_v1, model_v2,
                                         X_eval, feature_names)
    except Exception as e:
        results['canary'] = {'canary_score': None, 'alerts': [],
                              'measured': False, 'error': str(e)}

    # Topological Drift
    try:
        from modelcheck.topology import TopologicalDriftDetector
        topo = TopologicalDriftDetector(n_sample=60, bottleneck_thresh=0.3,
                                         betti0_thresh=2)
        results['topology'] = topo.compare(model_v1, model_v2,
                                             X_eval, feature_names)
    except Exception as e:
        results['topology'] = {'topo_score': None, 'alerts': [],
                                'measured': False, 'error': str(e)}

    # Geometry (GDD)
    try:
        from modelcheck.geometry import GeometryDivergenceDetector
        gdd = GeometryDivergenceDetector(n_sample=100)
        results['geometry'] = gdd.compare(model_v1, model_v2,
                                           X_eval, feature_names)
    except Exception as e:
        results['geometry'] = {'geometry_score': None, 'n_alerts': 0,
                               'measured': False, 'error': str(e)}

    # Score ponderado apenas sobre detectores efetivamente medidos
    parts = [(0.40, results['canary'].get('canary_score')),
             (0.35, results['topology'].get('topo_score')),
             (0.25, results['geometry'].get('geometry_score'))]
    valid = [(w, s) for w, s in parts if s is not None]
    if valid:
        wsum = sum(w for w, _ in valid)
        score2 = round(sum(w * s for w, s in valid) / wsum, 1)
        results['measured'] = True
    else:
        score2 = None
        results['measured'] = False

    all_alerts = (results['canary'].get('alerts', []) +
                  results['topology'].get('alerts', []))
    results['layer_score'] = score2
    results['n_alerts'] = len(all_alerts)
    return results


def _run_layer3(model_v1, model_v2, X_eval, feature_names):
    """Camada 3: Temporal e Extremos."""
    results = {}

    # OOD
    try:
        from modelcheck.ood import OODBehavioralDetector
        ood = OODBehavioralDetector(stress_multiplier=3.0,
                                    divergence_threshold=1.00)
        stress_feats = feature_names[:4]
        results['ood'] = ood.analyze(model_v1, model_v2, X_eval,
                                      feature_names, stress_feats)
    except Exception as e:
        results['ood'] = {'n_ood_alerts': 0, 'ood_features': []}
        _registrar_falha(results, 'out-of-distribution behaviour', e,
                         'comportamento em cenarios extremos nao foi verificado')

    n_ood = results['ood'].get('n_ood_alerts', 0)
    score3 = max(0, 100 - min(n_ood, 4) * 10)
    results['layer_score'] = round(score3, 1)
    results['n_alerts'] = n_ood
    return results


def _global_score(s1, s2, s3, mode: str = "retraining") -> tuple:
    """Score global ponderado, ignorando camadas não medidas (None)."""
    # No modo exploratório a camada 1 (forma local) pesa menos: espera-se
    # que arquiteturas distintas aprendam formas distintas.
    if mode == "exploratory":
        parts = [(0.15, s1), (0.60, s2), (0.25, s3)]
    else:
        parts = [(0.40, s1), (0.40, s2), (0.20, s3)]
    valid = [(w, s) for w, s in parts if s is not None]
    if not valid:
        return None, "NOT_MEASURED"
    wsum = sum(w for w, _ in valid)
    score = round(sum(w * s for w, s in valid) / wsum, 1)
    status = ("HEALTHY"   if score >= 85 else
              "ATTENTION" if score >= 70 else
              "UNSTABLE"  if score >= 55 else "CRITICAL")
    return score, status


def _interpret(layer1, layer2, layer3, score, status,
               label_v1, label_v2, mode: str = "retraining") -> dict:
    """Gera interpretação e recomendação."""
    shape_divs = layer1.get('sdd', {}).get('divs', [])
    slope_alerts = layer1.get('slope', {}).get('alerts', [])
    hetero = [r for r in layer1.get('rhale', [])
              if isinstance(r, dict) and r.get('hetero_alert')]
    canary_alerts = layer2.get('canary', {}).get('alerts', [])
    topo_alerts = layer2.get('topology', {}).get('alerts', [])
    ood_feats = layer3.get('ood', {}).get('ood_features', [])
    imp = layer1.get('importance', {}) or {}
    collapsed = [r['feature'] for r in imp.get('features', [])
                 if r.get('alert') == 'IMPORTANCE_COLLAPSE']

    # No modo exploratório, divergência de forma é informação, não anomalia
    if mode == "exploratory":
        if collapsed:
            problem_type = "IMPORTANCE_COLLAPSE"
        elif canary_alerts or topo_alerts:
            problem_type = "GEOMETRIC"
        elif shape_divs:
            problem_type = "EXPECTED_ARCH_DIFF"
        else:
            problem_type = "NONE"
        recs_exp = {
            "EXPECTED_ARCH_DIFF": [
                f"ℹ Diferenças estruturais esperadas entre arquiteturas.",
                f"   {len(shape_divs)} feature(s) com formas distintas:",
            ] + [f"   → '{d['feature']}': {d['shape_v1']} vs {d['shape_v2']}"
                 for d in shape_divs[:3]] + [
                "",
                "Isto NÃO é anomalia no modo exploratório.",
                "Use para escolher a arquitetura mais adequada ao domínio.",
            ],
            "IMPORTANCE_COLLAPSE": [
                f"⚠ Mesmo em modo exploratório, {len(collapsed)} feature(s) "
                f"perderam toda a importância:",
            ] + [f"   → '{f}'" for f in collapsed[:3]] + [
                "",
                "Verifique se a arquitetura ignora essa feature por design "
                "ou se há problema nos dados.",
            ],
            "GEOMETRIC": [
                f"⚠ Estrutura geométrica global divergente.",
                f"   Alertas: {', '.join(canary_alerts + topo_alerts)}",
                "",
                "Em modo exploratório isso indica que os modelos organizam "
                "o espaço de decisão de forma fundamentalmente diferente.",
            ],
            "NONE": [
                "✅ Arquiteturas equivalentes estruturalmente.",
                "   A escolha pode ser feita por custo/latência.",
            ],
        }
        return {
            'problem_type': problem_type,
            'recommendation_lines': recs_exp.get(problem_type, recs_exp["NONE"]),
            'n_shape_divs': len(shape_divs),
            'n_slope': len(slope_alerts),
            'n_hetero': len(hetero),
            'n_canary': len(canary_alerts),
            'n_topo': len(topo_alerts),
            'n_ood': len(ood_feats),
            'mode': mode,
        }

    # Classifica o tipo de problema
    problem_type = "NONE"
    if collapsed:
        problem_type = "IMPORTANCE_COLLAPSE"
    elif shape_divs and not canary_alerts:
        problem_type = "LOCAL_SHAPE"
    elif canary_alerts and shape_divs:
        problem_type = "MIXED"
    elif canary_alerts and not shape_divs:
        problem_type = "GEOMETRIC"
    elif topo_alerts:
        problem_type = "TOPOLOGICAL"
    elif ood_feats:
        problem_type = "OOD_ONLY"
    elif slope_alerts and not shape_divs:
        problem_type = "INTENSITY_ONLY"

    # Recomendações por tipo
    recs = {
        "IMPORTANCE_COLLAPSE": [
            f"⚠ COLAPSO DE IMPORTÂNCIA em {len(collapsed)} feature(s):",
        ] + [f"   → '{f}' perdeu praticamente todo o peso no modelo v2"
             for f in collapsed[:3]] + [
            "",
            "Ação recomendada:",
            "1. BLOQUEAR deploy — provável falha no pipeline de dados.",
            "2. Verificar se a coluna virou constante/nula na origem.",
            "3. Conferir joins, filtros e imputação do ETL.",
            "4. Este padrão é típico de bug de ingestão, não de retreino.",
        ],
        "NONE": [
            f"✅ Nenhuma anomalia detectada.",
            f"   {label_v2} pode ser deployado com confiança.",
        ],
        "LOCAL_SHAPE": [
            f"⚠ Mudança LOCAL detectada em {len(shape_divs)} feature(s):",
        ] + [f"   → '{d['feature']}': {d['shape_v1']} → {d['shape_v2']}"
             for d in shape_divs[:3]] + [
            "",
            "Ação recomendada:",
            "1. Investigar por que essas features mudaram de forma.",
            "2. Verificar se a mudança é intencional (ex: nova engenharia de features).",
            "3. Se acidental: retreinar com dados corretos.",
            f"4. Deploy com monitoramento adicional dessas features.",
        ],
        "MIXED": [
            f"⚠ Mudanças LOCAL + GEOMÉTRICA detectadas.",
            f"   {len(shape_divs)} feature(s) com divergência de forma.",
            f"   Espaço geométrico global também alterado.",
            "",
            "Ação recomendada:",
            "1. BLOQUEAR deploy até investigação.",
            "2. Verificar processo de retreinamento.",
            "3. Comparar dados de treino v1 e v2.",
        ],
        "GEOMETRIC": [
            f"⚠ Mudança GEOMÉTRICA detectada (sem divergência de forma local).",
            f"   Alertas: {', '.join(canary_alerts)}",
            "",
            "Ação recomendada:",
            "1. Executar análise de subgrupos (PairedDivergenceAnalyzer).",
            "2. Verificar distribuição dos dados de retreinamento.",
            "3. Deploy com cautela + monitoramento geométrico contínuo.",
        ],
        "TOPOLOGICAL": [
            f"⚠ Mudança TOPOLÓGICA detectada.",
            f"   Estrutura de conectividade do espaço SHAP alterada.",
            "",
            "Ação recomendada:",
            "1. Verificar se novos clusters de comportamento emergiram.",
            "2. Revisar dados de treino por subgrupos inesperados.",
        ],
        "OOD_ONLY": [
            f"⚠ Comportamento divergente em cenários extremos.",
            f"   Features: {', '.join(ood_feats[:3])}",
            "",
            "Ação recomendada:",
            "1. Testar modelo em dados de crise históricos.",
            "2. Deploy apenas para o range normal de dados.",
        ],
        "INTENSITY_ONLY": [
            f"⚠ Mudança de INTENSIDADE detectada (forma preservada).",
            f"   {len(slope_alerts)} feature(s) com slope diferente.",
            "",
            "Ação recomendada:",
            "1. Verificar se a mudança de intensidade é esperada.",
            "2. Recalibrar thresholds de alerta operacional.",
        ],
    }
    rec_lines = recs.get(problem_type, recs["NONE"])

    return {
        'problem_type': problem_type,
        'recommendation_lines': rec_lines,
        'n_shape_divs': len(shape_divs),
        'n_slope': len(slope_alerts),
        'n_hetero': len(hetero),
        'n_canary': len(canary_alerts),
        'n_topo': len(topo_alerts),
        'n_ood': len(ood_feats),
        'mode': mode,
    }


def _build_report(label_v1, label_v2, score, status,
                  layer1, layer2, layer3, interp,
                  feature_names, mode: str = "retraining") -> str:
    if score is None:
        icon, score_txt = "❔", "N/A"
    else:
        icon = ("🟢" if score >= 85 else "🟡" if score >= 70 else
                "🔶" if score >= 55 else "🔴")
        score_txt = f"{score}"

    # HIERARQUIA DA SAÍDA (v1.47): o diagnóstico e as features afetadas vêm
    # PRIMEIRO; o score global vai para o rodapé, como detalhe técnico.
    # Cinco medições mostraram que o score é o sinal mais fraco: dominado
    # pela sobreposição das janelas (rho=+0.54), não discrimina dentro de
    # estrato (p=0.31-0.53), inverte a ordem num caso medido, satura perto de
    # zero, e a calibração deu AUC 0.533. Ver LIMITATIONS.md secao 3.
    _tipo = interp.get('problem_type', 'NONE')
    _feats = []
    for d in layer1.get('sdd', {}).get('divs', []):
        _feats.append(d['feature'])
    _imp = layer1.get('importance', {}) or {}
    for r in _imp.get('features', []):
        if r.get('alert') == 'IMPORTANCE_COLLAPSE' and r['feature'] not in _feats:
            _feats.append(r['feature'])

    lines = [
        "═" * 66,
        "  MODELGUARD — DIAGNÓSTICO HIERÁRQUICO",
        "═" * 66,
        f"  {icon}  {status}   —   {_tipo}",
        f"  {label_v1}  →  {label_v2}",
        f"  Modo: {'AUDITORIA DE RETREINO' if mode=='retraining' else 'COMPARAÇÃO EXPLORATÓRIA'}",
    ]
    if _feats:
        _n = ", ".join(f"'{f}'" for f in _feats[:5])
        _extra = f" (+{len(_feats)-5})" if len(_feats) > 5 else ""
        lines.append(f"  Features afetadas: {_n}{_extra}")
    else:
        lines.append("  Features afetadas: nenhuma")
    lines += [f"  Features analisadas: {len(feature_names)}", ""]

    # CAMADA 1
    s1 = layer1.get('layer_score', 100)
    n1 = layer1.get('n_alerts', 0)
    icon1 = "✅" if n1==0 else ("⚠️" if n1<=2 else "🔴")
    lines += [
        "┌─────────────────────────────────────────────────────────┐",
        f"│  CAMADA 1 — FORMA LOCAL  {icon1}  Score={s1:.0f}/100          │",
        "├─────────────────────────────────────────────────────────┤",
    ]
    shape_divs = layer1.get('sdd',{}).get('divs',[])
    if shape_divs:
        lines.append(f"│  ⚠ Shape Divergences: {len(shape_divs)}")
        for d in shape_divs[:3]:
            lines.append(f"│    '{d['feature']}': {d['shape_v1']} → {d['shape_v2']}")
    slope_al = layer1.get('slope',{}).get('alerts',[])
    if slope_al:
        lines.append(f"│  ⚠ Slope Alerts: {len(slope_al)}")
        for r in slope_al[:2]:
            lines.append(f"│    '{r['feature']}': {r.get('alert_type','SLOPE')}  "
                         f"ratio={r.get('ratio',0):.1f}x")
    imp1 = layer1.get('importance', {}) or {}
    for r in imp1.get('features', []):
        if r.get('alert') == 'IMPORTANCE_COLLAPSE':
            lines.append(f"│  🔻 IMPORTANCE_COLLAPSE '{r['feature']}': "
                         f"{r['weight_v1']:.3f} → {r['weight_v2']:.3f}")
    if imp1.get('rank_shuffle'):
        lines.append(f"│  ⚠ RANK_SHUFFLE  ρ={imp1.get('rank_rho')}")
    hetero = [r for r in layer1.get('rhale',[])
              if isinstance(r,dict) and r.get('hetero_alert')]
    if hetero:
        lines.append(f"│  ⚠ HETEROGENEOUS_EFFECT: {len(hetero)} feature(s)")
    paired = layer1.get('paired',{})
    if paired.get('rules'):
        lines.append(f"│  📍 Top região de divergência:")
        r = paired['rules'][0]
        lines.append(f"│    {r['rule_text'][:55]}")
    if n1 == 0:
        lines.append("│  ✅ Nenhuma divergência de forma local")
    lines.append("└─────────────────────────────────────────────────────────┘")
    lines.append("")

    # CAMADA 2
    s2 = layer2.get('layer_score')
    n2 = layer2.get('n_alerts', 0)
    icon2 = ("❔" if s2 is None else
             "✅" if n2==0 else ("⚠️" if n2<=1 else "🔴"))
    s2_txt = "N/A" if s2 is None else f"{s2:.0f}"

    lines += [
        "┌─────────────────────────────────────────────────────────┐",
        f"│  CAMADA 2 — GEOMÉTRICA   {icon2}  Score={s2_txt}/100          │",
        "├─────────────────────────────────────────────────────────┤",
    ]
    can = layer2.get('canary',{})
    topo = layer2.get('topology',{})
    geo = layer2.get('geometry',{})
    _fmt = lambda v: f"{v:.1f}/100" if v is not None else "NÃO MEDIDO"
    lines.append(f"│  Canary:   {_fmt(can.get('canary_score'))}  "
                 f"{'⚠ '+', '.join(can.get('alerts',[])) if can.get('alerts') else '✅'}")
    lines.append(f"│  Topology: {_fmt(topo.get('topo_score'))}  "
                 f"{'⚠ '+', '.join(topo.get('alerts',[])) if topo.get('alerts') else '✅'}")
    lines.append(f"│  Geometry: {_fmt(geo.get('geometry_score'))}")
    if not layer2.get('measured', True):
        lines.append("│  ⚠ Camada 2 requer SHAP (só árvores). Use Zest/ALE.")
    if can.get('affected_features'):
        lines.append(f"│  Features afetadas: {can['affected_features'][:2]}")
    if n2 == 0:
        lines.append("│  ✅ Estrutura geométrica e topológica preservada")
    lines.append("└─────────────────────────────────────────────────────────┘")
    lines.append("")

    # CAMADA 3
    s3 = layer3.get('layer_score', 100)
    n3 = layer3.get('n_alerts', 0)
    icon3 = "✅" if n3==0 else "⚠️"
    lines += [
        "┌─────────────────────────────────────────────────────────┐",
        f"│  CAMADA 3 — EXTREMOS/OOD {icon3}  Score={s3:.0f}/100          │",
        "├─────────────────────────────────────────────────────────┤",
    ]
    ood = layer3.get('ood',{})
    ood_feats = ood.get('ood_features',[])
    if ood_feats:
        lines.append(f"│  ⚠ OOD alerts: {', '.join(ood_feats[:3])}")
    else:
        lines.append("│  ✅ Comportamento estável em cenários extremos")
    lines.append("└─────────────────────────────────────────────────────────┘")
    lines.append("")

    # RECOMENDAÇÃO
    lines += [
        "═" * 66,
        "  DIAGNÓSTICO E RECOMENDAÇÃO",
        "═" * 66,
        f"  Tipo de anomalia: {interp['problem_type']}",
        "",
    ]
    for l in interp['recommendation_lines']:
        lines.append(f"  {l}")
    lines += [
        "",
        "─" * 66,
        f"  Detalhe técnico — índice de saúde: {score_txt}/100 "
        f"(camadas: {layer1.get('layer_score')}/"
        f"{layer2.get('layer_score')}/{layer3.get('layer_score')})",
        "  O índice é sensível à configuração do pipeline e não deve ser "
        "usado isoladamente",
        "  para decidir. Use o tipo de anomalia e as features afetadas. "
        "Ver LIMITATIONS.md §3.",
        "═" * 66,
    ]

    return "\n".join(lines)


class ModelGuardOrchestrator:
    """
    Orquestrador hierárquico de todos os detectores ModelGuard.

    Integra 3 camadas de análise:
      Camada 1 (Forma Local):    SDD, ShapeSlope, RHALE, Paired
      Camada 2 (Geométrica):     Canary, GTSF, Topology
      Camada 3 (Extremos/OOD):   OODBehavioralDetector

    Uso:
        orc = ModelGuardOrchestrator()
        result = orc.diagnose(model_v1, model_v2, X_eval, y_eval,
                              features, label_v1="RF v1", label_v2="RF v2")
        print(result['report_text'])
    """

    def __init__(self, threshold: float = 0.30,
                 frechet_threshold: Optional[float] = None):
        self.threshold = threshold
        self.frechet_threshold = frechet_threshold

    def diagnose(self, model_v1, model_v2,
                 X_eval: pd.DataFrame,
                 y_eval: Optional[pd.Series] = None,
                 feature_names: Optional[List[str]] = None,
                 label_v1: str = "Model v1",
                 label_v2: str = "Model v2",
                 mode: str = "retraining",
                 parallel: bool = False,
                 min_rel_importance: float = 0.01) -> dict:
        """
        Diagnóstico hierárquico completo.

        Parâmetros:
            model_v1, model_v2 : modelos com interface predict(X)
            X_eval             : DataFrame de avaliação
            y_eval             : targets (opcional, melhora camada supervisionada)
            feature_names      : lista de features
            label_v1, label_v2 : rótulos para o relatório
            mode               : 'retraining' (default) espera modelos
                                 estruturalmente equivalentes — qualquer
                                 divergência é anomalia.
                                 'exploratory' compara arquiteturas
                                 deliberadamente distintas — divergências
                                 de forma são esperadas e reportadas como
                                 informação, não alerta. Apenas colapso de
                                 importância e poisoning seguem alertando.
            parallel           : True para rodar camadas em paralelo

        Retorna dict com:
            global_score, status, layer1, layer2, layer3,
            interpretation, report_text
        """
        if feature_names is None:
            feature_names = list(X_eval.columns)
        n1 = getattr(model_v1, 'n_features_in_', None)
        n2 = getattr(model_v2, 'n_features_in_', None)
        if n1 is not None and n2 is not None and n1 != n2:
            raise ValueError(
                f"Modelos incompatíveis: {n1} vs {n2} features.")
        if y_eval is None:
            y_eval = pd.Series(np.zeros(len(X_eval)))

        use_shap = _is_tree(model_v1) and _is_tree(model_v2)

        # Roda as 3 camadas
        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
                f1 = ex.submit(_run_layer1, model_v1, model_v2,
                                X_eval, y_eval, feature_names,
                                self.threshold, use_shap,
                                frechet_threshold=self.frechet_threshold)
                f2 = ex.submit(_run_layer2, model_v1, model_v2,
                                X_eval, feature_names)
                f3 = ex.submit(_run_layer3, model_v1, model_v2,
                                X_eval, feature_names)
                layer1 = f1.result()
                layer2 = f2.result()
                layer3 = f3.result()
        else:
            layer1 = _run_layer1(model_v1, model_v2, X_eval, y_eval,
                                  feature_names, self.threshold, use_shap,
                                  frechet_threshold=self.frechet_threshold)
            layer2 = _run_layer2(model_v1, model_v2, X_eval, feature_names)
            layer3 = _run_layer3(model_v1, model_v2, X_eval, feature_names)

        # Score global
        score, status = _global_score(
            layer1['layer_score'],
            layer2['layer_score'],
            layer3['layer_score'],
            mode=mode)

        # Interpretação
        interp = _interpret(layer1, layer2, layer3,
                            score, status, label_v1, label_v2, mode)

        # Relatório
        report = _build_report(label_v1, label_v2, score, status,
                                layer1, layer2, layer3, interp,
                                feature_names, mode)

        # BUG 72 (v1.0.4): `model_collapsed` só existia no audit(); pelo
        # Guard o ramo era código morto, e um v2 que prevê o mesmo valor
        # para todo mundo saía REVISAR ("não é falha de pipeline").
        model_collapsed = _modelo_degenerado(model_v1, model_v2, X_eval)

        return {
            'model_collapsed': model_collapsed,
            'global_score': score,
            'status':       status,
            'layer1':       layer1,
            'layer2':       layer2,
            'layer3':       layer3,
            'interpretation': interp,
            'mode':         mode,
            'report_text':  report,
            'label_v1':     label_v1,
            'label_v2':     label_v2,
        }
