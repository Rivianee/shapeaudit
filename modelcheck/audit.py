"""
audit.py — Interface unificada ModelGuard

Uso:
    from modelcheck import audit
    report = audit(model_v1, model_v2, X_eval, background=X_bg)
    print(report)

Internamente chama:
    - SDD (shape divergences via SHAP para árvores, ALE para outros)
    - ShapeSlopeDetector
    - ShapeSilenceAnalyzer
    - ShapeThresholdCalibrator (se calibrate=True)
    Health Score unificado 0-100
"""
import numpy as np
import pandas as pd
from typing import Optional, List

def _is_tree_model(model) -> bool:
    cls = type(model).__name__.lower()
    return any(k in cls for k in
               ['forest','tree','boost','xgb','lgb','catboost','extra'])

def audit(model_v1,
          model_v2,
          X_eval:      pd.DataFrame,
          background:  Optional[pd.DataFrame] = None,
          y_eval:      Optional[pd.Series] = None,
          feature_names: Optional[List[str]] = None,
          calibrate:   bool = False,
          representation: str = "auto",
          min_rel_importance: float = 0.01,
          threshold:   float = 0.30,
          verbose:     bool = True,
          confirm_bootstrap: bool = False,
          min_concordance: float = 0.85,
          n_boot_shape: int = 12,
          detector: str = "frechet",
          frechet_threshold: float = None) -> dict:
    """
    Auditoria completa de shape instability entre dois modelos.

    Parâmetros:
        model_v1, model_v2 : modelos com interface predict(X)
        X_eval             : DataFrame de avaliação
        background         : background para SHAP (opcional; usa amostra de X_eval se None)
        feature_names      : lista de features (default: todas as colunas de X_eval)
        min_rel_importance : ignora features com importância relativa abaixo
                             deste valor. Sem o filtro, ruído gera divergência
                             de forma espúria (medido em feature com 0.04% da
                             importância). Use 0.0 para desativar.
        representation     : 'shap'  — TreeExplainer (mais sensível: 93%
                             recall, F1=0.80, porém só árvores)
                             'ale'   — Accumulated Local Effects (universal,
                             porém menos sensível: F1=0.59)
                             'auto'  — SHAP para árvores, ALE caso contrário
        min_rel_importance : ignora features com importância relativa abaixo
                             deste valor. Sem o filtro, ruído gera divergência
                             de forma espúria (medido em feature com 0.04% da
                             importância). Use 0.0 para desativar.
        representation     : 'shap'  — TreeExplainer, mais sensível
                                       (recall 93%, F1 0.80) mas só árvores
                             'ale'   — model-agnostic, menos sensível
                                       (F1 0.59), funciona em qualquer modelo
                             'auto'  — SHAP se disponível, ALE caso contrário
        calibrate          : calibra o threshold automaticamente via bootstrap
        threshold          : threshold manual (ignorado se calibrate=True)
        verbose            : imprime o relatório

    Retorna dict com:
        health_score, status, divergences, slope_alerts,
        silence_categories, method_used, threshold_used, report_text
    """
    from modelcheck.ale_detector import ALEShapeDetector
    from modelcheck.shape import _shape_profile
    from modelcheck.slope import ShapeSlopeDetector
    from modelcheck.silence import ShapeSilenceAnalyzer

    from modelcheck._validate import validate_inputs
    X_eval, feature_names = validate_inputs(
        model_v1, model_v2, X_eval, feature_names, caller='audit')

    # ── Validação de compatibilidade (FIX: antes retornava score=0) ──
    n1 = getattr(model_v1, 'n_features_in_', None)
    n2 = getattr(model_v2, 'n_features_in_', None)
    # Compara o CONJUNTO de nomes quando disponível — mais informativo que
    # a contagem: dois modelos podem ter 3 features cada e serem
    # incompatíveis por usarem features diferentes.
    fn1 = getattr(model_v1, 'feature_names_in_', None)
    fn2 = getattr(model_v2, 'feature_names_in_', None)
    if fn1 is not None and fn2 is not None:
        s1, s2 = set(map(str, fn1)), set(map(str, fn2))
        if s1 != s2:
            faltam = sorted(s1 - s2)[:5]
            sobram = sorted(s2 - s1)[:5]
            raise ValueError(
                f"Modelos treinados com features diferentes. "
                f"Só em v1: {faltam or '—'}. Só em v2: {sobram or '—'}. "
                f"Não é possível comparar.")
    if n1 is not None and n2 is not None and n1 != n2:
        raise ValueError(
            f"Modelos incompatíveis: features diferentes em número — "
            f"v1 espera {n1}, v2 espera {n2}. Não é possível comparar.")
    if n1 is not None and n1 != len(feature_names):
        raise ValueError(
            f"X_eval tem {len(feature_names)} features mas o modelo "
            f"espera {n1}.")

    # ── Aviso de amostra insuficiente ───────────────────────────────
    warnings_list = []

    # ── Validação de dtypes (bug achado no dataset ENA do ONS) ──────
    # Colunas numéricas vindas como string causavam TypeError dentro dos
    # detectores, silenciado por except genérico no pipeline.
    non_num = [f for f in feature_names
               if f in X_eval.columns
               and not pd.api.types.is_numeric_dtype(X_eval[f])]
    if non_num:
        raise TypeError(
            f"Features não numéricas em X_eval: {non_num[:6]}. "
            f"Converta com pd.to_numeric() antes de auditar "
            f"(atenção a separador decimal e datas em formato misto).")
    inf_cols = []
    for f in feature_names:
        if f in X_eval.columns:
            try:
                if np.isinf(X_eval[f].to_numpy(dtype=float)).any():
                    inf_cols.append(f)
            except Exception:
                # coluna não numérica: não há infinito a checar. O silêncio
                # aqui é correto e local — não corresponde a nenhuma medição
                # que deixou de acontecer.
                continue     # noqa: lacuna-nao-aplicavel
    if inf_cols:
        warnings_list.append(
            f"INF_PRESENT: valores infinitos em {inf_cols[:4]}. "
            f"Perfis podem ficar distorcidos.")
    if len(X_eval) < 30:
        warnings_list.append(
            f"LOW_N: apenas {len(X_eval)} amostras (mínimo recomendado: 30). "
            f"Resultados podem ser falsos positivos.")
    if X_eval.isna().any().any():
        n_na = int(X_eval.isna().sum().sum())
        warnings_list.append(
            f"NAN_PRESENT: {n_na} valores ausentes em X_eval. "
            f"Perfis podem ser distorcidos.")

    if background is None:
        n_bg = min(100, len(X_eval))
        background = X_eval.sample(n_bg, random_state=42)

    # ── Calibração automática ─────────────────────────────────────
    tau = threshold
    if calibrate:
        try:
            from modelcheck.calibration import ShapeThresholdCalibrator
            cal = ShapeThresholdCalibrator()
            # assinatura real: fit(X, y, feature_names)
            y_cal = y_eval if y_eval is not None else None
            if y_cal is None:
                raise ValueError(
                    "calibrate=True exige y_eval para calibrar o threshold")
            cal.fit(X_eval, y_cal, feature_names)
            tau_cal = getattr(cal, 'global_threshold_', None)
            if tau_cal is None:
                tau_cal = getattr(cal, 'threshold_', None)
            if tau_cal is not None:
                tau = float(tau_cal)
            else:
                warnings_list.append(
                    "CALIB_FAILED: calibrador não expôs threshold; "
                    f"usando manual tau={threshold}")
        except Exception as e:
            warnings_list.append(
                f"CALIB_FAILED: {type(e).__name__}: {str(e)[:70]}. "
                f"Usando tau={threshold}")

    _falhas = []    # lacunas desta auditoria (política v1.66)


    # ── Detecção de modelo degenerado ────────────────────────────────
    # Um modelo que virou constante (previsão de variância ~zero) é um
    # COLAPSO, não uma mudança de forma. Sem esta checagem, o diagnóstico
    # dizia "divergência de forma em a e b" e o engenheiro iria investigar
    # as features — quando o problema é o modelo inteiro.
    def _degenerado(mdl):
        try:
            p = np.asarray(mdl.predict(X_eval), dtype=float)
            if len(p) < 2:
                return False, 0.0
            rng_p = float(np.ptp(p))
            ref = float(np.abs(p).mean()) + 1e-12
            return (rng_p / ref) < 1e-6, rng_p
        except Exception:
            # modelo sem predict utilizável: a checagem de degenerado não se
            # aplica, e a comparação principal já valida isso antes.
            return False, 0.0     # noqa: lacuna-nao-aplicavel

    deg1, rng1 = _degenerado(model_v1)
    deg2, rng2 = _degenerado(model_v2)
    if deg2 and not deg1:
        warnings_list.append(
            "MODEL_COLLAPSED: o modelo v2 prevê valor praticamente "
            f"constante (amplitude={rng2:.3g}). Isto é colapso total do "
            "modelo, não mudança de forma de features. Verifique o "
            "treinamento antes de investigar features individuais.")
    elif deg1 and not deg2:
        warnings_list.append(
            "MODEL_COLLAPSED_V1: o modelo v1 (referência) prevê valor "
            f"praticamente constante (amplitude={rng1:.3g}). A comparação "
            "não é informativa.")
    elif deg1 and deg2:
        warnings_list.append(
            "BOTH_MODELS_COLLAPSED: ambos os modelos preveem valor "
            "constante. Nenhuma comparação estrutural é possível.")

    # ── Decide método de representação ────────────────────────────
    rep = (representation or "auto").lower()
    if rep not in ("auto", "shap", "ale"):
        raise ValueError("representation deve ser 'auto', 'shap' ou 'ale'")
    is_tree = _is_tree_model(model_v1) and _is_tree_model(model_v2)
    if rep == "ale":
        use_shap = False
    elif rep == "shap":
        use_shap = True
        if not is_tree:
            warnings_list.append(
                "SHAP_FORCED: representation='shap' em modelo não-árvore; "
                "TreeExplainer provavelmente falhará e cairá para ALE.")
    else:
        use_shap = is_tree
    method = ("SHAP (TreeExplainer, F1=0.80, recall 93%)" if use_shap
              else "ALE (model-agnostic, F1=0.59)")

    shape_divs = []
    sv1 = sv2 = None

    # ── Alinhamento por NOME (bug 15, v1.44) ─────────────────────────
    # O SHAP devolve colunas na ordem de TREINO de cada modelo. Se v1 foi
    # treinado com (a,b,c) e v2 com (c,b,a), indexar sv1[:,i] e sv2[:,i] com
    # o mesmo i compara 'a' de v1 contra 'c' de v2 — divergência 100%
    # artificial. Medido: score 10 e 2 divergências espúrias com os MESMOS
    # dados, só reordenando as colunas de treino.
    # Cenário real: pipeline de feature engineering que reordena colunas.
    def _order_of(m):
        o = getattr(m, 'feature_names_in_', None)
        return [] if o is None else [str(x) for x in list(o)]
    order1 = _order_of(model_v1)
    order2 = _order_of(model_v2)
    if order1 and order2:
        if set(order1) != set(order2):
            raise ValueError(
                f"Modelos treinados com conjuntos de features diferentes. "
                f"Só em v1: {sorted(set(order1)-set(order2))[:5]}; "
                f"só em v2: {sorted(set(order2)-set(order1))[:5]}.")
        if order1 != order2:
            warnings_list.append(
                "FEATURE_ORDER_DIFFERS: os modelos foram treinados com as "
                "colunas em ordens diferentes. As atribuições foram alinhadas "
                "por NOME.")

    def _align(sv, order, target):
        """Reordena as colunas de sv da ordem do modelo para `target`."""
        if not order or list(order) == list(target):
            return sv
        idx = [order.index(f) for f in target if f in order]
        return sv[:, idx]

    if use_shap:
        try:
            import shap
            X1 = X_eval[order1] if order1 else X_eval
            X2 = X_eval[order2] if order2 else X_eval
            sv1 = shap.TreeExplainer(model_v1).shap_values(X1)
            sv2 = shap.TreeExplainer(model_v2).shap_values(X2)
            # garante 2D
            if sv1.ndim == 3: sv1 = sv1[:,:,1]
            if sv2.ndim == 3: sv2 = sv2[:,:,1]
            # alinha ambos à ordem de `feature_names`
            sv1 = _align(sv1, order1, feature_names)
            sv2 = _align(sv2, order2, feature_names)

            # FILTRO DE IMPORTÂNCIA (v1.37): sem ele, features irrelevantes
            # geram divergência de forma espúria. Medido: feature de ruído
            # com 0.04% da importância apontada como divergente tanto no
            # controle quanto no caso com causa real. O perfil SHAP de ruído
            # é instável, então a classificação de forma vira sozinha.
            # Mesma guarda que o ImportanceCollapseDetector já tinha.
            # v0.10.0: por padrão, a distância de Fréchet detecta e a classe
            # nomeia (modelcheck/frechet.py). detector='classe' mantém o
            # critério anterior.
            if detector == 'frechet':
                from modelcheck.frechet import divergencias_forma, LIMIAR_PADRAO
                shape_divs = divergencias_forma(
                    X_eval, sv1, sv2, feature_names,
                    min_rel_importance=min_rel_importance,
                    limiar=frechet_threshold or LIMIAR_PADRAO)
            else:
                tot1 = float(np.abs(sv1).mean(axis=0).sum()) + 1e-12
                tot2 = float(np.abs(sv2).mean(axis=0).sum()) + 1e-12
                for i, f in enumerate(feature_names):
                    imp1 = float(np.abs(sv1[:,i]).mean())
                    imp2 = float(np.abs(sv2[:,i]).mean())
                    if max(imp1/tot1, imp2/tot2) < min_rel_importance:
                        continue
                    p1 = _shape_profile(X_eval[f].values, sv1[:,i])
                    p2 = _shape_profile(X_eval[f].values, sv2[:,i])
                    if not p1 or not p2: continue
                    if p1['shape'] == p2['shape']: continue
                    spread = abs(imp1-imp2) / (max(imp1,imp2)+1e-8)
                    # v1.59: o critério de confirmação era `spread >= tau`, que
                    # exigia mudança de IMPORTÂNCIA junto com a mudança de forma.
                    # Medido em 160 pares com importância preservada por
                    # construção: detectava 6% (F1 0.12). A confirmação por
                    # reamostragem detecta 100% com 8% de FP (F1 0.96).
                    # O spread continua reportado como informação.
                    if confirm_bootstrap:
                        from modelcheck.shape import ShapeDivergenceDetector as _S
                        f1s, c1 = _S._forma_estavel(X_eval[f].values, sv1[:, i],
                                                    n_boot=n_boot_shape, seed=11)
                        f2s, c2 = _S._forma_estavel(X_eval[f].values, sv2[:, i],
                                                    n_boot=n_boot_shape, seed=23)
                        confirmado = (f1s is not None and f2s is not None
                                      and c1 >= min_concordance
                                      and c2 >= min_concordance and f1s != f2s)
                        forma1, forma2 = (f1s, f2s) if confirmado else (p1['shape'],
                                                                        p2['shape'])
                    else:
                        confirmado = spread >= tau
                        forma1, forma2 = p1['shape'], p2['shape']
                    if confirmado:
                        shape_divs.append({
                            'feature': f,
                            'shape_v1': forma1,
                            'shape_v2': forma2,
                            'spread': round(spread, 3),
                        })
        except Exception as e:
            use_shap = False
            method = f"ALE (SHAP falhou: {e})"
            _falhas.append({
                'layer': 'shape divergence (SHAP)',
                'reason': f'{type(e).__name__}: {e}'[:160],
                'consequence': ('caiu para ALE; os valores não são '
                                'comparáveis em escala com execuções SHAP'),
            })

    if not use_shap:
        det = ALEShapeDetector(n_bins=5, n_boot=12, min_confidence=0.6)
        results = det.compare(model_v1, model_v2, X_eval, feature_names)
        for r in results:
            if r['divergence']:
                shape_divs.append({
                    'feature':  r['feature'],
                    'shape_v1': r['shape1'],
                    'shape_v2': r['shape2'],
                    'spread':   round((r['conf1']+r['conf2'])/2, 3),
                })

    # ── Slope Detector ────────────────────────────────────────────
    slope_alerts = []
    if use_shap and sv1 is not None:
        try:
            slope_det = ShapeSlopeDetector(threshold_ratio=3.0)
            slope_res = slope_det.check(sv1, sv2, X_eval, feature_names)
            slope_alerts = [r for r in slope_res if r.get('divergence')]
        except Exception as e:
            _falhas.append({
                'layer': 'slope divergence',
                'reason': f'{type(e).__name__}: {e}'[:160],
                'consequence': ('não conseguimos comparar a inclinação das curvas '
                                'neste par'),
            })

    # ── Silence Analyzer ─────────────────────────────────────────
    silence_cats = {}
    if use_shap and sv1 is not None:
        try:
            sil = ShapeSilenceAnalyzer()
            sil_res = sil.analyze(sv1, sv2, X_eval, feature_names)
            silence_cats = {r['feature']: r.get('silence_type','UNKNOWN')
                            for r in sil_res
                            if r.get('silence_type') not in (None,'STABLE')}
        except Exception as e:
            _falhas.append({
                'layer': 'silence analysis',
                'reason': f'{type(e).__name__}: {e}'[:160],
                'consequence': ('variáveis que ficaram silenciosas não foram '
                                'identificadas'),
            })

    # ── Health Score unificado ────────────────────────────────────
    n_shape  = len(shape_divs)
    n_slope  = len(slope_alerts)
    n_dist   = sum(1 for v in silence_cats.values() if v == 'DIST_MISMATCH')

    # FIX (v1.22): penalidade linear zerava o score em datasets de alta
    # dimensionalidade. Com 30 features, ruído de seed já gerava
    # divergências suficientes para levar o score a 0.
    # Solução: penalidade sobre a FRAÇÃO de features divergentes, com
    # saturação exponencial (nunca chega a 0 abruptamente).
    n_feat = max(len(feature_names), 1)
    frac_shape = n_shape / n_feat
    frac_slope = n_slope / n_feat
    frac_dist  = n_dist  / n_feat
    penalty = (2.2 * frac_shape + 1.2 * frac_slope + 0.5 * frac_dist)
    score = int(round(100 * float(np.exp(-penalty))))
    status = ("HEALTHY"   if score >= 85 else
              "ATTENTION" if score >= 65 else "CRITICAL")

    # ── Relatório em linguagem de negócio ─────────────────────────
    lines = [
        "=" * 60,
        "  MODELGUARD AUDIT REPORT",
        "=" * 60,
        f"  Method:       {method}",
        f"  Threshold:    {tau:.2f}{'  (calibrated)' if calibrate else ''}",
        f"  Features:     {len(feature_names)}",
        f"  Eval samples: {len(X_eval)}",
        "",
        f"  Health Score: {score}/100  [{status}]",
        "",
    ]

    if not shape_divs and not slope_alerts and not silence_cats:
        lines.append("  ✅ No structural divergences detected.")
        lines.append("     Both models learned equivalent relationships.")
        # Ausência de alerta é explicada, nunca deixada em branco
        if not use_shap:
            lines.append("     (método ALE: verifique se n é suficiente por bin)")
    else:
        if shape_divs:
            lines.append(f"  ⚠  Shape Divergences ({n_shape}):")
            for d in shape_divs:
                lines.append(f"     '{d['feature']}': "
                             f"{d['shape_v1']} → {d['shape_v2']}  "
                             f"(spread={d['spread']:.2f})")
        if slope_alerts:
            lines.append(f"\n  ⚠  Slope Alerts ({n_slope}):")
            for r in slope_alerts:
                lines.append(f"     '{r['feature']}': "
                             f"{r.get('alert_type','SLOPE_DIV')}  "
                             f"ratio={r.get('ratio',0):.1f}x")
        if silence_cats:
            lines.append(f"\n  ℹ  Silence Categories:")
            for f, cat in silence_cats.items():
                if cat != 'STABLE':
                    lines.append(f"     '{f}': {cat}")

    if deg1 or deg2:
        lines.insert(8, "  🔻 MODELO DEGENERADO — previsão constante. "
                        "As divergências abaixo são consequência, não causa.")
    if warnings_list:
        lines.append("")
        for w in warnings_list:
            lines.append(f"  ⚠ {w}")
    lines += ["", "=" * 60]
    report_text = "\n".join(lines)

    if verbose:
        print(report_text)

    return {
        'health_score':       score,
        'status':             status,
        'divergences':        shape_divs,
        'slope_alerts':       slope_alerts,
        'silence_categories': silence_cats,
        'method_used':        method,
        'representation':     'shap' if use_shap else 'ale',
        'threshold_used':     tau,
        'warnings':           warnings_list,
        'model_collapsed':    bool(deg1 or deg2),
        '_failures':          _falhas,
        'report_text':        report_text,
    }
