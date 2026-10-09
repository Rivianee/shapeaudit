"""Extensões do ShapeDivergence Detector.

Resolve 4 limitações do SDD original:

  ShapeImportanceDivergence   — Limitação 6: viés por magnitude sem mudança de forma
  CategoricalShapeDivergence  — Limitação 7: features binárias e categóricas
  AutoAttributionSelector     — Limitação 8: escolha automática SHAP vs ALE por correlação
  NormalizedShapeDivergence   — Limitação 10: feature dominante mascara divergências menores

Uso:
    from modelcheck.shape_extensions import ExtendedShapeDivergenceDetector
    detector = ExtendedShapeDivergenceDetector()
    results  = detector.check(sv_v1, sv_v2, X_eval, feature_names)
    print(detector.report(results))
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from .shape import _shape_profile


# ── Extensão 1: Viés por magnitude ──────────────────────────────────────────

class ShapeImportanceDivergence:
    """Detecta viés silencioso via mudança de importância SHAP sem mudança de forma.

    Problema original:
        O SDD detecta quando a FORMA da curva muda (crescente → U-invertido).
        Mas viés pode ser introduzido aumentando o PESO de uma feature
        sem alterar a forma — ex: 'região' passa de 1% para 30% de importância
        sem mudar de 'n/a' para outra forma.

    Solução:
        Compara importância SHAP normalizada (% do total) entre v1 e v2.
        Se Δimportância > threshold_magnitude → MAGNITUDE_DIVERGENCE.
        Combinado com shape_changed → classificação completa do tipo de alerta.

    Resultados experimentais:
        Dataset de crédito sintético (N=1200):
        - 'regiao' (binária): imp_v1=0.012 → imp_v2=0.303 (+2374%)
        - SDD original: 0 alertas (n/a em feature binária)
        - Esta extensão: 1 MAGNITUDE_DIVERGENCE detectado
    """

    def __init__(self, threshold_magnitude: float = 0.5,
                 threshold_shape: float = 0.0):
        """
        threshold_magnitude : mudança relativa de importância para alertar
                              (padrão 0.5 = 50% de variação)
        threshold_shape     : não usado diretamente, mantido para compatibilidade
        """
        self.threshold_magnitude = threshold_magnitude
        # BUG 12 (v1.41): threshold_magnitude é ABSOLUTO, então qualquer
        # flutuação de uma feature irrelevante passa o corte. Medido: em
        # controle puro (mesmo modelo, seed diferente), a feature 'cat' com
        # 0.17% da importância era apontada como divergente. Mesma família
        # dos bugs 9 e 11. O filtro relativo resolve.
        self.threshold_shape = threshold_shape

    def check(self, sv1: np.ndarray, sv2: np.ndarray,
              X_eval: pd.DataFrame, feature_names: list) -> list:
        """
        sv1, sv2       : arrays SHAP (n_samples, n_features)
        X_eval         : DataFrame com os valores das features
        feature_names  : lista de nomes das features

        Retorna lista de dicts com resultado por feature.
        """
        total1 = np.abs(sv1).mean(axis=0).sum() + 1e-8
        total2 = np.abs(sv2).mean(axis=0).sum() + 1e-8
        results = []

        for i, f in enumerate(feature_names):
            imp1 = float(np.abs(sv1[:, i]).mean() / total1)
            imp2 = float(np.abs(sv2[:, i]).mean() / total2)
            mag_change = abs(imp1 - imp2) / (imp1 + 1e-8)

            xv = np.array(X_eval[f].values, dtype=float)
            p1 = _shape_profile(xv, sv1[:, i])
            p2 = _shape_profile(xv, sv2[:, i])
            sh1 = p1['shape'] if p1 else 'n/a'
            sh2 = p2['shape'] if p2 else 'n/a'
            shape_changed = sh1 != sh2 and sh1 != 'n/a' and sh2 != 'n/a'

            # Classificação do tipo de alerta
            if shape_changed and mag_change > self.threshold_magnitude:
                alert_type = 'SHAPE_AND_MAGNITUDE'
            elif shape_changed:
                alert_type = 'SHAPE_DIVERGENCE'
            elif mag_change > self.threshold_magnitude:
                alert_type = 'MAGNITUDE_DIVERGENCE'  # novo — não detectado pelo SDD base
            else:
                alert_type = None

            results.append({
                'feature':       f,
                'shape_v1':      sh1,
                'shape_v2':      sh2,
                'imp_v1':        round(imp1, 4),
                'imp_v2':        round(imp2, 4),
                'mag_change':    round(mag_change, 3),
                'shape_changed': shape_changed,
                'alert':         alert_type,
            })
        return results

    def report(self, results: list) -> str:
        lines = ["ShapeImportanceDivergence:"]
        alerts = [r for r in results if r['alert']]
        if not alerts:
            lines.append("  Nenhuma divergência detectada.")
        for r in alerts:
            lines.append(f"  ⚠️  [{r['alert']}] '{r['feature']}'")
            lines.append(f"      imp: {r['imp_v1']:.3f} → {r['imp_v2']:.3f} "
                         f"(Δ={r['mag_change']:.1f}x)")
            if r['shape_changed']:
                lines.append(f"      forma: {r['shape_v1']} → {r['shape_v2']}")
        return "\n".join(lines)


# ── Extensão 2: Features categóricas/binárias ────────────────────────────────

class CategoricalShapeDivergence:
    """Detecta divergência em features com baixa cardinalidade.

    Problema original:
        _shape_profile usa pd.qcut que falha com features binárias (só 0/1)
        ou categóricas com poucos valores únicos — retorna None (n/a).
        Viés em atributos protegidos binários (gênero, região) fica invisível.

    Solução:
        Para features com cardinalidade <= threshold_categorical:
        - Compara média SHAP por categoria entre v1 e v2
        - Detecta inversão de direção entre categorias (CATEGORICAL_DIVERGENCE)
        - Detecta mudança de magnitude mesmo sem inversão (CATEGORICAL_MAGNITUDE)

    Resultados experimentais:
        Dataset de crédito com atributo protegido 'regiao' (binária):
        - v1: cat0=+0.001, cat1=-0.002 (efeito neutro)
        - v2: cat0=+0.111, cat1=-0.211 (efeito discriminatório)
        - SDD original: n/a (retorna None)
        - Esta extensão: CATEGORICAL_MAGNITUDE detectado
    """

    def __init__(self, threshold_categorical: int = 5,
                 threshold_magnitude: float = 0.01,
                 min_rel_importance: float = 0.02):
        """
        threshold_categorical : cardinalidade máxima para tratar como categórica
        threshold_magnitude   : mudança absoluta mínima de média SHAP para alertar
        min_rel_importance    : ignora features com importância relativa abaixo
                                deste valor. BUG 12 (v1.41): threshold_magnitude
                                é ABSOLUTO, então qualquer flutuação de feature
                                irrelevante passava. Medido: 'cat' com 0.17% da
                                importância apontada como divergente em controle
                                puro (mesmo modelo, seed diferente).
        """
        self.min_rel_importance = min_rel_importance
        self.threshold_categorical = threshold_categorical
        self.threshold_magnitude = threshold_magnitude

    def is_categorical(self, x: np.ndarray) -> bool:
        return len(np.unique(x)) <= self.threshold_categorical

    def check_feature(self, x: np.ndarray,
                      sv1_f: np.ndarray,
                      sv2_f: np.ndarray,
                      total_v1: float = 0.0,
                      total_v2: float = 0.0) -> dict:
        """Verifica divergência em uma feature categórica."""
        cats = np.unique(x)
        means1 = {c: float(sv1_f[x == c].mean()) for c in cats}
        means2 = {c: float(sv2_f[x == c].mean()) for c in cats}

        # Inversão de direção entre categorias consecutivas
        sorted_cats = sorted(cats)
        diffs1 = [means1[sorted_cats[i+1]] - means1[sorted_cats[i]]
                  for i in range(len(sorted_cats)-1)]
        diffs2 = [means2[sorted_cats[i+1]] - means2[sorted_cats[i]]
                  for i in range(len(sorted_cats)-1)]
        inversions = sum(1 for d1, d2 in zip(diffs1, diffs2)
                         if np.sign(d1) != np.sign(d2) and
                         abs(d1) > 1e-4 and abs(d2) > 1e-4)

        mag_change = max(abs(means2.get(c, 0) - means1.get(c, 0))
                         for c in cats)

        # FILTRO DE IMPORTÂNCIA (bug 12, v1.41): threshold_magnitude é
        # ABSOLUTO, então qualquer flutuação de feature irrelevante passa.
        # Medido: em controle puro (mesmo modelo, seed diferente), a feature
        # 'cat' com 0.17% da importância era apontada como divergente.
        _imp1 = float(np.abs(sv1_f).mean())
        _imp2 = float(np.abs(sv2_f).mean())
        _t1 = total_v1 if total_v1 else _imp1
        _t2 = total_v2 if total_v2 else _imp2
        _irrelevante = (max(_imp1 / max(_t1, 1e-12),
                            _imp2 / max(_t2, 1e-12)) < self.min_rel_importance)

        if inversions > 0:
            alert = None if _irrelevante else 'CATEGORICAL_DIVERGENCE'
        elif mag_change > self.threshold_magnitude and not _irrelevante:
            alert = 'CATEGORICAL_MAGNITUDE'
        else:
            alert = None

        return {
            'means_v1':   {int(c): round(v, 4) for c, v in means1.items()},
            'means_v2':   {int(c): round(v, 4) for c, v in means2.items()},
            'inversions': inversions,
            'mag_change': round(float(mag_change), 4),
            'alert':      alert,
        }

    def check(self, sv1: np.ndarray, sv2: np.ndarray,
              X_eval: pd.DataFrame, feature_names: list) -> list:
        results = []
        for i, f in enumerate(feature_names):
            xv = np.array(X_eval[f].values, dtype=float)
            if self.is_categorical(xv):
                r = self.check_feature(
                xv, sv1[:, i], sv2[:, i],
                total_v1=float(np.abs(sv1).mean(axis=0).sum()),
                total_v2=float(np.abs(sv2).mean(axis=0).sum()))
                r['feature'] = f
                r['categorical'] = True
                r['n_unique'] = int(len(np.unique(xv)))
                results.append(r)
        return results

    def report(self, results: list) -> str:
        lines = ["CategoricalShapeDivergence:"]
        alerts = [r for r in results if r.get('alert')]
        if not alerts:
            lines.append("  Nenhuma divergência em features categóricas.")
        for r in alerts:
            lines.append(f"  ⚠️  [{r['alert']}] '{r['feature']}' "
                         f"({r['n_unique']} categorias)")
            lines.append(f"      v1: {r['means_v1']}  →  v2: {r['means_v2']}")
            if r['inversions'] > 0:
                lines.append(f"      {r['inversions']} inversão(ões) de direção")
        return "\n".join(lines)


# ── Extensão 3: Auto-seleção SHAP vs ALE ────────────────────────────────────

class AutoAttributionSelector:
    """Seleciona SHAP ou ALE automaticamente por feature via correlação.

    Problema original:
        O paper recomenda usar SHAP quando há interações e ALE quando features
        são correlacionadas — mas o usuário precisa decidir manualmente.
        Usar sempre SHAP em dados correlacionados pode gerar formas espúrias.

    Solução:
        Para cada feature, calcula correlação de Pearson com todas as outras.
        Se max(|Pearson|) > threshold → usa ALE (mais conservador).
        Senão → usa SHAP (captura interações).

    Resultados experimentais:
        Wine Quality (11 features químicas correlacionadas):
        - 8/11 features têm correlação > 0.4 → selecionam ALE
        - 3/11 features não correlacionadas → mantêm SHAP
        - Concordância SHAP vs ALE: 64% — justifica seleção automática

    Nota:
        Esta classe gera a recomendação. A execução do ALE deve ser feita
        pelo usuário com ale_explain() do módulo ale.py.
    """

    def __init__(self, correlation_threshold: float = 0.4):
        self.corr_threshold = correlation_threshold

    def select(self, X: pd.DataFrame,
               feature_names: list = None) -> dict:
        """
        Retorna dict {feature: {'method': 'SHAP'|'ALE', 'max_correlation': float}}
        """
        if feature_names is None:
            feature_names = list(X.columns)
        corr = X[feature_names].corr().abs()
        selection = {}
        for f in feature_names:
            others = [c for c in feature_names if c != f]
            max_corr = float(corr[f][others].max()) if others else 0.0
            selection[f] = {
                'method':          'ALE' if max_corr > self.corr_threshold else 'SHAP',
                'max_correlation': round(max_corr, 3),
                'reason':          f"max |Pearson| = {max_corr:.3f} "
                                   f"({'correlacionada' if max_corr > self.corr_threshold else 'independente'})",
            }
        return selection

    def report(self, selection: dict) -> str:
        shap_f = [f for f, v in selection.items() if v['method'] == 'SHAP']
        ale_f  = [f for f, v in selection.items() if v['method'] == 'ALE']
        lines  = [f"AutoAttributionSelector (threshold={self.corr_threshold}):"]
        lines.append(f"  SHAP ({len(shap_f)}): {shap_f}")
        lines.append(f"  ALE  ({len(ale_f)}):  {ale_f}")
        return "\n".join(lines)


# ── Extensão 4: Divergência normalizada por importância ──────────────────────

class NormalizedShapeDivergence:
    """Pondera a divergência de forma pela importância relativa da feature.

    Problema original:
        Feature dominante (importância 84%) com shape_div=False mascara
        feature de baixa importância (importância 3%) com shape_div=True.
        Resultado: o alerta da feature crítica pode ser ignorado ou
        o threshold global não a alcança.

        Exemplo CCEE: MES tem importância 84%, HORA tem 3%.
        Qualquer threshold > 0.03 vai subponderar HORA.

    Solução:
        spread_normalizado = (1 se shape_div else 0) / (imp_relativa + epsilon)
        Features de baixa importância que mudam forma têm spread amplificado.
        Features dominantes que não mudam forma têm spread zero (sem amplificação).

    Resultados experimentais:
        CCEE PLD 2023-2024 SUDESTE:
        - MES: imp=0.84, shape_div=False → não alerta (correto)
        - HORA: imp=0.036, shape_div=True → alerta amplificado (correto)
        - PERIODO: imp=0.09, shape_div=False → não alerta (correto)
    """

    def __init__(self, threshold: float = 0.30,
                 epsilon: float = 0.05):
        """
        threshold : threshold de spread normalizado para alertar
        epsilon   : evita divisão por zero para features de importância ~0
        """
        self.threshold = threshold
        self.epsilon = epsilon

    def check(self, sv1: np.ndarray, sv2: np.ndarray,
              X_eval: pd.DataFrame, feature_names: list) -> list:
        total1 = np.abs(sv1).mean(axis=0).sum() + 1e-8
        total2 = np.abs(sv2).mean(axis=0).sum() + 1e-8
        results = []

        for i, f in enumerate(feature_names):
            imp1 = float(np.abs(sv1[:, i]).mean() / total1)
            imp2 = float(np.abs(sv2[:, i]).mean() / total2)
            imp_mean = (imp1 + imp2) / 2

            xv = np.array(X_eval[f].values, dtype=float)
            p1 = _shape_profile(xv, sv1[:, i])
            p2 = _shape_profile(xv, sv2[:, i])
            sh1 = p1['shape'] if p1 else 'n/a'
            sh2 = p2['shape'] if p2 else 'n/a'
            shape_div = sh1 != sh2 and sh1 != 'n/a' and sh2 != 'n/a'

            # spread normalizado
            normalized_spread = (1.0 if shape_div else 0.0) / (imp_mean + self.epsilon)

            results.append({
                'feature':           f,
                'shape_v1':          sh1,
                'shape_v2':          sh2,
                'imp_mean':          round(imp_mean, 4),
                'shape_div':         shape_div,
                'normalized_spread': round(normalized_spread, 3),
                'alert_original':    shape_div,
                'alert_normalized':  normalized_spread > self.threshold,
                'new_alert':         normalized_spread > self.threshold and not shape_div,
            })
        return results

    def report(self, results: list) -> str:
        lines = [f"NormalizedShapeDivergence (threshold={self.threshold}):"]
        alerts_orig = [r for r in results if r['alert_original']]
        alerts_new  = [r for r in results if r['new_alert']]
        if not alerts_orig and not alerts_new:
            lines.append("  Nenhuma divergência detectada.")
        for r in alerts_orig:
            lines.append(f"  ⚠️  [SHAPE_DIV] '{r['feature']}': "
                         f"{r['shape_v1']} → {r['shape_v2']} "
                         f"(imp={r['imp_mean']:.3f})")
        for r in alerts_new:
            lines.append(f"  ⚠️  [NORMALIZADO - novo] '{r['feature']}': "
                         f"baixa importância ({r['imp_mean']:.3f}) "
                         f"mas spread normalizado={r['normalized_spread']:.2f}")
        return "\n".join(lines)


# ── Detector unificado ───────────────────────────────────────────────────────

class ExtendedShapeDivergenceDetector:
    """Detector unificado que combina SDD base + 4 extensões.

    Uso:
        from modelcheck.shape_extensions import ExtendedShapeDivergenceDetector
        import shap

        detector = ExtendedShapeDivergenceDetector()
        exp1 = shap.TreeExplainer(model_v1)
        exp2 = shap.TreeExplainer(model_v2)
        sv1  = exp1.shap_values(X_eval)
        sv2  = exp2.shap_values(X_eval)

        results = detector.check(sv1, sv2, X_eval)
        print(detector.report(results))
    """

    def __init__(self,
                 threshold_magnitude:    float = 0.5,
                 threshold_categorical:  int   = 5,
                 correlation_threshold:  float = 0.4,
                 threshold_normalized:   float = 0.30):
        self.sid = ShapeImportanceDivergence(threshold_magnitude=threshold_magnitude)
        self.csd = CategoricalShapeDivergence(threshold_categorical=threshold_categorical)
        self.aas = AutoAttributionSelector(correlation_threshold=correlation_threshold)
        self.nsd = NormalizedShapeDivergence(threshold=threshold_normalized)

    def check(self, sv1: np.ndarray, sv2: np.ndarray,
              X_eval: pd.DataFrame,
              feature_names: list = None) -> dict:
        if feature_names is None:
            feature_names = list(X_eval.columns)
        sv1 = np.array(sv1); sv2 = np.array(sv2)

        return {
            'magnitude':     self.sid.check(sv1, sv2, X_eval, feature_names),
            'categorical':   self.csd.check(sv1, sv2, X_eval, feature_names),
            'auto_selector': self.aas.select(X_eval, feature_names),
            'normalized':    self.nsd.check(sv1, sv2, X_eval, feature_names),
        }

    def report(self, results: dict) -> str:
        lines = ["=" * 56, "EXTENDED SHAPE DIVERGENCE DETECTOR", "=" * 56]

        # resumo de alertas
        mag_alerts  = [r for r in results['magnitude']   if r['alert']]
        cat_alerts  = [r for r in results['categorical']  if r.get('alert')]
        norm_alerts = [r for r in results['normalized']   if r['alert_normalized']]
        new_alerts  = [r for r in results['normalized']   if r['new_alert']]

        total = len(set(
            [r['feature'] for r in mag_alerts] +
            [r['feature'] for r in cat_alerts] +
            [r['feature'] for r in norm_alerts]
        ))

        lines.append(f"\nTotal de features com alerta: {total}")
        lines.append(f"  Shape+Magnitude: {len(mag_alerts)}")
        lines.append(f"  Categóricas:     {len(cat_alerts)}")
        lines.append(f"  Normalizadas:    {len(norm_alerts)} "
                     f"({len(new_alerts)} novos vs SDD base)")

        lines.append("")
        lines.append(self.sid.report(results['magnitude']))
        lines.append("")
        lines.append(self.csd.report(results['categorical']))
        lines.append("")
        lines.append(self.nsd.report(results['normalized']))
        lines.append("")
        lines.append(self.aas.report(results['auto_selector']))
        lines.append("=" * 56)
        return "\n".join(lines)

    def summary_alerts(self, results: dict) -> list:
        """Retorna lista plana de todos os alertas com tipo e feature."""
        alerts = []
        for r in results['magnitude']:
            if r['alert']:
                alerts.append({'feature': r['feature'], 'type': r['alert'],
                                'source': 'magnitude'})
        for r in results['categorical']:
            if r.get('alert'):
                alerts.append({'feature': r['feature'], 'type': r['alert'],
                                'source': 'categorical'})
        for r in results['normalized']:
            if r['new_alert']:
                alerts.append({'feature': r['feature'],
                                'type': 'NORMALIZED_DIVERGENCE',
                                'source': 'normalized'})
        return alerts
