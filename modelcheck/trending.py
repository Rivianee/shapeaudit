"""
trending.py — ShapeTrendingDetector

Detecta deriva gradual acumulada ao longo de N retreinamentos.
Cada passo individual pode ser pequeno, mas a tendência é significativa.

Algoritmo:
  1. Para cada versão t, calcula perfil SHAP de cada feature
  2. Calcula distância de forma entre versões consecutivas
  3. Aplica Mann-Kendall test para detectar tendência crescente
  4. Alerta SHAPE_DRIFT_ACCUMULATING se p < alpha

Uso:
    from modelcheck.trending import ShapeTrendingDetector
    detector = ShapeTrendingDetector()
    detector.add_snapshot(model_v1, X_eval, label="v1")
    detector.add_snapshot(model_v2, X_eval, label="v2")
    ...
    result = detector.analyze()
    print(detector.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats


def _mann_kendall(x: np.ndarray) -> tuple:
    """Mann-Kendall trend test. Retorna (tau, p_value, trend)."""
    n = len(x)
    if n < 4:
        return 0.0, 1.0, 'no trend'
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(x[j] - x[i])
    # Variância de S
    var_s = n * (n - 1) * (2 * n + 5) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
    tau = s / (n * (n - 1) / 2)
    trend = ('increasing' if s > 0 and p < 0.05 else
             'decreasing' if s < 0 and p < 0.05 else 'no trend')
    return round(float(tau), 4), round(float(p), 4), trend


def _shape_distance(sh1: str, sh2: str) -> float:
    """Distância qualitativa entre duas formas."""
    if sh1 == 'n/a' or sh2 == 'n/a':
        return 0.0
    if sh1 == sh2:
        return 0.0
    # Formas opostas têm distância maior
    opposites = {('crescente','decrescente'), ('u','u_invertido')}
    if (sh1, sh2) in opposites or (sh2, sh1) in opposites:
        return 1.0
    return 0.5  # formas diferentes mas não opostas


class ShapeTrendingDetector:
    """
    Monitora deriva gradual de shape instability ao longo do tempo.

    Uso:
        det = ShapeTrendingDetector()
        for i, (model, X) in enumerate(versions):
            det.add_snapshot(model, X, label=f"v{i+1}")
        result = det.analyze()
        print(det.report(result))
    """

    def __init__(self, alpha: float = 0.05, min_versions: int = 4):
        self.alpha        = alpha
        self.min_versions = min_versions
        self.snapshots    = []  # list of {'label', 'shapes', 'shap_means'}

    def add_snapshot(self, model, X_eval: pd.DataFrame,
                     label: str = None,
                     feature_names: Optional[List[str]] = None):
        """Adiciona snapshot de um modelo."""
        import shap
        from modelcheck.shape import _shape_profile

        if feature_names is None:
            feature_names = list(X_eval.columns)
        if label is None:
            label = f"v{len(self.snapshots)+1}"

        # Calcula SHAP e shapes
        sample = X_eval.sample(min(150, len(X_eval)), random_state=42)
        try:
            sv = shap.TreeExplainer(model).shap_values(sample)
            if sv.ndim == 3: sv = sv[:,:,1]
        except Exception:
            # Fallback: ALE shape
            from modelcheck.rhale import rhale_profile
            shapes = {}
            shap_means = {}
            for f in feature_names:
                p = rhale_profile(model, X_eval, f)
                shapes[f]     = p['shape'] if p else 'n/a'
                shap_means[f] = 0.0
            self.snapshots.append({
                'label': label, 'shapes': shapes,
                'shap_means': shap_means
            })
            return

        shapes = {}
        shap_means = {}
        for i, f in enumerate(feature_names):
            p = _shape_profile(sample[f].values, sv[:, i])
            shapes[f]     = p['shape'] if p else 'n/a'
            shap_means[f] = round(float(np.abs(sv[:, i]).mean()), 4)

        self.snapshots.append({
            'label': label, 'shapes': shapes,
            'shap_means': shap_means
        })

    def analyze(self) -> dict:
        """Analisa tendências ao longo dos snapshots."""
        n = len(self.snapshots)
        if n < 2:
            return {'error': 'Precisa de ao menos 2 snapshots'}

        if not self.snapshots[0]['shapes']:
            return {'error': 'Snapshots sem shapes'}

        feature_names = list(self.snapshots[0]['shapes'].keys())
        results = {}

        for f in feature_names:
            # Série de formas ao longo do tempo
            shapes_series = [s['shapes'].get(f, 'n/a') for s in self.snapshots]
            means_series  = [s['shap_means'].get(f, 0.0) for s in self.snapshots]

            # Distâncias consecutivas
            distances = [_shape_distance(shapes_series[i], shapes_series[i+1])
                         for i in range(len(shapes_series)-1)]

            # Distância acumulada
            cumulative = float(np.cumsum(distances)[-1]) if distances else 0.0

            # Mann-Kendall na importância SHAP (detecta deriva de magnitude)
            tau, p_mk, trend_mk = _mann_kendall(np.array(means_series))

            # Detecta se houve mudanças de forma
            n_shape_changes = sum(1 for d in distances if d > 0)
            drift_alert = (n_shape_changes >= 2 or
                           (trend_mk in ('increasing','decreasing')
                            and p_mk < self.alpha
                            and n >= self.min_versions))

            results[f] = {
                'shapes_series':   shapes_series,
                'means_series':    means_series,
                'distances':       distances,
                'cumulative_dist': round(cumulative, 3),
                'n_shape_changes': n_shape_changes,
                'mk_tau':          tau,
                'mk_p':            p_mk,
                'mk_trend':        trend_mk,
                'drift_alert':     drift_alert,
                'alert_type':      ('SHAPE_DRIFT_ACCUMULATING'
                                    if drift_alert else None),
            }

        drifting = [f for f, r in results.items() if r['drift_alert']]
        return {
            'n_snapshots':  n,
            'labels':       [s['label'] for s in self.snapshots],
            'feature_results': results,
            'drifting_features': drifting,
            'overall_alert': len(drifting) > 0,
        }

    def report(self, result: dict) -> str:
        if 'error' in result:
            return f"ShapeTrendingDetector: {result['error']}"
        labels = result['labels']
        drifting = result['drifting_features']
        lines = [
            f"ShapeTrendingDetector ({result['n_snapshots']} versões: "
            f"{' → '.join(labels)}):",
        ]
        if not drifting:
            lines.append("  ✅ Nenhuma deriva acumulada detectada.")
        else:
            lines.append(f"  ⚠ SHAPE_DRIFT_ACCUMULATING em "
                         f"{len(drifting)} feature(s):")
            for f in drifting:
                r = result['feature_results'][f]
                shapes = ' → '.join(r['shapes_series'])
                lines.append(
                    f"    '{f}': {shapes}\n"
                    f"           dist_acumulada={r['cumulative_dist']}  "
                    f"n_mudanças={r['n_shape_changes']}  "
                    f"MK_p={r['mk_p']:.3f} ({r['mk_trend']})")
        return "\n".join(lines)
