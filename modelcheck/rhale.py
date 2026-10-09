"""
rhale.py — RHALE: Robust and Heterogeneity-aware ALE

Problema do ALE padrão:
  Efeito médio de X2 pode ser zero se X2 tem efeito +5 quando X3>0
  e efeito -5 quando X3<0. O ALE mostra 0 e ninguém vê o problema.

RHALE adiciona:
  1. Bins de tamanho variável ótimos (reduz viés/variância)
  2. Heterogeneidade por bin: std dos efeitos locais
  3. Alerta HETEROGENEOUS_EFFECT quando std > threshold

Uso:
    from modelcheck.rhale import RHALEDetector
    det = RHALEDetector()
    result = det.compare(model_v1, model_v2, X_eval, features)
    print(det.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from collections import Counter


def _optimal_bins(x: np.ndarray, max_bins: int = 10,
                  min_bin_n: int = 5) -> np.ndarray:
    """
    Bins de tamanho variável: começa com max_bins quantílicos,
    funde bins adjacentes onde amostras são insuficientes.
    """
    bins = np.unique(np.percentile(x, np.linspace(0, 100, max_bins + 1)))
    # Funde bins com poucas amostras
    final = [bins[0]]
    for b in bins[1:]:
        mask = (x >= final[-1]) & (x <= b)
        if mask.sum() >= min_bin_n:
            final.append(b)
        # senão: funde com o próximo (não adiciona o limite)
    if final[-1] < bins[-1]:
        final.append(bins[-1])
    return np.array(final)


def rhale_profile(model, X: pd.DataFrame, feature: str,
                  max_bins: int = 8, min_bin_n: int = 5) -> Optional[dict]:
    """
    Calcula perfil RHALE para uma feature.

    Retorna dict com:
        ale_mean   : efeito médio acumulado por bin
        ale_std    : heterogeneidade (std dos efeitos) por bin
        centers    : centros dos bins
        shape      : classificação da forma (sobre ale_mean)
        max_hetero : máxima heterogeneidade normalizada
        hetero_alert: True se efeitos heterogêneos detectados
    """
    from modelcheck.shape import _shape_profile_small_n

    x = np.asarray(X[feature].values, dtype=float)
    if np.std(x) < 1e-8:
        return None

    bins = _optimal_bins(x, max_bins=max_bins, min_bin_n=min_bin_n)
    if len(bins) < 4:
        return None

    ale_means, ale_stds, centers = [], [], []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x <= bins[i + 1])
        if mask.sum() < min_bin_n:
            continue
        Xlo = X[mask].copy(); Xhi = X[mask].copy()
        Xlo[feature] = bins[i]; Xhi[feature] = bins[i + 1]
        deltas = model.predict(Xhi.values) - model.predict(Xlo.values)
        ale_means.append(float(deltas.mean()))
        ale_stds.append(float(deltas.std()))
        centers.append((bins[i] + bins[i + 1]) / 2)

    if len(ale_means) < 3:
        return None

    ale_cum = np.cumsum(ale_means)
    ale_cum -= ale_cum.mean()

    # Classifica a forma
    p = _shape_profile_small_n(
        np.array(centers, dtype=float), ale_cum, min_n=3)
    shape = p['shape'] if p else 'n/a'

    # Heterogeneidade normalizada: std / (|mean| + epsilon)
    hetero_norm = [s / (abs(m) + 1e-6)
                   for m, s in zip(ale_means, ale_stds)]
    max_hetero = float(max(hetero_norm)) if hetero_norm else 0.0
    # Alerta: heterogeneidade alta E efeito médio pequeno
    # (efeitos opostos que se cancelam)
    hetero_alert = max_hetero > 2.0  # std > 2× |mean|

    return {
        'shape':        shape,
        'ale_cum':      ale_cum.tolist(),
        'ale_stds':     ale_stds,
        'centers':      centers,
        'max_hetero':   round(max_hetero, 3),
        'hetero_alert': hetero_alert,
        'n_bins':       len(ale_means),
    }


class RHALEDetector:
    """
    Detecta Shape Instability E Heterogeneidade de Efeito entre dois modelos.

    Dois tipos de alerta:
      SHAPE_DIV        — forma do efeito médio mudou (como ALEShapeDetector)
      HETEROGENEOUS_EFFECT — efeito médio esconde efeitos opostos em subgrupos

    Uso:
        det = RHALEDetector()
        result = det.compare(model_v1, model_v2, X_eval, features)
        print(det.report(result))
    """

    def __init__(self, max_bins: int = 8, min_bin_n: int = 5,
                 n_boot: int = 30, min_conf: float = 0.6,
                 hetero_threshold: float = 2.0):
        self.max_bins          = max_bins
        self.min_bin_n         = min_bin_n
        self.n_boot            = n_boot
        self.min_conf          = min_conf
        self.hetero_threshold  = hetero_threshold

    def _bootstrap_shape(self, model, X, feature):
        shapes = []
        for seed in range(self.n_boot):
            np.random.seed(seed)
            idx = np.random.choice(len(X), len(X), replace=True)
            Xb  = X.iloc[idx].reset_index(drop=True)
            p   = rhale_profile(model, Xb, feature,
                                self.max_bins, self.min_bin_n)
            if p:
                shapes.append(p['shape'])
        if not shapes:
            return 'n/a', 0.0
        sh, c = Counter(shapes).most_common(1)[0]
        return sh, round(c / len(shapes), 2)

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> list:
        if feature_names is None:
            feature_names = list(X_eval.columns)

        results = []
        for f in feature_names:
            # Shapes com bootstrap
            sh1, c1 = self._bootstrap_shape(model_v1, X_eval, f)
            sh2, c2 = self._bootstrap_shape(model_v2, X_eval, f)

            shape_div = (sh1 != sh2
                         and sh1 not in ('n/a', 'irregular')
                         and sh2 not in ('n/a', 'irregular')
                         and c1 >= self.min_conf
                         and c2 >= self.min_conf)

            # Heterogeneidade (nos dados completos)
            p1 = rhale_profile(model_v1, X_eval, f,
                               self.max_bins, self.min_bin_n)
            p2 = rhale_profile(model_v2, X_eval, f,
                               self.max_bins, self.min_bin_n)

            hetero1 = p1['max_hetero'] if p1 else 0.0
            hetero2 = p2['max_hetero'] if p2 else 0.0
            # FIX: heterogeneidade absoluta é propriedade dos DADOS.
            # O alerta relevante é a MUDANÇA de heterogeneidade entre v1 e v2.
            base = max(hetero1, hetero2, 1e-8)
            hetero_shift = abs(hetero1 - hetero2) / base
            hetero_alert = (hetero_shift > 0.5 and
                            max(hetero1, hetero2) > self.hetero_threshold)

            alerts = []
            if shape_div:
                alerts.append('SHAPE_DIV')
            if hetero_alert:
                alerts.append('HETEROGENEOUS_EFFECT')

            results.append({
                'feature':      f,
                'shape1':       sh1,  'conf1':   c1,
                'shape2':       sh2,  'conf2':   c2,
                'shape_div':    shape_div,
                'hetero_v1':    round(hetero1, 3),
                'hetero_v2':    round(hetero2, 3),
                'hetero_shift': round(hetero_shift, 3),
                'hetero_alert': hetero_alert,
                'alerts':       alerts,
            })
        return results

    def report(self, results: list) -> str:
        shape_divs  = [r for r in results if r['shape_div']]
        hetero_alerts = [r for r in results if r['hetero_alert']]
        n = len(results)

        lines = [f"RHALEDetector (n_boot={self.n_boot}):"]

        if not shape_divs and not hetero_alerts:
            lines.append(f"  ✅ Nenhuma divergência ({n} features)")
        else:
            if shape_divs:
                lines.append(f"  ⚠ SHAPE_DIV ({len(shape_divs)}/{n}):")
                for r in shape_divs:
                    lines.append(f"    '{r['feature']}': "
                                 f"{r['shape1']}→{r['shape2']} "
                                 f"(conf {r['conf1']:.0%}/{r['conf2']:.0%})")
            if hetero_alerts:
                lines.append(
                    f"  ⚠ HETEROGENEOUS_EFFECT ({len(hetero_alerts)}/{n}):")
                for r in hetero_alerts:
                    lines.append(
                        f"    '{r['feature']}': "
                        f"v1_hetero={r['hetero_v1']:.2f}  "
                        f"v2_hetero={r['hetero_v2']:.2f}  "
                        f"shift={r['hetero_shift']:.0%} "
                        f"(heterogeneidade mudou entre versões)")

        score = max(0, 100 - len(shape_divs)*15 - len(hetero_alerts)*10)
        lines.append(f"  Health Score: {score}/100")
        return "\n".join(lines)
