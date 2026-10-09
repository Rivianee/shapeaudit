"""
topology.py — TopologicalDriftDetector

Detecta mudanças na estrutura topológica do espaço de representação
usando homologia persistente simplificada (Betti numbers via VR complex).

Intuição: a "forma" global do espaço SHAP tem componentes conectados
(Betti-0) e loops (Betti-1). Quando um modelo deriva, essa estrutura
topológica muda — clusters se fundem, loops aparecem ou somem.

Implementação sem gudhi/ripser (dependências pesadas):
  - Usa filtração de Vietoris-Rips via matriz de distâncias
  - Calcula Betti-0 (componentes conectados) por threshold
  - Calcula persistence diagram simplificado
  - Compara diagramas entre v1 e v2 via bottleneck distance

Uso:
    from modelcheck.topology import TopologicalDriftDetector
    tdd = TopologicalDriftDetector()
    result = tdd.compare(model_v1, model_v2, X_eval, features)
    print(tdd.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy.spatial.distance import cdist
from scipy import stats as scipy_stats


def _shap_sample(model, X: pd.DataFrame,
                 feature_names: List[str],
                 n: int = 100) -> np.ndarray:
    import shap
    Xs = X.sample(min(n, len(X)), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if sv.ndim == 3: sv = sv[:,:,1]
    return sv.astype(float)


def _betti0_curve(D: np.ndarray,
                  n_thresh: int = 20) -> tuple:
    """
    Calcula curva Betti-0 (número de componentes conectados) por threshold.
    
    Para cada epsilon, conta componentes no grafo onde dist(i,j) <= epsilon.
    Usa Union-Find eficiente.
    """
    n = D.shape[0]
    thresholds = np.linspace(0, D.max(), n_thresh + 1)
    betti0 = []

    for eps in thresholds:
        # Union-Find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i+1, n):
                if D[i,j] <= eps:
                    union(i, j)

        components = len({find(i) for i in range(n)})
        betti0.append(components)

    return thresholds, np.array(betti0)


def _persistence_diagram(betti0_curve: np.ndarray,
                          thresholds: np.ndarray) -> list:
    """
    Extrai pares de birth/death do diagrama de persistência.
    Um componente "nasce" quando epsilon = 0 e "morre" quando se funde.
    """
    pairs = []
    prev = betti0_curve[0]
    for i in range(1, len(betti0_curve)):
        curr = betti0_curve[i]
        if curr < prev:  # componente(s) morreram
            for _ in range(prev - curr):
                pairs.append((0.0, float(thresholds[i])))
        prev = curr
    # Componente que nunca morre (componente principal)
    pairs.append((0.0, float('inf')))
    return pairs


def _bottleneck_distance(diag1: list, diag2: list) -> float:
    """
    Bottleneck distance entre dois diagramas de persistência.
    Simplificado: usa apenas os pares finitos (exclui infinitos).
    """
    finite1 = [(b, d) for b, d in diag1 if d != float('inf')]
    finite2 = [(b, d) for b, d in diag2 if d != float('inf')]

    if not finite1 and not finite2:
        return 0.0

    # Persistence = d - b
    pers1 = sorted([d - b for b, d in finite1], reverse=True)
    pers2 = sorted([d - b for b, d in finite2], reverse=True)

    # Padeia com zeros
    n = max(len(pers1), len(pers2))
    pers1 += [0.0] * (n - len(pers1))
    pers2 += [0.0] * (n - len(pers2))

    return float(max(abs(p1 - p2) for p1, p2 in zip(pers1, pers2)))


def _wasserstein_pd(diag1: list, diag2: list, p: int = 1) -> float:
    """Wasserstein distance entre diagramas de persistência."""
    finite1 = sorted([d - b for b, d in diag1 if d != float('inf')])
    finite2 = sorted([d - b for b, d in diag2 if d != float('inf')])
    n = max(len(finite1), len(finite2), 1)
    finite1 += [0.0] * (n - len(finite1))
    finite2 += [0.0] * (n - len(finite2))
    return float(np.mean(np.abs(np.array(finite1) - np.array(finite2))))


class TopologicalDriftDetector:
    """
    Detecta deriva topológica no espaço de representação SHAP.

    Métricas:
      betti0_drift      : mudança no número de componentes conectados
      bottleneck_dist   : distância bottleneck entre diagramas de persistência
      wasserstein_dist  : distância Wasserstein entre diagramas
      persistence_entropy: entropia dos diagramas (medida de complexidade)

    Threshold defaults:
      bottleneck > 0.3   → TOPOLOGICAL_DRIFT
      betti0_diff > 2    → CONNECTIVITY_CHANGE
    """

    def __init__(self, n_sample: int = 80, n_thresh: int = 15,
                 bottleneck_thresh: float = 0.3,
                 betti0_thresh: int = 2):
        self.n_sample          = n_sample
        self.n_thresh          = n_thresh
        self.bottleneck_thresh = bottleneck_thresh
        self.betti0_thresh     = betti0_thresh

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:

        if feature_names is None:
            feature_names = list(X_eval.columns)

        sv1 = _shap_sample(model_v1, X_eval, feature_names, self.n_sample)
        sv2 = _shap_sample(model_v2, X_eval, feature_names, self.n_sample)

        # Matrizes de distância no espaço SHAP
        D1 = cdist(sv1, sv1, 'euclidean')
        D2 = cdist(sv2, sv2, 'euclidean')

        # Curvas Betti-0
        th1, b01 = _betti0_curve(D1, self.n_thresh)
        th2, b02 = _betti0_curve(D2, self.n_thresh)

        # Betti-0 inicial (em epsilon=0) = n de pontos isolados
        betti0_v1_init = int(b01[0])
        betti0_v2_init = int(b02[0])
        # Betti-0 no meio do range
        mid = self.n_thresh // 2
        betti0_v1_mid = int(b01[mid])
        betti0_v2_mid = int(b02[mid])
        betti0_diff = abs(betti0_v1_mid - betti0_v2_mid)

        # Diagramas de persistência
        pd1 = _persistence_diagram(b01, th1)
        pd2 = _persistence_diagram(b02, th2)

        bn_dist   = _bottleneck_distance(pd1, pd2)
        wass_dist = _wasserstein_pd(pd1, pd2)

        # Entropia de persistência
        def _entropy(diag):
            pers = [d - b for b, d in diag if d != float('inf')]
            if not pers: return 0.0
            p = np.array(pers); p = p / (p.sum() + 1e-8)
            return float(-np.sum(p * np.log(p + 1e-10)))

        ent1 = _entropy(pd1); ent2 = _entropy(pd2)
        entropy_drift = abs(ent1 - ent2)

        # Correlação entre curvas Betti-0 (estrutura similar = alta correlação)
        b0_corr, b0_p = scipy_stats.spearmanr(b01, b02)

        # Alertas
        topo_alert = bn_dist > self.bottleneck_thresh
        conn_alert = betti0_diff > self.betti0_thresh
        alerts = []
        if topo_alert: alerts.append('TOPOLOGICAL_DRIFT')
        if conn_alert: alerts.append('CONNECTIVITY_CHANGE')

        topo_score = max(0.0,
            100
            - len(alerts) * 20
            - min(bn_dist / self.bottleneck_thresh, 2) * 15
            - betti0_diff * 5
            - entropy_drift * 10)

        return {
            'betti0_v1':        betti0_v1_mid,
            'betti0_v2':        betti0_v2_mid,
            'betti0_diff':      betti0_diff,
            'bottleneck_dist':  round(bn_dist, 4),
            'wasserstein_dist': round(wass_dist, 4),
            'entropy_v1':       round(ent1, 4),
            'entropy_v2':       round(ent2, 4),
            'entropy_drift':    round(entropy_drift, 4),
            'betti0_corr':      round(float(b0_corr), 4),
            'topo_alert':       topo_alert,
            'conn_alert':       conn_alert,
            'alerts':           alerts,
            'topo_score':       round(topo_score, 1),
        }

    def report(self, result: dict) -> str:
        alerts = result['alerts']
        icon = "🟢" if not alerts else ("🟡" if len(alerts)==1 else "🔴")
        lines = [
            f"TopologicalDriftDetector: {icon}  Score={result['topo_score']}/100",
            f"  Betti-0: v1={result['betti0_v1']}  v2={result['betti0_v2']}"
            f"  diff={result['betti0_diff']}"
            f"  {'⚠ CONNECTIVITY_CHANGE' if result['conn_alert'] else '✅'}",
            f"  Bottleneck dist: {result['bottleneck_dist']:.4f}"
            f"  {'⚠ TOPOLOGICAL_DRIFT' if result['topo_alert'] else '✅'}",
            f"  Wasserstein dist: {result['wasserstein_dist']:.4f}",
            f"  Entropy: v1={result['entropy_v1']:.4f}  v2={result['entropy_v2']:.4f}"
            f"  drift={result['entropy_drift']:.4f}",
            f"  Betti-0 correlation: {result['betti0_corr']:.4f}",
        ]
        if alerts:
            lines.append(f"  Alerts: {', '.join(alerts)}")
        return "\n".join(lines)
