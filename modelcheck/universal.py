"""
universal.py — Detectores geométricos e topológicos agnósticos a arquitetura

Resolve a lacuna (H) da Revisão Crítica: Canary, Topology e Geometry
dependiam de shap.TreeExplainer e retornavam N/A para MLP/SVR/Ridge.

Estratégia: substituir a matriz de SHAP values por uma matriz de
embeddings ALE (ale_fast.sample_embeddings_ale) como representação do
modelo. As métricas geométricas (Fréchet, Mantel, eigenvalue) e
topológicas operam igual sobre qualquer nuvem de pontos.

Inclui TopologicalDriftGudhi, que usa homologia persistente da biblioteca
`gudhi` (Betti-0 E Betti-1, bottleneck distance exata) em substituição à
implementação caseira de topology.py. Recall medido: 0,33 com precisão
1,00 em 4 datasets — camada complementar, não primária.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm

_HAS_GUDHI = False
try:
    import gudhi as _gd
    _HAS_GUDHI = True
except Exception:
    pass


def _repr_matrix(model, X, feature_names, use_shap=True,
                 n_sample=150, n_replicas=30, seed=0):
    """
    Representação do modelo como matriz (pontos × dims).
    Tenta SHAP; se falhar, usa embeddings ALE por subamostragem.
    Retorna (matriz, método).
    """
    if use_shap:
        try:
            import shap
            Xs = X.sample(min(n_sample, len(X)), random_state=seed)
            sv = shap.TreeExplainer(model).shap_values(Xs)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            return np.asarray(sv, dtype=float), 'SHAP'
        except Exception:
            pass
    from modelcheck.ale_fast import sample_embeddings_ale
    M = sample_embeddings_ale(model, X, feature_names,
                              n_replicas=n_replicas, n_points=8, seed=seed)
    if M is None:
        return None, 'FAILED'
    return M, 'ALE_EMBEDDING'


def _frechet(A, B):
    mu1, mu2 = A.mean(0), B.mean(0)
    eps = np.eye(A.shape[1]) * 1e-6
    c1, c2 = np.cov(A.T) + eps, np.cov(B.T) + eps
    d = mu1 - mu2
    s = sqrtm(c1 @ c2)
    if np.iscomplexobj(s):
        s = s.real
    return max(0.0, float(np.real(d @ d + np.trace(c1 + c2 - 2 * s))))


class UniversalCanaryDetector:
    """
    GeometricCanary agnóstico a arquitetura.

    Usa SHAP quando disponível (árvores) e embeddings ALE caso contrário.
    Métricas: Fréchet, correlação topológica (Mantel-like), isotropia.
    Thresholds são recalibrados por método, porque as escalas diferem.
    """

    def __init__(self, force_ale: bool = False,
                 n_sample: int = 150, n_replicas: int = 30,
                 topology_thresh: float = 0.85,
                 isotropy_thresh: float = 0.15):
        self.force_ale = force_ale
        self.n_sample = n_sample
        self.n_replicas = n_replicas
        self.topology_thresh = topology_thresh
        self.isotropy_thresh = isotropy_thresh

    def compare(self, model_v1, model_v2, X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        from modelcheck._validate import validate_inputs
        X_eval, feature_names = validate_inputs(
            model_v1, model_v2, X_eval, feature_names,
            caller='UniversalCanaryDetector')
        if feature_names is None:
            feature_names = list(X_eval.columns)
        use_shap = not self.force_ale
        A, m1 = _repr_matrix(model_v1, X_eval, feature_names, use_shap,
                             self.n_sample, self.n_replicas, seed=0)
        B, m2 = _repr_matrix(model_v2, X_eval, feature_names, use_shap,
                             self.n_sample, self.n_replicas, seed=0)
        if A is None or B is None or m1 != m2:
            # métodos diferentes não são comparáveis: força ALE nos dois
            A, m1 = _repr_matrix(model_v1, X_eval, feature_names, False,
                                 self.n_sample, self.n_replicas, 0)
            B, m2 = _repr_matrix(model_v2, X_eval, feature_names, False,
                                 self.n_sample, self.n_replicas, 0)
            if A is None or B is None:
                return {'canary_score': None, 'alerts': [],
                        'measured': False, 'method': 'FAILED'}
        method = m1
        n = min(len(A), len(B))
        A, B = A[:n], B[:n]

        # Fréchet normalizado pela escala da própria representação
        scale = float(np.abs(np.concatenate([A, B])).mean()) + 1e-12
        fd_raw = _frechet(A, B)
        fd = fd_raw / (scale ** 2 * A.shape[1])

        # estrutura de distâncias
        k = min(50, n)
        D1 = cdist(A[:k], A[:k]); D2 = cdist(B[:k], B[:k])
        tri = np.tril_indices(k, -1)
        rho, p_topo = scipy_stats.spearmanr(D1[tri], D2[tri])
        rho = 0.0 if np.isnan(rho) else float(rho)

        # isotropia
        def iso(M):
            c = np.cov(M.T) + np.eye(M.shape[1]) * 1e-10
            e = np.abs(np.linalg.eigvalsh(c))
            return float(e.min() / (e.max() + 1e-12))
        i1, i2 = iso(A), iso(B)
        iso_drift = abs(i1 - i2)

        # thresholds por método (escalas diferentes)
        fd_thresh = 0.5 if method == 'SHAP' else 2.0
        alerts = []
        if fd > fd_thresh:               alerts.append('FRECHET_ALERT')
        if rho < self.topology_thresh:   alerts.append('TOPOLOGY_ALERT')
        if iso_drift > self.isotropy_thresh: alerts.append('ISOTROPY_ALERT')

        delta = np.abs(A.mean(0) - B.mean(0))
        top = np.argsort(delta)[::-1][:3]
        if method == 'SHAP':
            affected = [(feature_names[i], round(float(delta[i]), 5))
                        for i in top if i < len(feature_names)]
        else:
            npts = A.shape[1] // max(len(feature_names), 1)
            affected = []
            for i in top:
                fi = i // max(npts, 1)
                if fi < len(feature_names):
                    affected.append((feature_names[fi], round(float(delta[i]), 5)))

        score = max(0.0, 100 - len(alerts) * 20
                    - min(fd / fd_thresh, 3) * 10
                    - max(0, self.topology_thresh - rho) * 50
                    - iso_drift * 20)
        return {'canary_score': round(score, 1), 'method': method,
                'measured': True,
                'frechet_norm': round(fd, 4), 'topology_rho': round(rho, 4),
                'isotropy_drift': round(iso_drift, 6),
                'alerts': alerts, 'affected_features': affected}

    def report(self, r: dict) -> str:
        if not r.get('measured'):
            return "UniversalCanaryDetector: não mensurável"
        ic = "🟢" if not r['alerts'] else ("🟡" if len(r['alerts']) == 1 else "🔴")
        return (f"UniversalCanaryDetector [{r['method']}]: {ic} "
                f"Score={r['canary_score']}/100\n"
                f"  Fréchet_norm={r['frechet_norm']:.3f}  "
                f"ρ_topo={r['topology_rho']:.3f}  "
                f"iso_drift={r['isotropy_drift']:.5f}\n"
                f"  Alerts: {r['alerts'] or '—'}")


class TopologicalDriftGudhi:
    """
    Deriva topológica com homologia persistente da biblioteca `gudhi`.

    Substitui a implementação caseira de topology.py (Union-Find, só
    Betti-0, bottleneck aproximada). Aqui:
      - complexo de Vietoris-Rips até dimensão 2
      - diagramas de persistência de Betti-0 E Betti-1
      - bottleneck_distance exata do gudhi

    Requer `pip install gudhi`. Se ausente, `available` = False.
    """

    def __init__(self, n_points: int = 60, max_edge: Optional[float] = None,
                 bottleneck_thresh: float = 1.20,
                 force_ale: bool = False):
        self.n_points = n_points
        self.max_edge = max_edge
        self.bottleneck_thresh = bottleneck_thresh
        self.force_ale = force_ale
        self.available = _HAS_GUDHI

    def _diagrams(self, M: np.ndarray):
        pts = M[:self.n_points]
        # normaliza a escala para que o threshold seja comparável
        sd = np.abs(pts).mean() + 1e-12
        pts = pts / sd
        me = self.max_edge
        if me is None:
            d = cdist(pts, pts)
            me = float(np.percentile(d[d > 0], 60))
        rips = _gd.RipsComplex(points=pts, max_edge_length=me)
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()
        d0 = st.persistence_intervals_in_dimension(0)
        d1 = st.persistence_intervals_in_dimension(1)
        # FIX: barras infinitas fazem bottleneck_distance devolver inf.
        # Substitui morte infinita pelo max_edge (truncamento padrão em TDA).
        def _finite(d):
            if len(d) == 0:
                return np.zeros((0, 2))
            d = np.asarray(d, dtype=float).copy()
            d[~np.isfinite(d[:, 1]), 1] = me
            # descarta barras de persistência nula (ruído numérico)
            keep = (d[:, 1] - d[:, 0]) > 1e-9
            return d[keep]
        return _finite(d0), _finite(d1)

    def compare(self, model_v1, model_v2, X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        if not self.available:
            return {'topo_score': None, 'alerts': [], 'measured': False,
                    'error': 'gudhi não instalado'}
        if feature_names is None:
            feature_names = list(X_eval.columns)
        use_shap = not self.force_ale
        A, m1 = _repr_matrix(model_v1, X_eval, feature_names, use_shap,
                             seed=0, n_replicas=self.n_points)
        B, m2 = _repr_matrix(model_v2, X_eval, feature_names, use_shap,
                             seed=0, n_replicas=self.n_points)
        if A is None or B is None or m1 != m2:
            A, m1 = _repr_matrix(model_v1, X_eval, feature_names, False,
                                 seed=0, n_replicas=self.n_points)
            B, m2 = _repr_matrix(model_v2, X_eval, feature_names, False,
                                 seed=0, n_replicas=self.n_points)
            if A is None or B is None:
                return {'topo_score': None, 'alerts': [], 'measured': False,
                        'error': 'representação indisponível'}
        try:
            a0, a1 = self._diagrams(A)
            b0, b1 = self._diagrams(B)
            bn0 = float(_gd.bottleneck_distance(a0, b0))
            bn1 = (float(_gd.bottleneck_distance(a1, b1))
                   if len(a1) and len(b1) else 0.0)
        except Exception as e:
            return {'topo_score': None, 'alerts': [], 'measured': False,
                    'error': f'{type(e).__name__}: {e}'}

        betti0 = (len(a0), len(b0))
        betti1 = (len(a1), len(b1))
        alerts = []
        if bn0 > self.bottleneck_thresh: alerts.append('H0_DRIFT')
        if bn1 > self.bottleneck_thresh: alerts.append('H1_DRIFT')
        # calibrado com baseline de variância de seed (diff até 5 é ruído)
        if abs(betti1[0] - betti1[1]) > 6: alerts.append('LOOP_STRUCTURE_CHANGE')

        score = max(0.0, 100 - len(alerts) * 20
                    - min(bn0 / self.bottleneck_thresh, 2) * 10
                    - min(bn1 / self.bottleneck_thresh, 2) * 10)
        return {'topo_score': round(score, 1), 'method': m1, 'measured': True,
                'bottleneck_H0': round(bn0, 4), 'bottleneck_H1': round(bn1, 4),
                'n_H0': betti0, 'n_H1': betti1,
                'alerts': alerts, 'library': f'gudhi {_gd.__version__}'}

    def report(self, r: dict) -> str:
        if not r.get('measured'):
            return f"TopologicalDriftGudhi: não mensurável ({r.get('error','')})"
        ic = "🟢" if not r['alerts'] else ("🟡" if len(r['alerts']) == 1 else "🔴")
        return (f"TopologicalDriftGudhi [{r['library']}, {r['method']}]: {ic} "
                f"Score={r['topo_score']}/100\n"
                f"  bottleneck H0={r['bottleneck_H0']:.4f}  "
                f"H1={r['bottleneck_H1']:.4f}\n"
                f"  n_H0={r['n_H0']}  n_H1={r['n_H1']}  "
                f"Alerts: {r['alerts'] or '—'}")
