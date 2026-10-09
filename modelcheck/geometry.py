"""
geometry.py — GeometryDivergenceDetector

Detecta mudanças na estrutura geométrica global do espaço de representação
do modelo — invisíveis para o SDD que analisa feature por feature.

Ideia: o vetor SHAP de uma amostra é sua "assinatura comportamental".
A nuvem de assinaturas comportamentais de v1 vs v2 pode diferir geometricamente
mesmo que nenhuma feature individual mude de forma.

Métricas:
  - Distância de Fréchet entre distribuições de SHAP vectors
  - Correlação estrutural (Mantel test)
  - Eigenvalue drift da matriz de covariância SHAP
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist


def _shap_matrix(model, X: pd.DataFrame, n_sample: int = 200):
    """Retorna matriz de SHAP values (n_sample × n_features)."""
    import shap
    X_s = X.sample(min(n_sample, len(X)), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(X_s)
    if sv.ndim == 3: sv = sv[:,:,1]
    return sv, X_s


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Distância de Fréchet entre duas distribuições Gaussianas."""
    diff = mu1 - mu2
    # sqrt de produto de matrizes de covariância
    from scipy.linalg import sqrtm
    cov_prod = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(cov_prod):
        cov_prod = cov_prod.real
    fd = float(np.real(diff @ diff + np.trace(sigma1 + sigma2 - 2*cov_prod)))
    return max(0.0, fd)


def mantel_test(D1: np.ndarray, D2: np.ndarray,
                n_perm: int = 999) -> tuple:
    """
    Mantel test: correlação entre duas matrizes de distância.
    Retorna (rho, p_value).
    """
    n = D1.shape[0]
    idx = np.tril_indices(n, k=-1)
    v1 = D1[idx]; v2 = D2[idx]
    rho_obs, _ = scipy_stats.spearmanr(v1, v2)

    # Permutation test
    rhos = []
    for _ in range(n_perm):
        perm = np.random.permutation(n)
        D2p = D2[np.ix_(perm, perm)]
        rho_p, _ = scipy_stats.spearmanr(v1, D2p[idx])
        rhos.append(rho_p)
    p = (np.sum(np.array(rhos) >= rho_obs) + 1) / (n_perm + 1)
    return round(float(rho_obs), 4), round(float(p), 4)


class GeometryDivergenceDetector:
    """
    Detecta mudanças na geometria global do espaço comportamental.

    Complementa o SDD:
      - SDD: shape de cada feature individualmente
      - GDD: estrutura geométrica de TODAS as features juntas

    Casos que SDD não detecta mas GDD detecta:
      - Rotação do espaço de importâncias (features trocam de papel entre si)
      - Compressão/expansão da variância das importâncias
      - Mudança nas correlações entre importâncias de features

    Uso:
        gdd = GeometryDivergenceDetector()
        result = gdd.compare(model_v1, model_v2, X_eval, feature_names)
        print(gdd.report(result))
    """

    def __init__(self,
                 n_sample:       int   = 150,
                 frechet_thresh: float = 10.0,
                 eigen_thresh:   float = 0.30,
                 mantel_alpha:   float = 0.05):
        self.n_sample       = n_sample
        self.frechet_thresh = frechet_thresh
        self.eigen_thresh   = eigen_thresh
        self.mantel_alpha   = mantel_alpha

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        """
        Compara a geometria dos espaços SHAP de dois modelos.

        Retorna dict com:
            frechet_distance   : distância de Fréchet entre distribuições SHAP
            frechet_alert      : True se acima do threshold
            eigenvalue_drift   : mudança relativa nos eigenvalues da cov matrix
            eigen_alert        : True se drift acima do threshold
            mantel_rho         : correlação estrutural entre matrizes de distância
            mantel_p           : p-value do Mantel test
            mantel_alert       : True se correlação baixa (estrutura mudou)
            dominant_features  : features que mais contribuem para a divergência
            geometry_score     : 0-100, 100=geometria idêntica
        """
        sv1, _ = _shap_matrix(model_v1, X_eval, self.n_sample)
        sv2, _ = _shap_matrix(model_v2, X_eval, self.n_sample)

        if feature_names is None:
            feature_names = [f'f{i}' for i in range(sv1.shape[1])]

        # ── 1. Fréchet Distance ───────────────────────────────────
        mu1, mu2 = sv1.mean(0), sv2.mean(0)
        cov1 = np.cov(sv1.T) + np.eye(sv1.shape[1]) * 1e-6
        cov2 = np.cov(sv2.T) + np.eye(sv2.shape[1]) * 1e-6
        # BUG 16 (v1.43): mesma causa do bug 15 — Fréchet absoluta com
        # limiar absoluto. Medido: alerta extra em escala ×1000, inclusive
        # no controle.
        _esc = max(float(np.abs(np.concatenate([sv1, sv2])).mean()), 1e-12)
        fd = frechet_distance(mu1, cov1, mu2, cov2) / (_esc ** 2 * sv1.shape[1])
        frechet_alert = fd > self.frechet_thresh

        # ── 2. Eigenvalue Drift ───────────────────────────────────
        eig1 = np.sort(np.linalg.eigvalsh(cov1))[::-1]
        eig2 = np.sort(np.linalg.eigvalsh(cov2))[::-1]
        eig1_norm = eig1 / (eig1.sum() + 1e-8)
        eig2_norm = eig2 / (eig2.sum() + 1e-8)
        eigen_drift = float(np.abs(eig1_norm - eig2_norm).max())
        eigen_alert = eigen_drift > self.eigen_thresh

        # ── 3. Mantel Test (estrutura de distâncias) ──────────────
        n_man = min(60, len(sv1), len(sv2))
        D1 = cdist(sv1[:n_man], sv1[:n_man], metric='euclidean')
        D2 = cdist(sv2[:n_man], sv2[:n_man], metric='euclidean')
        mantel_rho, mantel_p = mantel_test(D1, D2, n_perm=499)
        mantel_alert = mantel_p > self.mantel_alpha  # estrutura divergiu

        # ── 4. Features dominantes na divergência ─────────────────
        mean_diff = np.abs(mu1 - mu2)
        std_diff  = np.abs(sv1.std(0) - sv2.std(0))
        importance = mean_diff + std_diff
        top_idx = np.argsort(importance)[::-1][:3]
        dominant_features = [feature_names[i] for i in top_idx
                             if i < len(feature_names)]

        # ── 5. Geometry Score ─────────────────────────────────────
        alerts = sum([frechet_alert, eigen_alert, mantel_alert])
        geometry_score = max(0, 100 - alerts * 25
                             - min(fd / self.frechet_thresh, 2) * 10
                             - eigen_drift * 20)
        geometry_score = round(float(geometry_score), 1)

        return {
            'frechet_distance':  round(float(fd), 4),
            'frechet_alert':     frechet_alert,
            'eigenvalue_drift':  round(eigen_drift, 4),
            'eigen_alert':       eigen_alert,
            'mantel_rho':        mantel_rho,
            'mantel_p':          mantel_p,
            'mantel_alert':      mantel_alert,
            'dominant_features': dominant_features,
            'geometry_score':    geometry_score,
            'n_alerts':          alerts,
        }

    def report(self, result: dict) -> str:
        lines = [
            "GeometryDivergenceDetector:",
            f"  Geometry Score: {result['geometry_score']}/100",
            f"  Fréchet distance:   {result['frechet_distance']:.2f}"
            f"  {'⚠ GEOMETRY_SHIFT' if result['frechet_alert'] else '✅ ok'}",
            f"  Eigenvalue drift:   {result['eigenvalue_drift']:.3f}"
            f"  {'⚠ EIGEN_DRIFT' if result['eigen_alert'] else '✅ ok'}",
            f"  Mantel ρ={result['mantel_rho']:.3f}  p={result['mantel_p']:.3f}"
            f"  {'⚠ STRUCTURE_CHANGED' if result['mantel_alert'] else '✅ stable'}",
        ]
        if result['n_alerts'] > 0:
            lines.append(f"  Dominant features in divergence: "
                        f"{result['dominant_features']}")
        return "\n".join(lines)
