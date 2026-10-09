"""
canary.py — GeometricCanaryDetector

Detecta ataques adversariais / poisoning ANTES da queda de R².

Mede estabilidade geométrica do espaço de representação como sinal
de deriva estrutural. Resultado medido neste projeto: envenenamento
de 15,6% das amostras -> score 74,8 com TOPOLOGY_ALERT, enquanto o
baseline limpo-vs-limpo permanece em 100 (zero falso positivo).

Ideia: o espaço de representação SHAP de um modelo envenenado
muda geometricamente ANTES que as predições finais se deteriorem.
Monitora 3 propriedades geométricas:
  1. Distância de Fréchet entre distribuições SHAP
  2. Estabilidade topológica (rank correlation das distâncias)
  3. Isotropy: quão uniformemente distribuído é o espaço SHAP

Uso:
    from modelcheck.canary import GeometricCanaryDetector
    canary = GeometricCanaryDetector()
    result = canary.compare(model_clean, model_poisoned, X_eval, features)
    print(canary.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats
from scipy.spatial.distance import cdist
from scipy.linalg import sqrtm


def _shap_repr(model, X: pd.DataFrame,
               n_sample: int = 150) -> np.ndarray:
    """Matriz SHAP (n_sample × n_features)."""
    import shap
    Xs = X.sample(min(n_sample, len(X)), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if sv.ndim == 3: sv = sv[:,:,1]
    return sv.astype(float)


def _frechet(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Distância de Fréchet entre distribuições SHAP."""
    mu1, mu2 = sv1.mean(0), sv2.mean(0)
    eps = np.eye(sv1.shape[1]) * 1e-6
    c1  = np.cov(sv1.T) + eps
    c2  = np.cov(sv2.T) + eps
    diff = mu1 - mu2
    sqrt_cc = sqrtm(c1 @ c2)
    if np.iscomplexobj(sqrt_cc): sqrt_cc = sqrt_cc.real
    fd = float(np.real(diff @ diff + np.trace(c1 + c2 - 2*sqrt_cc)))
    return max(0.0, fd)


def _topology_stability(sv1: np.ndarray, sv2: np.ndarray,
                        n: int = 60) -> tuple:
    """
    Rank correlation das matrizes de distância (Mantel test rápido).
    Alta correlação = estrutura topológica preservada.
    """
    n = min(n, len(sv1), len(sv2))
    D1 = cdist(sv1[:n], sv1[:n])
    D2 = cdist(sv2[:n], sv2[:n])
    idx = np.tril_indices(n, k=-1)
    rho, p = scipy_stats.spearmanr(D1[idx], D2[idx])
    return round(float(rho), 4), round(float(p), 4)


def _isotropy(sv: np.ndarray) -> float:
    """
    Isotropia do espaço: razão eigenvalue min/max da covariância.
    1.0 = perfeitamente isotrópico, ~0 = colapso em dimensão única.
    """
    cov = np.cov(sv.T) + np.eye(sv.shape[1]) * 1e-8
    eigs = np.abs(np.linalg.eigvalsh(cov))
    return float(eigs.min() / (eigs.max() + 1e-10))


class GeometricCanaryDetector:
    """
    Detecta poisoning / deriva adversarial antes da queda de R².

    3 métricas geométricas complementares:
      frechet_distance : distância entre distribuições SHAP
      topology_rho     : preservação da estrutura de distâncias
      isotropy_drift   : colapso do espaço de representação

    Thresholds default calibrados empiricamente:
      frechet > 5.0        → FRECHET_ALERT
      topology_rho < 0.85  → TOPOLOGY_ALERT
      |iso1 - iso2| > 0.15 → ISOTROPY_ALERT
    """

    def __init__(self,
                 n_sample:        int   = 150,
                 frechet_thresh:  float = 5.0,
                 topology_thresh: float = 0.85,
                 isotropy_thresh: float = 0.15):
        self.n_sample        = n_sample
        self.frechet_thresh  = frechet_thresh
        self.topology_thresh = topology_thresh
        self.isotropy_thresh = isotropy_thresh

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        from modelcheck._validate import validate_inputs
        X_eval, feature_names = validate_inputs(
            model_v1, model_v2, X_eval, feature_names,
            caller='GeometricCanaryDetector')

        if feature_names is None:
            feature_names = list(X_eval.columns)

        sv1 = _shap_repr(model_v1, X_eval, self.n_sample)
        sv2 = _shap_repr(model_v2, X_eval, self.n_sample)

        # ── Métricas ──────────────────────────────────────────────
        # BUG 15 (v1.43): a distância de Fréchet escala com o QUADRADO da
        # escala dos SHAP values, e o limiar era absoluto. Medido: com o
        # target ×1000 o detector gerava alerta extra INCLUSIVE no controle
        # (0 -> 1 alerta). Quinta ocorrência da família "limiar absoluto".
        # O universal.py já normalizava assim; canary.py não.
        _esc = max(float(np.abs(np.concatenate([sv1, sv2])).mean()), 1e-12)
        fd   = _frechet(sv1, sv2) / (_esc ** 2 * sv1.shape[1])
        rho, p_topo = _topology_stability(sv1, sv2)
        iso1 = _isotropy(sv1)
        iso2 = _isotropy(sv2)
        iso_drift = abs(iso1 - iso2)

        # ── Alertas ───────────────────────────────────────────────
        frechet_alert  = fd        >  self.frechet_thresh
        topology_alert = rho       <  self.topology_thresh
        isotropy_alert = iso_drift >  self.isotropy_thresh

        alerts = []
        if frechet_alert:  alerts.append('FRECHET_ALERT')
        if topology_alert: alerts.append('TOPOLOGY_ALERT')
        if isotropy_alert: alerts.append('ISOTROPY_ALERT')

        # Features mais afetadas (maior delta na média SHAP)
        delta = np.abs(sv1.mean(0) - sv2.mean(0))
        top_idx = np.argsort(delta)[::-1][:3]
        affected = [(feature_names[i], round(float(delta[i]), 4))
                    for i in top_idx if i < len(feature_names)]

        # Canary score: 100 = idêntico, 0 = completamente diferente
        n_alerts = len(alerts)
        canary_score = max(0.0,
            100
            - n_alerts * 20
            - min(fd / self.frechet_thresh, 3) * 10
            - max(0, self.topology_thresh - rho) * 50
            - iso_drift * 30)

        return {
            'frechet_distance':  round(fd, 4),
            'frechet_alert':     frechet_alert,
            'topology_rho':      rho,
            'topology_p':        p_topo,
            'topology_alert':    topology_alert,
            'isotropy_v1':       round(iso1, 6),
            'isotropy_v2':       round(iso2, 6),
            'isotropy_drift':    round(iso_drift, 6),
            'isotropy_alert':    isotropy_alert,
            'alerts':            alerts,
            'affected_features': affected,
            'canary_score':      round(canary_score, 1),
        }

    def report(self, result: dict) -> str:
        alerts = result['alerts']
        icon   = "🟢" if not alerts else ("🟡" if len(alerts)==1 else "🔴")
        lines  = [
            f"GeometricCanaryDetector: {icon}  Score={result['canary_score']}/100",
            f"  Fréchet distance:   {result['frechet_distance']:.4f}"
            f"  {'⚠ FRECHET_ALERT' if result['frechet_alert'] else '✅'}",
            f"  Topology ρ={result['topology_rho']:.3f}  p={result['topology_p']:.4f}"
            f"  {'⚠ TOPOLOGY_ALERT' if result['topology_alert'] else '✅'}",
            f"  Isotropy drift:     {result['isotropy_drift']:.6f}"
            f"  {'⚠ ISOTROPY_ALERT' if result['isotropy_alert'] else '✅'}",
        ]
        if alerts:
            lines.append(f"  Alerts: {', '.join(alerts)}")
            lines.append(f"  Features mais afetadas: "
                        f"{result['affected_features']}")
        return "\n".join(lines)
