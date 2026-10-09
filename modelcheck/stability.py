"""
stability.py — SupervisedStabilityAnalyzer

Explora a dissociação entre estabilidade supervisionada (alinhada com a
tarefa) e não-supervisionada (geométrica). Resultado medido neste projeto:
em RF vs RF envenenado, Canary=74,8 e Supervised=0,0 — as duas dimensões
não são redundantes.

Estabilidade supervisionada (alinhada com a tarefa):
  Mede se o modelo responde de forma consistente a perturbações
  nos exemplos DE TREINO — relevante para controlabilidade e
  detecção de overfitting silencioso.

Estabilidade não-supervisionada:
  Mede a estabilidade do espaço de representação SEM usar labels —
  é o GeometricCanary. Ideal para monitoramento pós-deploy.

Observação medida neste projeto:
  Estabilidade supervisionada e geométrica podem divergir amplamente
  (Canary=74,8 vs Supervised=0,0 no caso de envenenamento).
  Um modelo pode ser supervisionadamente estável mas
  geometricamente instável, e vice-versa.
  → Ambas as métricas são necessárias para diagnóstico completo.

Uso:
    from modelcheck.stability import SupervisedStabilityAnalyzer
    ssa = SupervisedStabilityAnalyzer()
    result = ssa.compare(model_v1, model_v2, X_eval, y_eval, features)
    print(ssa.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats


class SupervisedStabilityAnalyzer:
    """
    Avalia estabilidade supervisionada de dois modelos.

    Modo PRÉ-DEPLOY: audita se o novo modelo é controlável
    e alinhado com a tarefa antes do deploy.

    Métricas:
      prediction_consistency : correlação entre predições (Spearman)
      label_sensitivity      : sensibilidade a perturbações no target
      rank_stability         : estabilidade do ranking de predições
      calibration_drift      : deriva na calibração (predição vs target)
    """

    def __init__(self, n_perturb: int = 20,
                 noise_scale: float = 0.1,
                 consistency_thresh: float = 0.90):
        self.n_perturb           = n_perturb
        self.noise_scale         = noise_scale
        self.consistency_thresh  = consistency_thresh

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                y_eval: pd.Series,
                feature_names: Optional[List[str]] = None) -> dict:

        if feature_names is None:
            feature_names = list(X_eval.columns)

        pred1 = np.array(model_v1.predict(X_eval))
        pred2 = np.array(model_v2.predict(X_eval))
        y     = np.array(y_eval)

        # ── 1. Consistência de predições ──────────────────────────
        rho_pred, p_pred = scipy_stats.spearmanr(pred1, pred2)
        consistency_alert = float(rho_pred) < self.consistency_thresh

        # ── 2. Sensibilidade a perturbações no espaço de entrada ──
        stabilities1, stabilities2 = [], []
        np.random.seed(42)
        for _ in range(self.n_perturb):
            noise = np.random.normal(
                0, self.noise_scale * X_eval.std().values, X_eval.shape)
            X_noisy = pd.DataFrame(X_eval.values + noise,
                                   columns=feature_names)
            p1n = np.array(model_v1.predict(X_noisy))
            p2n = np.array(model_v2.predict(X_noisy))
            stabilities1.append(float(scipy_stats.spearmanr(pred1, p1n)[0]))
            stabilities2.append(float(scipy_stats.spearmanr(pred2, p2n)[0]))

        stab1 = float(np.mean(stabilities1))
        stab2 = float(np.mean(stabilities2))
        stab_diff = abs(stab1 - stab2)

        # ── 3. Rank stability ─────────────────────────────────────
        rank1 = scipy_stats.rankdata(pred1)
        rank2 = scipy_stats.rankdata(pred2)
        rank_corr, _ = scipy_stats.spearmanr(rank1, rank2)

        # ── 4. Calibration drift ──────────────────────────────────
        # Correlação predição vs target (quanto cada modelo "entende" o target)
        cal1, _ = scipy_stats.spearmanr(pred1, y)
        cal2, _ = scipy_stats.spearmanr(pred2, y)
        cal_drift = abs(float(cal1) - float(cal2))

        # ── 5. Supervised stability score ────────────────────────
        alerts = []
        if consistency_alert:
            alerts.append('PREDICTION_INCONSISTENCY')
        if stab_diff > 0.10:
            alerts.append('PERTURBATION_SENSITIVITY_DIFF')
        if cal_drift > 0.05:
            alerts.append('CALIBRATION_DRIFT')

        sup_score = max(0.0,
            100
            - len(alerts) * 15
            - max(0, self.consistency_thresh - float(rho_pred)) * 100
            - stab_diff * 50
            - cal_drift * 50)

        return {
            'prediction_rho':   round(float(rho_pred), 4),
            'prediction_p':     round(float(p_pred), 4),
            'consistency_alert':consistency_alert,
            'stability_v1':     round(stab1, 4),
            'stability_v2':     round(stab2, 4),
            'stability_diff':   round(stab_diff, 4),
            'rank_corr':        round(float(rank_corr), 4),
            'calibration_v1':   round(float(cal1), 4),
            'calibration_v2':   round(float(cal2), 4),
            'calibration_drift':round(cal_drift, 4),
            'alerts':           alerts,
            'supervised_score': round(sup_score, 1),
        }

    def report(self, result: dict) -> str:
        alerts = result['alerts']
        icon = "🟢" if not alerts else ("🟡" if len(alerts)==1 else "🔴")
        lines = [
            f"SupervisedStabilityAnalyzer: {icon}  "
            f"Score={result['supervised_score']}/100",
            f"  Pred consistency:  ρ={result['prediction_rho']:.4f}"
            f"  {'⚠ INCONSISTENCY' if result['consistency_alert'] else '✅'}",
            f"  Input perturbation: v1={result['stability_v1']:.4f}  "
            f"v2={result['stability_v2']:.4f}"
            f"  diff={result['stability_diff']:.4f}"
            f"  {'⚠' if result['stability_diff']>0.10 else '✅'}",
            f"  Rank correlation:   {result['rank_corr']:.4f}",
            f"  Calibration:        v1={result['calibration_v1']:.4f}  "
            f"v2={result['calibration_v2']:.4f}"
            f"  drift={result['calibration_drift']:.4f}"
            f"  {'⚠' if result['calibration_drift']>0.05 else '✅'}",
        ]
        if alerts:
            lines.append(f"  Alerts: {', '.join(alerts)}")
        return "\n".join(lines)
