"""Detector de possível vazamento de dados (target leakage)."""
import numpy as np
from scipy import stats


class LeakageDetector:
    def __init__(self, dominance_threshold=0.7, correlation_threshold=0.95):
        self.dominance_threshold = dominance_threshold
        self.correlation_threshold = correlation_threshold

    def check(self, shap_values, X, y=None):
        """Alerta se uma feature domina demais ou espelha o target."""
        importance = np.abs(shap_values.values).mean(axis=0)
        total = importance.sum()
        if total == 0:
            return []
        weights = importance / total
        alerts = []
        for i, feature in enumerate(X.columns):
            reasons = []
            if weights[i] > self.dominance_threshold:
                reasons.append(
                    f"domina {weights[i]:.0%} da importância total"
                )
            if y is not None and np.std(X.iloc[:, i].to_numpy(dtype=float)) > 0:
                corr, _ = stats.spearmanr(X.iloc[:, i], y)
                if abs(corr) > self.correlation_threshold:
                    reasons.append(
                        f"correlação quase perfeita com o target ({corr:.2f})"
                    )
            if reasons:
                alerts.append({
                    "feature": feature,
                    "weight": round(float(weights[i]), 3),
                    "reasons": reasons,
                    "message": (
                        f"Possível vazamento: '{feature}' "
                        + " e ".join(reasons)
                        + ". Verifique se essa feature existiria no momento "
                          "da predição."
                    ),
                })
        return alerts
