"""Drift de explicação: compara SHAP entre dois momentos."""
import json
import numpy as np
import pandas as pd


class ExplanationSnapshot:
    """Salva/carrega a 'assinatura' de explicação de um modelo."""

    @staticmethod
    def create(shap_values, X):
        importance = np.abs(shap_values.values).mean(axis=0)
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return dict(zip(X.columns, importance.round(6)))

    @staticmethod
    def save(snapshot, path):
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

    @staticmethod
    def load(path):
        with open(path) as f:
            return json.load(f)


class ExplanationDriftDetector:
    def __init__(self, threshold=0.15):
        self.threshold = threshold

    def compare(self, baseline: dict, current: dict):
        drifts = []
        features = set(baseline) | set(current)
        for feature in sorted(features):
            before = baseline.get(feature, 0.0)
            now = current.get(feature, 0.0)
            change = abs(now - before)
            if change > self.threshold:
                drifts.append({
                    "feature": feature,
                    "baseline": round(before, 3),
                    "current": round(now, 3),
                    "change": round(change, 3),
                })
        return drifts
