"""Wrapper de explicabilidade: SHAP com fallback pra permutation em dados grandes."""
import numpy as np
import pandas as pd
import shap


class _PermutationResult:
    """Mesma interface de shap_values (.values) usando permutation importance."""
    def __init__(self, values):
        self.values = values


class ShapExplainer:
    def __init__(self, model, X_background, max_shap_rows=2000,
                 background_sample=100):
        self.model = model
        self.max_shap_rows = max_shap_rows
        self.method = "shap"

        if hasattr(model, "predict_proba"):
            self.predict_fn = lambda X: model.predict_proba(X)[:, 1]
        else:
            self.predict_fn = model.predict

        # amostra o background pra acelerar SHAP
        if len(X_background) > background_sample:
            X_background = X_background.sample(
                background_sample, random_state=0
            )
        self.X_background = X_background
        self.explainer = shap.Explainer(self.predict_fn, X_background)

    def explain(self, X):
        if len(X) > self.max_shap_rows:
            self.method = "permutation_fallback"
            return self._permutation_explain(X)
        self.method = "shap"
        return self.explainer(X)

    def _permutation_explain(self, X, n_repeats=3):
        """Aproximação rápida: impacto médio de permutar cada feature.

        Mais grosseira que SHAP, mas escala pra dados grandes.
        Direção estimada via correlação feature vs predição.
        """
        rng = np.random.RandomState(0)
        base_pred = self.predict_fn(X)
        values = np.zeros((len(X), X.shape[1]))

        for j in range(X.shape[1]):
            deltas = np.zeros(len(X))
            for _ in range(n_repeats):
                X_perm = X.copy()
                X_perm.iloc[:, j] = rng.permutation(X_perm.iloc[:, j].values)
                deltas += base_pred - self.predict_fn(X_perm)
            values[:, j] = deltas / n_repeats

        return _PermutationResult(values)

    def feature_importance(self, X):
        shap_values = self.explain(X)
        importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
        })
        importance = importance.sort_values("mean_abs_shap", ascending=False)
        return importance.reset_index(drop=True)
