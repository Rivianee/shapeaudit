"""Compara feature importance entre múltiplos modelos."""
import pandas as pd

from modelcheck.explainers.shap_explainer import ShapExplainer


class ModelComparator:
    def __init__(self, models: dict, X_background, max_shap_rows=2000):
        self.models = models
        self.X_background = X_background
        self.max_shap_rows = max_shap_rows
        self.importances = {}

    def compute_all(self, X):
        for name, model in self.models.items():
            explainer = ShapExplainer(model, self.X_background,
                                      max_shap_rows=self.max_shap_rows)
            self.importances[name] = explainer.feature_importance(X)
        return self.importances

    def build_comparison_table(self):
        tables = []
        for model_name, importance_df in self.importances.items():
            renamed = importance_df.rename(
                columns={"mean_abs_shap": model_name}
            )
            tables.append(renamed.set_index("feature"))

        comparison = pd.concat(tables, axis=1)
        return comparison

    def find_divergences(self, threshold=0.3):
        comparison = self.build_comparison_table()
        normalized = comparison.div(comparison.max(axis=0), axis=1)

        divergences = []
        for feature in normalized.index:
            row = normalized.loc[feature]
            spread = row.max() - row.min()
            if spread > threshold:
                divergences.append({
                    "feature": feature,
                    "spread": round(spread, 3),
                    "values": row.to_dict(),
                })

        return divergences
