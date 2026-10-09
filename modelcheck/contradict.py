"""Detecta contradições entre a direção do impacto SHAP e regras de negócio.

Suporta relações não-lineares (análise por bins) e features categóricas.
"""
import numpy as np
import pandas as pd




def _suggest_action(item, X, feature_name):
    """Sugestões práticas conforme o padrão da contradição."""
    suggestions = []
    if item["severity"] == "check":
        suggestions.append("Relação não-linear: considere constraint monotônica "
                           "ou análise por faixas antes de concluir.")
        suggestions.append(f"Feature engineering: padrão em U detectado — teste "
                           f"adicionar '{feature_name}²' como feature ou usar "
                           "modelo não-linear (árvores).")
    else:
        suggestions.append("Verifique vazamento de dados (feature derivada do target?).")
        suggestions.append("Considere retreinar sem essa feature ou com "
                           "constraint monotônica (XGBoost/LightGBM suportam).")
    # multicolinearidade: correlação alta com outra feature numérica
    num = X.select_dtypes("number")
    if feature_name in num.columns and num.shape[1] > 1:
        corrs = num.corr()[feature_name].drop(feature_name).abs()
        top = corrs.idxmax()
        if corrs.max() > 0.7:
            suggestions.append(
                f"Alta correlação com '{top}' ({corrs.max():.2f}): possível "
                "multicolinearidade invertendo a direção — avalie remover uma das duas.")
    return suggestions


class ContradictionDetector:
    def __init__(self, rule_set, min_strength=0.1, n_bins=5):
        self.rule_set = rule_set
        self.min_strength = min_strength
        self.n_bins = n_bins

    def _analyze_numeric(self, feature_values, feature_shap):
        """Retorna (direção, força, monotônica?).

        Divide a feature em bins e verifica se o SHAP médio
        sobe/desce de forma consistente — pega relações não-lineares.
        """
        if np.std(feature_values) == 0 or np.std(feature_shap) == 0:
            return None, 0.0, True

        try:
            bins = pd.qcut(feature_values, self.n_bins, duplicates="drop")
        except ValueError:
            return None, 0.0, True

        df = pd.DataFrame({"bin": bins, "shap": feature_shap})
        means = df.groupby("bin", observed=True)["shap"].mean().to_numpy()

        if len(means) < 2:
            return None, 0.0, True

        diffs = np.diff(means)
        n_up = int((diffs > 0).sum())
        n_down = int((diffs < 0).sum())
        total = len(diffs)

        # monotonia: fração de passos na direção dominante
        monotonic_ratio = max(n_up, n_down) / total if total else 0
        is_monotonic = monotonic_ratio >= 0.8

        corr = np.corrcoef(feature_values, feature_shap)[0, 1]
        if np.isnan(corr):
            return None, 0.0, True

        direction = "positive" if corr > 0 else "negative"
        return direction, abs(corr), is_monotonic

    def _analyze_categorical(self, feature_values, feature_shap):
        """Para categóricas: sem direção global; retorna SHAP médio por categoria."""
        df = pd.DataFrame({"cat": feature_values, "shap": feature_shap})
        means = df.groupby("cat", observed=True)["shap"].mean()
        return means.round(4).to_dict()

    def check(self, shap_values, X):
        contradictions = []

        for feature_index, feature_name in enumerate(X.columns):
            rule = self.rule_set.get_rule_for_feature(feature_name)
            if not rule:
                continue

            col = X.iloc[:, feature_index]
            feature_shap = shap_values.values[:, feature_index]

            # categórica: direção positive/negative não se aplica
            if not pd.api.types.is_numeric_dtype(col):
                contradictions.append({
                    "feature": feature_name,
                    "expected": rule.expected_direction,
                    "found": "categorical",
                    "strength": None,
                    "severity": "info",
                    "rule_description": rule.description,
                    "note": "Feature categórica: regra direcional não se aplica. "
                            "SHAP médio por categoria: "
                            f"{self._analyze_categorical(col, feature_shap)}",
                })
                continue

            values = col.to_numpy(dtype=float)
            direction, strength, is_monotonic = self._analyze_numeric(
                values, feature_shap
            )

            if direction is None or strength < self.min_strength:
                continue

            if direction != rule.expected_direction:
                if not is_monotonic:
                    severity = "check"  # relação não-linear: revisar, não alarme
                elif strength > 0.6:
                    severity = "high"
                elif strength > 0.3:
                    severity = "medium"
                else:
                    severity = "low"

                item = {
                    "feature": feature_name,
                    "expected": rule.expected_direction,
                    "found": direction,
                    "strength": round(strength, 3),
                    "severity": severity,
                    "rule_description": rule.description,
                }
                if not is_monotonic:
                    item["note"] = (
                        "Relação não-monotônica detectada (ex: forma de U). "
                        "Direção global pode não representar o comportamento real."
                    )
                item["suggestions"] = _suggest_action(item, X, feature_name)
                contradictions.append(item)

        return contradictions
