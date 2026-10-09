"""Aprende regras de negócio dos dados, com score de confiança e avisos."""
import numpy as np
import pandas as pd
import yaml
from scipy import stats


class RuleLearner:
    def __init__(self, min_correlation=0.2, max_pvalue=0.05,
                 min_confidence=50, n_splits=4, random_state=0):
        self.min_correlation = min_correlation
        self.max_pvalue = max_pvalue
        self.min_confidence = min_confidence
        self.n_splits = n_splits
        self.random_state = random_state
        self.warnings = []

    def _confidence(self, values, y):
        """Score 0-100: estabilidade da direção entre subamostras + força."""
        rng = np.random.RandomState(self.random_state)
        n = len(values)
        directions = []
        for _ in range(self.n_splits):
            idx = rng.choice(n, size=max(20, n // 2), replace=False)
            c, _ = stats.spearmanr(values[idx], y.iloc[idx])
            if not np.isnan(c):
                directions.append(np.sign(c))
        if not directions:
            return 0
        stability = abs(np.mean(directions))  # 1.0 = mesma direção sempre
        corr, _ = stats.spearmanr(values, y)
        strength = min(abs(corr) / 0.6, 1.0)
        return int(round(100 * (0.6 * stability + 0.4 * strength)))

    def _check_data_health(self, X, y):
        self.warnings = []
        n, p = X.shape
        if n < 10 * p:
            self.warnings.append(
                f"POUCOS DADOS: {n} linhas para {p} features (recomendado ≥ "
                f"{10*p}). Regras aprendidas podem ser espúrias — revise antes de usar."
            )
        for col in X.columns:
            nan_pct = X[col].isna().mean()
            if nan_pct == 1.0:
                self.warnings.append(f"'{col}': 100% NaN — ignorada.")
            elif nan_pct > 0.4:
                self.warnings.append(
                    f"'{col}': {nan_pct:.0%} NaN — regra baseada em dados parciais."
                )
        for w in self.warnings:
            print(f"[modelcheck aviso] {w}")

    def learn(self, X: pd.DataFrame, y: pd.Series):
        self._check_data_health(X, y)
        # poucos dados: exige confiança alta pra reduzir regras espúrias
        min_conf = self.min_confidence
        if any("POUCOS DADOS" in w for w in self.warnings):
            min_conf = max(min_conf, 80)
        learned = []
        for feature in X.columns:
            if not pd.api.types.is_numeric_dtype(X[feature]):
                continue
            col = X[feature]
            mask = col.notna() & y.notna()
            if mask.sum() < 20:
                continue
            values = col[mask].to_numpy(dtype=float)
            y_valid = y[mask].reset_index(drop=True)
            if np.std(values) == 0:
                continue

            corr, pvalue = stats.spearmanr(values, y_valid)
            if np.isnan(corr) or abs(corr) < self.min_correlation \
                    or pvalue > self.max_pvalue:
                continue

            confidence = self._confidence(values, y_valid)
            direction = "positive" if corr > 0 else "negative"
            rule = {
                "feature": feature,
                "expected_direction": direction,
                "correlation": round(float(corr), 3),
                "pvalue": float(f"{pvalue:.2e}"),
                "confidence": confidence,
                "description": (
                    f"Aprendida dos dados: '{feature}' tem relação "
                    f"{direction} com o target (rho={corr:.2f}, "
                    f"confiança={confidence}%)"
                ),
            }
            if confidence < min_conf:
                rule["low_confidence"] = True
            learned.append(rule)
        return learned

    def to_yaml(self, learned_rules, path, include_low_confidence=False):
        rules = [
            {"feature": r["feature"],
             "expected_direction": r["expected_direction"],
             "description": r["description"]}
            for r in learned_rules
            if include_low_confidence or not r.get("low_confidence")
        ]
        with open(path, "w") as f:
            yaml.safe_dump({"rules": rules}, f, allow_unicode=True)
        return path


    def learn_conditional(self, X: pd.DataFrame, y: pd.Series,
                          condition_features=None, min_group=30):
        """Regras condicionais: direção de F pode depender de Z.

        Ex: "temp sobe -> alugueis sobem, mas só no inverno".
        condition_features: colunas condicionantes (categóricas ou binárias);
        numéricas são divididas pela mediana.
        """
        if condition_features is None:
            condition_features = [c for c in X.columns
                                  if X[c].nunique() <= 5]
        conditional = []
        numeric = [c for c in X.columns
                   if pd.api.types.is_numeric_dtype(X[c])
                   and c not in condition_features]
        for z in condition_features:
            if pd.api.types.is_numeric_dtype(X[z]) and X[z].nunique() > 5:
                groups = X[z] > X[z].median()
                labels = {True: f"{z}>mediana", False: f"{z}<=mediana"}
            else:
                groups = X[z]
                labels = {v: f"{z}={v}" for v in groups.unique()}
            for f in numeric:
                if f == z:
                    continue
                dirs = {}
                for val in pd.unique(groups):
                    mask = (groups == val) & X[f].notna() & y.notna()
                    if mask.sum() < min_group:
                        continue
                    corr, p = stats.spearmanr(X.loc[mask, f], y[mask])
                    if not np.isnan(corr) and abs(corr) >= 0.25 and p <= 0.05:
                        dirs[labels[val]] = ("positive" if corr > 0
                                             else "negative", round(corr, 2))
                directions = {d for d, _ in dirs.values()}
                if len(directions) > 1:  # direção muda conforme o grupo
                    conditional.append({
                        "feature": f, "condition": z, "by_group": dirs,
                        "description": (
                            f"A direção de '{f}' depende de '{z}': "
                            + "; ".join(f"{g}: {d} (rho={r})"
                                        for g, (d, r) in dirs.items())
                            + ". Regra global seria enganosa."),
                    })
        return conditional
