"""Detector de vazamento temporal: features que vêm do futuro."""
import numpy as np
import pandas as pd
from scipy import stats


class TemporalLeakageDetector:
    def __init__(self, shift_max=3, mirror_threshold=0.95,
                 future_margin=0.15):
        self.shift_max = shift_max
        self.mirror_threshold = mirror_threshold
        self.future_margin = future_margin

    def check(self, X: pd.DataFrame, y: pd.Series, date_col: str):
        """Ordena por data e procura:
        (a) feature que é cópia futura de outra (f[t] ≈ g[t+k]);
        (b) feature que conhece o target futuro melhor que o presente.
        """
        df = X.copy()
        df["_y"] = y.values
        df = df.sort_values(date_col).reset_index(drop=True)
        numeric = [c for c in X.columns if c != date_col
                   and pd.api.types.is_numeric_dtype(X[c])]
        alerts = []

        for f in numeric:
            fv = df[f].to_numpy(dtype=float)
            if np.std(fv) == 0:
                continue

            # (a) espelho futuro de outra feature
            for g in numeric:
                if g == f:
                    continue
                gv = df[g].to_numpy(dtype=float)
                if np.std(gv) == 0:
                    continue
                for k in range(1, self.shift_max + 1):
                    c = np.corrcoef(fv[:-k], gv[k:])[0, 1]
                    if abs(c) > self.mirror_threshold:
                        alerts.append({
                            "feature": f, "type": "future_mirror",
                            "detail": f"'{f}'[t] ≈ '{g}'[t+{k}] (corr={c:.2f})",
                            "message": (
                                f"Vazamento temporal: '{f}' parece ser o valor "
                                f"FUTURO de '{g}' ({k} passo(s) à frente). "
                                "Essa informação não existiria no momento da "
                                "predição."),
                        })
                        break
                else:
                    continue
                break

            # (b) conhece o target futuro
            c_now, _ = stats.spearmanr(fv, df["_y"])
            best_future = 0.0
            for k in range(1, self.shift_max + 1):
                c_fut, _ = stats.spearmanr(fv[:-k], df["_y"].to_numpy()[k:])
                if abs(c_fut) > abs(best_future):
                    best_future = c_fut
            if (abs(best_future) > abs(c_now) + self.future_margin
                    and abs(best_future) > 0.5):
                alerts.append({
                    "feature": f, "type": "knows_future_target",
                    "detail": f"corr futuro={best_future:.2f} vs "
                              f"presente={c_now:.2f}",
                    "message": (
                        f"Vazamento temporal: '{f}' prevê o target FUTURO "
                        f"melhor que o presente ({best_future:.2f} vs "
                        f"{c_now:.2f}). Verifique se é derivada de dado "
                        "posterior à predição."),
                })
        return alerts
