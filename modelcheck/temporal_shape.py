"""Temporal shape monitoring: detecta mudança de forma entre versões do modelo."""
import numpy as np
import pandas as pd
from modelcheck.shape import _shape_profile, ShapeDivergenceDetector
from modelcheck.explainers.shap_explainer import ShapExplainer


class ShapeSnapshot:
    """Salva a 'assinatura de forma' de um modelo em um dado momento."""

    @staticmethod
    def create(model, X_eval, attribution="shap", X_background=None):
        """Cria snapshot de formas para todas as features numéricas."""
        if attribution == "ale":
            from modelcheck.ale import ale_explain
            sv = ale_explain(model, X_eval)
        else:
            if X_background is None:
                X_background = X_eval
            sv = ShapExplainer(model, X_background).explain(X_eval)

        shapes = {}
        imp = np.abs(sv.values).mean(axis=0)
        tot = imp.sum()
        weights = imp / tot if tot > 0 else imp

        for i, col in enumerate(X_eval.columns):
            if not pd.api.types.is_numeric_dtype(X_eval[col]):
                continue
            p = _shape_profile(X_eval[col].to_numpy(dtype=float),
                               sv.values[:, i])
            shapes[col] = {
                "shape": p["shape"] if p else "undefined",
                # BUG 16 outra vez (v1.73): "monotonic" só existe no perfil
                # de amostra normal; o de amostra pequena não tem. Este
                # acesso quebrava com n < 30 — o mesmo defeito que já
                # corrigimos em shape.py e snapshot.py.
                "monotonic": p.get("monotonic") if p else None,
                "weight": round(float(weights[i]), 4),
            }
        return shapes

    @staticmethod
    def diff(snapshot_before: dict, snapshot_after: dict,
             min_weight=0.03) -> list:
        """Compara dois snapshots e retorna mudanças de forma."""
        changes = []
        all_features = set(snapshot_before) | set(snapshot_after)

        for feat in sorted(all_features):
            b = snapshot_before.get(feat)
            a = snapshot_after.get(feat)
            if not b or not a:
                continue
            # só features relevantes em ao menos uma versão
            if b["weight"] < min_weight and a["weight"] < min_weight:
                continue
            if b["shape"] != a["shape"]:
                changes.append({
                    "feature": feat,
                    "shape_before": b["shape"],
                    "shape_after": a["shape"],
                    "weight_before": b["weight"],
                    "weight_after": a["weight"],
                    "severity": "high" if (
                        b["monotonic"] != a["monotonic"]) else "medium",
                    "message": (
                        f"'{feat}': forma mudou de '{b['shape']}' para "
                        f"'{a['shape']}' entre versões "
                        f"({'mudança de monotonia' if b['monotonic'] != a['monotonic'] else 'mudança de curvatura'})."),
                })
        return changes
