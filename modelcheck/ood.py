"""
ood.py — OODBehavioralDetector

Detecta se dois modelos se comportam de forma diferente
em situações extremas / fora da distribuição (OOD).

Dois modelos podem concordar perfeitamente nos dados de treino
mas divergir dramaticamente em cenários de crise.

Para o setor elétrico: como os modelos se comportam quando
PLD = 3x o máximo histórico? EAR cai para 5%?

Uso:
    from modelcheck.ood import OODBehavioralDetector
    det = OODBehavioralDetector()
    result = det.analyze(model_v1, model_v2, X_eval, features,
                         stress_features=['EAR', 'CMO'])
    print(det.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class OODBehavioralDetector:
    """
    Testa comportamento dos modelos em cenários extremos.

    Métodos de stress:
      - percentile_extreme: push features para p1 ou p99
      - multiplier: multiplica features por um fator
      - zero_out: zera features (testa dependência)
    """

    def __init__(self,
                 stress_multiplier: float = 3.0,
                 percentile_low: float = 1.0,
                 percentile_high: float = 99.0,
                 divergence_threshold: float = 0.20):
        self.stress_multiplier    = stress_multiplier
        self.percentile_low       = percentile_low
        self.percentile_high      = percentile_high
        self.divergence_threshold = divergence_threshold

    def _stress_dataset(self, X: pd.DataFrame,
                        feature: str, mode: str) -> pd.DataFrame:
        Xs = X.copy()
        if mode == 'max_extreme':
            Xs[feature] = np.percentile(X[feature], self.percentile_high)
        elif mode == 'min_extreme':
            Xs[feature] = np.percentile(X[feature], self.percentile_low)
        elif mode == 'multiplier_up':
            Xs[feature] = X[feature] * self.stress_multiplier
        elif mode == 'multiplier_down':
            Xs[feature] = X[feature] / self.stress_multiplier
        elif mode == 'zero_out':
            Xs[feature] = 0.0
        return Xs

    def analyze(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None,
                stress_features: Optional[List[str]] = None) -> dict:
        """
        Compara predições dos modelos em cenários de stress.

        Retorna dict com resultados por feature e cenário.
        """
        if feature_names is None:
            feature_names = list(X_eval.columns)
        if stress_features is None:
            stress_features = feature_names

        # Baseline: predições nos dados normais
        pred_base_v1 = np.array(model_v1.predict(X_eval))
        pred_base_v2 = np.array(model_v2.predict(X_eval))
        base_agreement = float(np.abs(pred_base_v1 - pred_base_v2).mean())

        results = {}
        modes = ['max_extreme', 'min_extreme', 'multiplier_up']

        for f in stress_features:
            feat_results = []
            for mode in modes:
                Xs = self._stress_dataset(X_eval, f, mode)
                pred_v1 = np.array(model_v1.predict(Xs))
                pred_v2 = np.array(model_v2.predict(Xs))

                # Divergência nos dados estressados
                stress_div = float(np.abs(pred_v1 - pred_v2).mean())
                # Mudança relativa em relação ao baseline
                div_ratio = stress_div / (base_agreement + 1e-8)
                # Mudança na resposta de cada modelo
                v1_response = float(np.mean(pred_v1 - pred_base_v1))
                v2_response = float(np.mean(pred_v2 - pred_base_v2))
                response_divergence = abs(v1_response - v2_response)
                # Alerta: modelos respondem de forma diferente ao stress
                alert = (div_ratio > (1 + self.divergence_threshold) or
                         response_divergence > np.std(pred_base_v1) * 0.5)

                feat_results.append({
                    'mode':               mode,
                    'stress_divergence':  round(stress_div, 4),
                    'div_ratio':          round(div_ratio, 3),
                    'v1_response':        round(v1_response, 4),
                    'v2_response':        round(v2_response, 4),
                    'response_divergence':round(response_divergence, 4),
                    'alert':              alert,
                })

            any_alert = any(r['alert'] for r in feat_results)
            results[f] = {
                'scenarios':  feat_results,
                'ood_alert':  any_alert,
                'alert_type': 'OOD_BEHAVIOR_DIVERGENCE' if any_alert else None,
            }

        ood_features = [f for f, r in results.items() if r['ood_alert']]
        return {
            'base_agreement':  round(base_agreement, 4),
            'feature_results': results,
            'ood_features':    ood_features,
            'n_ood_alerts':    len(ood_features),
        }

    def report(self, result: dict, unit: str = '') -> str:
        ood = result['ood_features']
        lines = [
            f"OODBehavioralDetector (threshold={self.divergence_threshold:.0%}):",
            f"  Divergência baseline: {result['base_agreement']:.4f}{unit}",
            f"  Features com OOD alert: {len(ood)}/{len(result['feature_results'])}",
        ]
        if not ood:
            lines.append("  ✅ Modelos concordam em todos os cenários extremos.")
        else:
            lines.append(f"  ⚠ OOD_BEHAVIOR_DIVERGENCE:")
            for f in ood:
                r = result['feature_results'][f]
                for s in r['scenarios']:
                    if s['alert']:
                        lines.append(
                            f"    '{f}' [{s['mode']}]: "
                            f"ratio={s['div_ratio']:.2f}x  "
                            f"v1Δ={s['v1_response']:+.3f}  "
                            f"v2Δ={s['v2_response']:+.3f}{unit}")
        return "\n".join(lines)
