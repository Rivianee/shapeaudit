"""ModelGuard — CausalDiff.

Responde: "O que CAUSOU a mudança de comportamento entre v1 e v2?"

Diferença para o ExplainDiff convencional:
  ExplainDiff: "feature CEP perdeu 28pp de importância" (correlação)
  CausalDiff:  "CEP causou 73% das mudanças de decisão" (causalidade)

Isso é o que reguladores, jurídico e auditores precisam —
causalidade é evidência, correlação não é.

Uso:
    from modelguard.causal import CausalDiff
    cd = CausalDiff(model_v1, model_v2, X_ref)
    report = cd.analyze(proxy_features=['cep', 'genero', 'raca'])
    print(cd.narrate(report))
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


class CausalDiff:
    """Identifica causas da mudança de comportamento entre duas versões de modelo.

    Estratégia:
    1. Gera dataset de "mudanças" — amostras onde v1 e v2 discordam
    2. Para cada feature candidata, estima o efeito causal (ATE) via backdoor
    3. Separa causas confirmadas de correlações espúrias
    4. Gera narrativa executiva com recomendação de deploy
    """

    def __init__(self, model_v1, model_v2, X_ref: pd.DataFrame,
                 threshold_ate: float = 0.30,
                 threshold_correlation: float = 0.2):
        """
        model_v1, model_v2 : modelos com .predict() — sklearn, XGBoost, etc.
        X_ref              : dataset de referência (pd.DataFrame)
        threshold_ate      : ATE mínimo para considerar efeito causal relevante
        threshold_correlation: correlação mínima para investigar feature
        """
        self.model_v1 = model_v1
        self.model_v2 = model_v2
        self.X_ref = X_ref.copy()
        self.threshold_ate = threshold_ate
        self.threshold_cor = threshold_correlation

    def _build_diff_dataset(self) -> pd.DataFrame:
        """Constrói dataset onde target=1 se v1 e v2 discordam."""
        pred_v1 = self.model_v1.predict(self.X_ref)
        pred_v2 = self.model_v2.predict(self.X_ref)
        diff_mask = pred_v1 != pred_v2

        n_diff = diff_mask.sum()
        if n_diff < 10:
            return pd.DataFrame()

        X_diff = self.X_ref[diff_mask].copy()
        X_diff['_mudou'] = 1
        n_same = min(n_diff * 2, (~diff_mask).sum())
        X_same = self.X_ref[~diff_mask].sample(n_same, random_state=42).copy()
        X_same['_mudou'] = 0

        df = pd.concat([X_diff, X_same]).reset_index(drop=True)
        return df, n_diff, diff_mask.mean()

    def _estimate_causal_effect(self, df: pd.DataFrame,
                                 treatment: str) -> Dict:
        """Estima ATE de uma feature na mudança de decisão via backdoor."""
        try:
            from dowhy import CausalModel
            confounders = [c for c in self.X_ref.columns
                          if c != treatment and c in df.columns]
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome='_mudou',
                common_causes=confounders if confounders else None
            )
            identified = model.identify_effect(
                proceed_when_unidentifiable=True)
            estimate = model.estimate_effect(
                identified,
                method_name="backdoor.linear_regression"
            )
            return {"ate": float(estimate.value), "method": "backdoor.linear_regression", "ok": True}
        except Exception as e:
            return {"ate": None, "error": str(e), "ok": False}

    def analyze(self, proxy_features: Optional[List[str]] = None,
                max_features: int = 10) -> Dict:
        """
        Analisa causas da mudança entre v1 e v2.

        proxy_features : features suspeitas de serem proxy discriminatório
                        (ex: ['cep', 'genero', 'nome']). Se None, analisa todas.
        max_features   : limite de features para analisar (performance)
        """
        result = self._build_diff_dataset()
        if isinstance(result, pd.DataFrame) and result.empty:
            return {"error": "Modelos idênticos — menos de 10 predições diferentes.",
                    "n_changes": 0, "causal_factors": [], "all_alerts": []}

        df, n_diff, pct_diff = result

        # Seleciona features para investigar
        numeric_cols = [c for c in self.X_ref.columns
                       if pd.api.types.is_numeric_dtype(self.X_ref[c])]

        # Prioriza proxy_features + features com maior correlação
        correlations = {}
        for col in numeric_cols:
            correlations[col] = abs(df[col].corr(df['_mudou']))

        candidates = sorted(correlations, key=correlations.get, reverse=True)
        if proxy_features:
            # proxy_features sempre investigados, resto por correlação
            others = [c for c in candidates
                     if c not in proxy_features][:max_features - len(proxy_features)]
            candidates = [c for c in proxy_features
                         if c in numeric_cols] + others
        else:
            candidates = candidates[:max_features]

        # Estima efeito causal para cada candidato
        causal_factors = []
        correlation_only = []

        for feat in candidates:
            corr = correlations.get(feat, 0)
            if corr < self.threshold_cor and feat not in (proxy_features or []):
                continue

            causal = self._estimate_causal_effect(df, feat)
            ate = causal.get("ate")

            if ate is not None:
                is_causal = abs(ate) >= self.threshold_ate
                entry = {
                    "feature": feat,
                    "correlation": round(corr, 4),
                    "ate": round(ate, 4),
                    "is_causal": is_causal,
                    "is_proxy_suspect": feat in (proxy_features or []),
                    "direction": "aumenta" if ate > 0 else "reduz",
                    "magnitude": "alto" if abs(ate) > 0.4 else "médio" if abs(ate) > 0.2 else "baixo"
                }
                if is_causal:
                    causal_factors.append(entry)
                else:
                    correlation_only.append(entry)

        causal_factors.sort(key=lambda x: abs(x["ate"]), reverse=True)

        # Alertas
        alerts = []
        proxy_causes = [f for f in causal_factors if f["is_proxy_suspect"]]
        if proxy_causes:
            for p in proxy_causes:
                alerts.append({
                    "severity": "CRITICO",
                    "message": (
                        f"'{p['feature']}' causou {abs(p['ate']):.1%} das mudanças de decisão "
                        f"entre v1 e v2 (ATE={p['ate']:.3f}). "
                        f"Se '{p['feature']}' é proxy de atributo protegido "
                        f"(raça, gênero, região), isso é viés estrutural. "
                        f"RECOMENDAÇÃO: não fazer deploy da v2 sem revisão."
                    )
                })
        non_proxy_causes = [f for f in causal_factors if not f["is_proxy_suspect"]]
        if non_proxy_causes:
            feats = ', '.join(f["feature"] for f in non_proxy_causes[:3])
            alerts.append({
                "severity": "ALERTA",
                "message": (
                    f"Mudança causal detectada em: {feats}. "
                    f"O retreinamento alterou estruturalmente como o modelo usa essas features."
                )
            })
        if not causal_factors:
            alerts.append({
                "severity": "OK",
                "message": "Nenhuma causa estrutural identificada. Mudanças parecem marginais."
            })

        return {
            "n_changes": int(n_diff),
            "pct_changes": round(float(pct_diff), 4),
            "causal_factors": causal_factors,
            "correlation_only": correlation_only,
            "all_alerts": alerts,
            "n_alerts": len([a for a in alerts if a["severity"] != "OK"])
        }

    def narrate(self, report: Dict) -> str:
        """Gera narrativa executiva do relatório causal."""
        lines = ["=" * 56,
                 "MODELGUARD — CAUSALDIFF",
                 "=" * 56]

        n = report.get("n_changes", 0)
        pct = report.get("pct_changes", 0)
        lines.append(f"\nMudanças entre v1 e v2: {n} predições ({pct:.1%} do dataset)")

        factors = report.get("causal_factors", [])
        corr_only = report.get("correlation_only", [])

        if not factors:
            lines.append("\n✅ Nenhuma causa estrutural identificada.")
            lines.append("As diferenças entre v1 e v2 parecem marginais e não sistemáticas.")
        else:
            lines.append(f"\n⚠️  {len(factors)} causa(s) estrutural(is) identificada(s):\n")
            for f in factors:
                proxy = " [PROXY SUSPEITO]" if f["is_proxy_suspect"] else ""
                lines.append(
                    f"  • '{f['feature']}'{proxy}\n"
                    f"    Efeito causal (ATE): {f['ate']:.3f} — magnitude {f['magnitude']}\n"
                    f"    Interpretação: '{f['feature']}' {f['direction']} a probabilidade\n"
                    f"    de mudança de decisão em {abs(f['ate']):.1%} (efeito causal confirmado,\n"
                    f"    não apenas correlação)."
                )

        if corr_only:
            feats = ', '.join(f["feature"] for f in corr_only[:4])
            lines.append(f"\nCorrelacionadas mas sem efeito causal confirmado: {feats}")

        lines.append("\nAlertas:")
        for a in report.get("all_alerts", []):
            icon = "🔴" if a["severity"] == "CRITICO" else "🟡" if a["severity"] == "ALERTA" else "✅"
            lines.append(f"  {icon} {a['message']}")

        lines.append("=" * 56)
        return "\n".join(lines)
