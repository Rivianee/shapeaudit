"""
gtsf.py — Geometric-Topological Stability Framework (GTSF)

Framework unificado que integra todos os detectores em dois modos:

  MODO PRÉ-DEPLOY (supervisionado):
    Usa SupervisedStabilityAnalyzer + GeometricCanary
    Responde: "É seguro fazer deploy do modelo v2?"

  MODO PÓS-DEPLOY (não-supervisionado):
    Usa GeometricCanary + TopologicalDriftDetector + ShapeTrendingDetector
    Responde: "O modelo em produção está derivando?"

Insight (Shesha):
  Estabilidade supervisionada e não-supervisionada são não correlacionadas.
  Ambas são necessárias para diagnóstico completo.

Uso:
    from modelcheck.gtsf import GTSF
    gtsf = GTSF()

    # Pré-deploy
    result = gtsf.pre_deploy(model_v1, model_v2, X_eval, y_eval, features)
    print(gtsf.report(result))

    # Pós-deploy (sem y)
    result = gtsf.post_deploy(model_v1, model_v2, X_eval, features)
    print(gtsf.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional


class GTSF:
    """
    Geometric-Topological Stability Framework.

    Unifica GeometricCanary + TopologicalDrift + SupervisedStability
    em dois modos de operação:

      pre_deploy(model_v1, model_v2, X_eval, y_eval, features)
        → Auditoria completa antes do deploy

      post_deploy(model_v1, model_v2, X_eval, features)
        → Monitoramento contínuo sem labels
    """

    def __init__(self):
        from modelcheck.canary    import GeometricCanaryDetector
        from modelcheck.topology  import TopologicalDriftDetector
        from modelcheck.stability import SupervisedStabilityAnalyzer

        self.canary    = GeometricCanaryDetector(
            n_sample=120, frechet_thresh=3.0,
            topology_thresh=0.90, isotropy_thresh=0.10)
        self.topo      = TopologicalDriftDetector(
            n_sample=70, bottleneck_thresh=0.3, betti0_thresh=2)
        self.sup_stab  = SupervisedStabilityAnalyzer(
            n_perturb=15, consistency_thresh=0.90)

    def pre_deploy(self, model_v1, model_v2,
                   X_eval: pd.DataFrame,
                   y_eval: pd.Series,
                   feature_names: Optional[List[str]] = None) -> dict:
        """Auditoria pré-deploy com labels disponíveis."""
        if feature_names is None:
            feature_names = list(X_eval.columns)

        canary_r  = self.canary.compare(model_v1, model_v2,
                                         X_eval, feature_names)
        topo_r    = self.topo.compare(model_v1, model_v2,
                                       X_eval, feature_names)
        sup_r     = self.sup_stab.compare(model_v1, model_v2,
                                           X_eval, y_eval, feature_names)

        return self._assemble('pre_deploy', canary_r, topo_r, sup_r,
                              feature_names)

    def post_deploy(self, model_v1, model_v2,
                    X_eval: pd.DataFrame,
                    feature_names: Optional[List[str]] = None) -> dict:
        """Monitoramento pós-deploy sem labels."""
        if feature_names is None:
            feature_names = list(X_eval.columns)

        canary_r = self.canary.compare(model_v1, model_v2,
                                        X_eval, feature_names)
        topo_r   = self.topo.compare(model_v1, model_v2,
                                      X_eval, feature_names)

        return self._assemble('post_deploy', canary_r, topo_r, None,
                              feature_names)

    def _assemble(self, mode, canary_r, topo_r, sup_r,
                  feature_names) -> dict:
        all_alerts = (canary_r['alerts'] + topo_r['alerts'] +
                      (sup_r['alerts'] if sup_r else []))

        # Score unificado: média ponderada
        if sup_r:
            gtsf_score = round(
                0.40 * canary_r['canary_score'] +
                0.30 * topo_r['topo_score'] +
                0.30 * sup_r['supervised_score'], 1)
        else:
            gtsf_score = round(
                0.55 * canary_r['canary_score'] +
                0.45 * topo_r['topo_score'], 1)

        status = ("STABLE"   if gtsf_score >= 85 else
                  "UNSTABLE" if gtsf_score >= 65 else "CRITICAL")

        return {
            'mode':         mode,
            'gtsf_score':   gtsf_score,
            'status':       status,
            'all_alerts':   all_alerts,
            'canary':       canary_r,
            'topology':     topo_r,
            'supervised':   sup_r,
            'feature_names':feature_names,
        }

    def report(self, result: dict) -> str:
        from modelcheck.canary    import GeometricCanaryDetector
        from modelcheck.topology  import TopologicalDriftDetector
        from modelcheck.stability import SupervisedStabilityAnalyzer

        mode_label = ("PRÉ-DEPLOY" if result['mode'] == 'pre_deploy'
                      else "PÓS-DEPLOY")
        score = result['gtsf_score']
        status = result['status']
        icon = "🟢" if score >= 85 else ("🟡" if score >= 65 else "🔴")
        alerts = result['all_alerts']

        lines = [
            "=" * 64,
            f"  GTSF — Geometric-Topological Stability Framework",
            f"  Modo: {mode_label}",
            "=" * 64,
            f"  {icon}  GTSF Score: {score}/100  [{status}]",
            f"  Total alerts: {len(alerts)}",
            "",
        ]

        if alerts:
            lines.append(f"  ⚠ Alerts: {', '.join(alerts)}")
            lines.append("")

        lines.append("  ── Geometric Canary ──────────────────────────")
        lines.append(
            "  " + self.canary.report(result['canary'])
            .replace('\n', '\n  '))

        lines.append("\n  ── Topological Drift ─────────────────────────")
        lines.append(
            "  " + self.topo.report(result['topology'])
            .replace('\n', '\n  '))

        if result['supervised']:
            lines.append("\n  ── Supervised Stability (pré-deploy only) ────")
            lines.append(
                "  " + self.sup_stab.report(result['supervised'])
                .replace('\n', '\n  '))

        lines.append("\n" + "=" * 64)

        # Dissociação supervisionada vs não-supervisionada
        if result['supervised']:
            can_score = result['canary']['canary_score']
            sup_score = result['supervised']['supervised_score']
            delta = abs(can_score - sup_score)
            if delta > 15:
                lines.append(
                    f"  ℹ Dissociação geométrica/supervisionada detectada "
                    f"(Δ={delta:.1f})")
                lines.append(
                    f"    Canary={can_score}  Supervised={sup_score}")
                lines.append(
                    f"    → Verifique ambas as dimensões antes do deploy")

        return "\n".join(lines)
