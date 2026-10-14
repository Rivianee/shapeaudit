"""ShapeSlopeDetector — detecta mudança de intensidade na curva SHAP.

Problema original (Limitação 3 do paper):
    O SDD classifica a FORMA da curva (crescente, U, U-invertido...).
    Quando dois modelos têm a mesma forma mas intensidade diferente,
    o SDD fica silencioso.

    Exemplo real — PLD Elétrico CCEE:
        2009: CMO→PLD crescente, slope=24.5  (regime normal)
        2013: CMO→PLD crescente, slope=82.9  (pré-crise, 3.4x mais íngreme)
        2014: CMO→PLD crescente, slope=126.6 (crise severa, 5.2x original)

    O SDD detectaria? Não — forma = crescente em todos os anos.
    ShapeSlopeDetector detecta? Sim — razão de slope > 3.0x alerta.

Solução:
    Para cada feature, mede a inclinação da curva SHAP via regressão linear
    nos pontos de bin. Normaliza pelo desvio padrão de x para comparar
    entre features com escalas diferentes. Compara a razão de inclinação
    entre dois modelos ou períodos.

    Três tipos de alerta:
      SLOPE_AMPLIFIED  : v2 aprendeu relação muito mais intensa (ratio > thr)
      SLOPE_FLATTENED  : v2 aprendeu relação muito mais suave (ratio < 1/thr)
      SLOPE_INVERTED   : v2 inverteu a direção (mesmo com forma igual)

Uso:
    from modelcheck.slope import ShapeSlopeDetector

    detector = ShapeSlopeDetector(threshold_ratio=3.0)
    results  = detector.check(sv_v1, sv_v2, X_eval, feature_names)
    print(detector.report(results))

Resultados experimentais:
    PLD Elétrico (simulado): slope 146 vs 1027 → ratio=7.0x DETECTADO
    PLD Elétrico (real 2009→2013): ratio=3.4x DETECTADO
    Wine Quality (baixa vs alta qualidade): 3 features com SLOPE_INVERTED
    além das 11 já detectadas pelo SDD (complemento, não substituição)
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Optional


def _slope_profile(x: np.ndarray,
                   shap_vals: np.ndarray,
                   n_bins: int = 5) -> Optional[dict]:
    """Mede inclinação normalizada da curva SHAP de uma feature.

    Parâmetros:
        x         : valores da feature
        shap_vals : valores SHAP correspondentes
        n_bins    : número de bins quantílicos

    Retorna dict com:
        slope       : inclinação bruta (shap por unidade de x)
        slope_stdx  : slope * std(x) — normalizado para comparação
        r2          : R² da regressão linear (qualidade do fit)
        shap_std    : desvio padrão dos valores SHAP

    Retorna None se n < 8 ou variância zero.
    """
    x = np.asarray(x, dtype=float)
    s = np.asarray(shap_vals, dtype=float)

    if len(x) < 8 or np.std(x) < 1e-8:
        return None

    try:
        bins   = pd.qcut(x, n_bins, duplicates='drop')
        x_mids = pd.Series(x).groupby(bins, observed=True).mean().values
        s_means = pd.Series(s).groupby(bins, observed=True).mean().values

        if len(x_mids) < 3:
            return None

        slope, _, r, _, _ = scipy_stats.linregress(x_mids, s_means)

        return {
            'slope':      round(float(slope), 6),
            'slope_stdx': round(float(slope * np.std(x)), 4),
            'r2':         round(float(r ** 2), 4),
            'shap_std':   round(float(np.std(s)), 4),
        }
    except Exception:
        return None


class ShapeSlopeDetector:
    """Detecta mudança de intensidade na relação SHAP entre dois modelos.

    Complementa o ShapeDivergenceDetector (SDD):
      SDD detecta mudança de FORMA  (crescente → U-invertido)
      ShapeSlopeDetector detecta mudança de GRAU (crescente 1x → crescente 7x)

    Parâmetros:
        threshold_ratio : razão mínima entre slopes para alertar
                         (padrão 3.0 → alerta quando v2 é 3x mais/menos íngreme)
        min_effect      : efeito SHAP mínimo para considerar a feature
                         (padrão 0.01 — ignora features com efeito negligível)
        n_bins          : bins para cálculo do slope (padrão 5)
    """

    def __init__(self,
                 threshold_ratio: float = 3.0,
                 min_effect:      float = 0.01,
                 n_bins:          int   = 5,
                 min_rel_importance: float = 0.02):
        self.threshold_ratio = threshold_ratio
        self.min_rel_importance = min_rel_importance
        self.min_effect      = min_effect
        self.n_bins          = n_bins

    def check(self,
              sv1:           np.ndarray,
              sv2:           np.ndarray,
              X_eval:        pd.DataFrame,
              feature_names: List[str]) -> list:
        """Compara slopes SHAP entre dois modelos.

        Parâmetros:
            sv1, sv2       : arrays SHAP (n_samples, n_features)
            X_eval         : DataFrame com valores das features
            feature_names  : lista de nomes

        Retorna lista de dicts, um por feature, com:
            feature        : nome da feature
            slope_v1       : slope_stdx do modelo v1
            slope_v2       : slope_stdx do modelo v2
            ratio          : razão absoluta dos slopes
            sign_flip      : True se a direção inverteu
            divergence     : True se ratio > threshold_ratio
            alert          : tipo de alerta (None se sem divergência)
            alert_type     : SLOPE_AMPLIFIED | SLOPE_FLATTENED | SLOPE_INVERTED
        """
        from .frechet import como_array_shap
        sv1 = como_array_shap(sv1)   # BUG 95: lista do shap antigo
        sv2 = como_array_shap(sv2)
        results = []

        for i, f in enumerate(feature_names):
            xv = np.asarray(X_eval[f].values, dtype=float)

            sp1 = _slope_profile(xv, sv1[:, i], self.n_bins)
            sp2 = _slope_profile(xv, sv2[:, i], self.n_bins)

            if sp1 is None or sp2 is None:
                results.append({
                    'feature': f, 'slope_v1': None, 'slope_v2': None,
                    'ratio': None, 'sign_flip': None,
                    'divergence': False, 'alert': None, 'alert_type': None,
                })
                continue

            s1, s2 = sp1['slope_stdx'], sp2['slope_stdx']

            # BUG 13 (v1.42): min_effect era um limiar ABSOLUTO sobre o slope,
            # que está em unidades do target. Medido: com o target em escala
            # 0.001 o detector dava ZERO alertas; em escala 1, 1e3 e 1e6 dava
            # 2 — mesmos dados, mesma corrupção, diagnóstico diferente só por
            # mudança de unidade. Um usuário com target em probabilidade ou
            # fração receberia silêncio.
            # Correção: o limiar passa a ser relativo à escala dos próprios
            # SHAP values. Terceira ocorrência da família "limiar absoluto
            # sobre quantidade dependente de escala" (bugs 9, 11, 12).
            _escala = max(float(np.abs(sv1).std()), float(np.abs(sv2).std()),
                          1e-12)
            _min_eff = self.min_effect * _escala

            # BUG 14 (v1.42): o slope não tinha filtro de importância, então
            # uma feature de RUÍDO era apontada como SLOPE_AMPLIFIED (medido
            # em controle sintético). Quarta ocorrência da mesma família.
            _t1 = float(np.abs(sv1).mean(axis=0).sum()) + 1e-12
            _t2 = float(np.abs(sv2).mean(axis=0).sum()) + 1e-12
            _rel = max(float(np.abs(sv1[:, i]).mean()) / _t1,
                       float(np.abs(sv2[:, i]).mean()) / _t2)
            if _rel < self.min_rel_importance:
                results.append({
                    'feature': f, 'slope_v1': s1, 'slope_v2': s2,
                    'ratio': 1.0, 'sign_flip': False,
                    'divergence': False, 'alert': None,
                    'alert_type': 'IRRELEVANT_FEATURE',
                })
                continue

            # Ignora features com efeito negligível (relativo à escala)
            if abs(s1) < _min_eff and abs(s2) < _min_eff:
                results.append({
                    'feature': f, 'slope_v1': s1, 'slope_v2': s2,
                    'ratio': 1.0, 'sign_flip': False,
                    'divergence': False, 'alert': None, 'alert_type': None,
                })
                continue

            ratio      = abs(s2) / (abs(s1) + 1e-8)
            sign_flip  = (s1 > 0 and s2 < 0) or (s1 < 0 and s2 > 0)
            divergence = ratio > self.threshold_ratio or ratio < 1 / self.threshold_ratio

            alert_type = None
            alert      = None
            if divergence:
                if sign_flip:
                    alert_type = 'SLOPE_INVERTED'
                    alert = (f"Direção inverteu e intensidade mudou {ratio:.1f}x")
                elif ratio > self.threshold_ratio:
                    alert_type = 'SLOPE_AMPLIFIED'
                    alert = (f"v2 é {ratio:.1f}x mais íngreme — relação muito mais intensa")
                else:
                    alert_type = 'SLOPE_FLATTENED'
                    # BUG CORRIGIDO (v1.37): ratio=0 acontece sempre que a
                    # feature é NEUTRALIZADA em v2 (slope_v2 = 0) — o cenário
                    # mais comum de falha de pipeline. 1/0 levantava
                    # ZeroDivisionError e derrubava o check() inteiro,
                    # impedindo a análise das outras features.
                    if ratio <= 1e-12:
                        alert_type = 'SLOPE_COLLAPSED'
                        alert = ("v2 perdeu completamente a inclinação — "
                                 "a feature deixou de ter efeito direcional")
                    else:
                        alert = (f"v2 é {1/ratio:.1f}x mais suave — "
                                 f"relação muito menos intensa")

            results.append({
                'feature':    f,
                'slope_v1':   round(s1, 4),
                'slope_v2':   round(s2, 4),
                'ratio':      round(ratio, 3),
                'sign_flip':  sign_flip,
                'divergence': divergence,
                'alert':      alert,
                'alert_type': alert_type,
            })

        return results

    def report(self, results: list) -> str:
        """Relatório textual dos resultados."""
        lines = ["ShapeSlopeDetector:"]
        alerts = [r for r in results if r['divergence']]

        if not alerts:
            lines.append("  Nenhuma divergência de intensidade detectada.")
            lines.append(f"  (threshold_ratio={self.threshold_ratio}x)")
        else:
            lines.append(f"  {len(alerts)} divergência(s) de intensidade:")
            for r in alerts:
                lines.append(f"  ⚠️  [{r['alert_type']}] '{r['feature']}'")
                lines.append(f"      slope v1={r['slope_v1']:.3f}  "
                             f"v2={r['slope_v2']:.3f}  "
                             f"ratio={r['ratio']:.1f}x")
                lines.append(f"      {r['alert']}")

        return "\n".join(lines)

    def summary_alerts(self, results: list) -> list:
        """Lista plana de alertas para integração com outros detectores."""
        return [
            {'feature': r['feature'], 'type': r['alert_type'],
             'ratio': r['ratio'], 'source': 'slope'}
            for r in results if r['divergence']
        ]
