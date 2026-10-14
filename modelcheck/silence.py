"""ShapeSilenceAnalyzer — categoriza o silêncio do SDD.

Problema original (Limitação 4 do paper):
    Quando o SDD não alerta, o usuário não sabe se é porque:
    A) Os modelos estão de acordo (bom sinal)
    B) Os dados têm variação insuficiente (inconclusivo)
    C) O drift é tão severo que o SDD não consegue classificar (perigo)

    Exemplo real — PLD 2021 vs 2022:
        2021: std=199 R$/MWh — variação real, SDD classifica
        2022: std=9.8 R$/MWh — preço tabelado, SHAP flat
        Modelo 2021 avaliado em 2022: R² = -910
        SDD: silêncio (shapes n/a ou iguais)
        Usuário não sabe que há um problema gravíssimo

Solução:
    ShapeSilenceAnalyzer categoriza cada silêncio em 5 tipos:

    STABLE        → shapes iguais e válidos → OK
    LOW_N         → n < min_n → inconclusivo
    CONST_X       → feature constante no período → verificar dados
    LOW_VAR_SHAP  → efeito SHAP negligível → feature irrelevante ou target flat
    DIST_MISMATCH → distribuições muito diferentes → drift severo, SDD inválido

    Severidades: ok → info → warning → error

Uso:
    from modelcheck.silence import ShapeSilenceAnalyzer

    analyzer = ShapeSilenceAnalyzer()
    results  = analyzer.analyze(sv1, sv2, X_eval1, X_eval2,
                                feature_names, shapes1, shapes2)
    print(analyzer.report(results))

Resultados experimentais:
    PLD 2021 vs 2022 (drift severo):
      MES: DIST_MISMATCH (KS p=0.000, Wasserstein=1.8) → 🔴 ERRO
      DIA_ANO: STABLE → ✅ OK
    Wine Quality (RF vs mesma arquitetura):
      volatile acidity, sulphates, alcohol: STABLE
      features de baixo efeito: LOW_VAR_SHAP (informativo, não crítico)
"""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Optional, Union


class ShapeSilenceAnalyzer:
    """Categoriza o silêncio do SDD em tipos interpretáveis.

    Parâmetros:
        min_n           : mínimo de amostras para resultado válido (padrão 30)
        min_std_x       : desvio padrão mínimo da feature (padrão 1e-4)
        min_shap_std    : desvio padrão mínimo do SHAP para ser relevante (0.005)
        ks_pvalue_thr   : p-value KS para detectar drift de distribuição (0.01)
        wasserstein_thr : distância Wasserstein normalizada para drift severo (1.0)
    """

    def __init__(self,
                 min_n:          int   = 30,
                 min_std_x:      float = 1e-4,
                 min_shap_std:   float = 0.005,
                 ks_pvalue_thr:  float = 0.01,
                 wasserstein_thr: float = 1.0):
        self.min_n           = min_n
        self.min_std_x       = min_std_x
        self.min_shap_std    = min_shap_std
        self.ks_pvalue_thr   = ks_pvalue_thr
        self.wasserstein_thr = wasserstein_thr

    def analyze_feature(self,
                        x1:     np.ndarray,
                        x2:     np.ndarray,
                        sv1_f:  np.ndarray,
                        sv2_f:  np.ndarray,
                        shape1: Optional[str] = None,
                        shape2: Optional[str] = None) -> dict:
        """Categoriza o silêncio para uma feature."""
        n1, n2 = len(x1), len(x2)

        # 1. Amostra pequena
        if n1 < self.min_n or n2 < self.min_n:
            return {
                'category': 'LOW_N',
                'severity': 'info',
                'message':  (f"n={min(n1,n2)} < {self.min_n} — "
                             f"amostra pequena, resultado inconclusivo"),
                'action':   "Colete mais dados antes de interpretar o silêncio",
                'details':  {'n1': n1, 'n2': n2}
            }

        # 2. Feature constante
        std_x1, std_x2 = float(np.std(x1)), float(np.std(x2))
        if std_x1 < self.min_std_x or std_x2 < self.min_std_x:
            return {
                'category': 'CONST_X',
                'severity': 'warning',
                'message':  (f"feature quase constante: "
                             f"std_v1={std_x1:.6f}, std_v2={std_x2:.6f}"),
                'action':   "Verifique se a feature tem variação real nos dados",
                'details':  {'std_x1': std_x1, 'std_x2': std_x2}
            }

        # 3. Efeito SHAP negligível (feature irrelevante ou target flat)
        shap_std1, shap_std2 = float(np.std(sv1_f)), float(np.std(sv2_f))
        if shap_std1 < self.min_shap_std and shap_std2 < self.min_shap_std:
            return {
                'category': 'LOW_VAR_SHAP',
                'severity': 'info',
                'message':  (f"efeito SHAP negligível: "
                             f"std_v1={shap_std1:.4f}, std_v2={shap_std2:.4f} — "
                             f"feature tem pouco impacto ou target sem variação"),
                'action':   ("Verifique variação do target. "
                             "Se target está constante, o silêncio é esperado."),
                'details':  {'shap_std1': shap_std1, 'shap_std2': shap_std2}
            }

        # 4. Drift severo — mede nos valores SHAP (não em X)
        # O drift de X é irrelevante se as features têm mesma distribuição
        # O que importa é se o MODELO aprendeu relações diferentes
        # → medir KS e Wasserstein nos SHAPs detecta isso corretamente
        try:
            ks_stat, ks_p = scipy_stats.ks_2samp(sv1_f, sv2_f)
            # Normaliza por std para comparar features com escalas diferentes
            sv1_norm = sv1_f / (np.std(sv1_f) + 1e-8)
            sv2_norm = sv2_f / (np.std(sv2_f) + 1e-8)
            w_dist = float(scipy_stats.wasserstein_distance(sv1_norm, sv2_norm))

            if ks_p < self.ks_pvalue_thr and w_dist > self.wasserstein_thr:
                return {
                    'category': 'DIST_MISMATCH',
                    'severity': 'error',
                    'message':  (f"distribuições SHAP muito diferentes: "
                                 f"KS p={ks_p:.4f}, Wasserstein={w_dist:.3f} — "
                                 f"modelos aprenderam relações estruturalmente "
                                 f"diferentes, SDD pode ser inválido"),
                    'action':   ("ATENÇÃO: o SDD está silencioso mas os modelos "
                                 "aprenderam relações diferentes nesta feature. "
                                 "Verifique R² cross-period — pode ser muito negativo. "
                                 "Retreine em dados do mesmo regime."),
                    'details':  {'ks_stat': round(ks_stat,4),
                                 'ks_p': round(ks_p,6),
                                 'wasserstein': round(w_dist,3)}
                }
        except Exception:
            pass

        # 5. Shapes iguais → estável
        if (shape1 and shape2
                and shape1 != 'n/a' and shape2 != 'n/a'
                and shape1 == shape2):
            return {
                'category': 'STABLE',
                'severity': 'ok',
                'message':  (f"forma estável ({shape1}) — "
                             f"modelos concordam nesta feature"),
                'action':   None,
                'details':  {'shape': shape1}
            }

        # 6. Indeterminado
        return {
            'category': 'UNDETERMINED',
            'severity': 'info',
            'message':  "forma não classificável — verificar manualmente",
            'action':   "Inspecione a feature manualmente",
            'details':  {'shape1': shape1, 'shape2': shape2}
        }

    def analyze(self,
                sv1:           np.ndarray,
                sv2:           np.ndarray,
                X_eval1:       Union[pd.DataFrame, np.ndarray],
                X_eval2:       Union[pd.DataFrame, np.ndarray],
                feature_names: List[str],
                shapes1:       Optional[dict] = None,
                shapes2:       Optional[dict] = None) -> list:
        """Analisa silêncios para todas as features."""
        from .frechet import como_array_shap
        sv1 = como_array_shap(sv1); sv2 = como_array_shap(sv2)   # BUG 95
        results = []

        for i, f in enumerate(feature_names):
            x1 = (np.asarray(X_eval1[f].values)
                  if hasattr(X_eval1, 'columns') else X_eval1[:, i])
            x2 = (np.asarray(X_eval2[f].values)
                  if hasattr(X_eval2, 'columns') else X_eval2[:, i])
            sh1 = shapes1.get(f) if shapes1 else None
            sh2 = shapes2.get(f) if shapes2 else None
            r = self.analyze_feature(
                np.asarray(x1, dtype=float),
                np.asarray(x2, dtype=float),
                sv1[:, i], sv2[:, i], sh1, sh2)
            r['feature'] = f
            results.append(r)

        return results

    def report(self, results: list) -> str:
        """Relatório textual com priorização por severidade."""
        sev_icon = {'error': '🔴', 'warning': '🟡', 'info': '🔵', 'ok': '✅'}
        sev_order = {'error': 0, 'warning': 1, 'info': 2, 'ok': 3}
        by_sev = {}
        for r in results:
            s = r['severity']
            by_sev.setdefault(s, []).append(r)

        lines = ["ShapeSilenceAnalyzer — Diagnóstico de Silêncios:"]

        for sev in ['error', 'warning', 'info', 'ok']:
            if sev not in by_sev:
                continue
            grp = by_sev[sev]
            label = {
                'error':   f"DRIFT SEVERO — Silêncio enganoso ({len(grp)} feature(s))",
                'warning': f"ATENÇÃO — Dados insuficientes ({len(grp)} feature(s))",
                'info':    f"INFORMATIVO ({len(grp)} feature(s))",
                'ok':      f"ESTÁVEL ({len(grp)} feature(s))",
            }[sev]
            lines.append(f"\n  {sev_icon[sev]} {label}:")
            for r in grp:
                lines.append(f"    '{r['feature']}' [{r['category']}]: {r['message']}")
                if r.get('action'):
                    lines.append(f"    → {r['action']}")

        # Resumo
        n_errors = len(by_sev.get('error', []))
        n_warn   = len(by_sev.get('warning', []))
        n_stable = len(by_sev.get('ok', []))
        lines.append(f"\n  Resumo: {n_errors} drift severo | "
                     f"{n_warn} atenção | {n_stable} estável")
        if n_errors > 0:
            lines.append("  ⚠️  AÇÃO NECESSÁRIA: silêncio não significa estabilidade!")

        return "\n".join(lines)

    def summary(self, results: list) -> dict:
        """Resumo estruturado para uso programático."""
        cats = {}
        for r in results:
            cats.setdefault(r['category'], []).append(r['feature'])
        return {
            'has_severe_drift': any(r['severity'] == 'error' for r in results),
            'categories':       cats,
            'n_stable':         len([r for r in results if r['category'] == 'STABLE']),
            'n_errors':         len([r for r in results if r['severity'] == 'error']),
        }
