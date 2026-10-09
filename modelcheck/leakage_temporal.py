"""
leakage_temporal.py — Vazamento TEMPORAL de dados

O `leakage.py` existente detecta dominância de importância e espelhamento do
target. Não cobre o caso mais comum e mais caro em produção: uma feature que
só fica conhecida DEPOIS do momento da predição.

Exemplo real do setor elétrico: usar a média mensal de ENA para prever o dia
seguinte. No treino a média está disponível (o mês acabou); na inferência,
não. O modelo tem R² alto no backtest e falha em produção.

Três assinaturas testáveis, todas sem precisar de metadados do usuário:

  1) LOOKAHEAD_CORRELATION
     A feature no instante t correlaciona mais com o target FUTURO (t+h) do
     que com o target presente. Sinal clássico de feature que já "sabe" o
     futuro.

  2) SUSPICIOUS_DOMINANCE_WITH_LAG
     A feature domina a importância E sua versão defasada (shift para o
     passado) perde quase toda a capacidade preditiva — indica que só a
     versão contemporânea/futura serve.

  3) TARGET_RECONSTRUCTION
     A feature reconstrói o target quase perfeitamente (|ρ| acima do limiar),
     mesmo sem ser o próprio target.

Requer dados em ordem temporal. Não aplicável a dados IID.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats


class TemporalLeakageDetector:
    """
    Detecta features que provavelmente não estarão disponíveis no momento
    da predição.

        det = TemporalLeakageDetector()
        r = det.check(df_ordenado, feature_names, target='y', horizon=1)
        print(det.report(r))
    """

    def __init__(self,
                 lookahead_margin: float = 0.05,
                 reconstruction_rho: float = 0.95,
                 lag_drop: float = 0.50,
                 min_rho: float = 0.30):
        self.lookahead_margin = lookahead_margin
        self.reconstruction_rho = reconstruction_rho
        self.lag_drop = lag_drop
        self.min_rho = min_rho

    def check(self, data: pd.DataFrame,
              feature_names: List[str],
              target: str,
              horizon: int = 1,
              lag: int = 1) -> dict:
        """
        data deve estar ORDENADA no tempo. `target` é a coluna alvo já
        alinhada ao horizonte de previsão. `horizon` é usado apenas para
        construir o target futuro de comparação.
        """
        d = data.reset_index(drop=True)
        y = d[target].to_numpy(dtype=float)
        y_fut = d[target].shift(-horizon).to_numpy(dtype=float)

        results, alerts = [], []
        for f in feature_names:
            if f not in d.columns or not pd.api.types.is_numeric_dtype(d[f]):
                continue
            x = d[f].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 30 or np.nanstd(x[ok]) < 1e-12:
                continue

            rho_now = abs(float(scipy_stats.spearmanr(x[ok], y[ok])[0] or 0.0))

            okf = np.isfinite(x) & np.isfinite(y_fut)
            rho_fut = (abs(float(scipy_stats.spearmanr(x[okf], y_fut[okf])[0] or 0.0))
                       if okf.sum() >= 30 else np.nan)

            x_lag = d[f].shift(lag).to_numpy(dtype=float)
            okl = np.isfinite(x_lag) & np.isfinite(y)
            rho_lag = (abs(float(scipy_stats.spearmanr(x_lag[okl], y[okl])[0] or 0.0))
                       if okl.sum() >= 30 else np.nan)

            feat_alerts = []
            # 1) correlaciona melhor com o futuro que com o presente
            if (np.isfinite(rho_fut) and rho_now >= self.min_rho
                    and rho_fut > rho_now + self.lookahead_margin):
                feat_alerts.append('LOOKAHEAD_CORRELATION')
            # 2) versão defasada perde muito poder
            if (np.isfinite(rho_lag) and rho_now >= self.min_rho
                    and (rho_now - rho_lag) / max(rho_now, 1e-9) > self.lag_drop):
                feat_alerts.append('SUSPICIOUS_DOMINANCE_WITH_LAG')
            # 3) reconstrói o target
            if rho_now >= self.reconstruction_rho:
                feat_alerts.append('TARGET_RECONSTRUCTION')

            results.append({'feature': f,
                            'rho_presente': round(rho_now, 4),
                            'rho_futuro': (None if not np.isfinite(rho_fut)
                                           else round(rho_fut, 4)),
                            'rho_defasado': (None if not np.isfinite(rho_lag)
                                             else round(rho_lag, 4)),
                            'alerts': feat_alerts})
            alerts.extend(feat_alerts)

        flagged = [r['feature'] for r in results if r['alerts']]
        return {'features': results, 'flagged': flagged,
                'alerts': sorted(set(alerts)),
                'leak_suspected': bool(flagged),
                'n_features': len(results)}

    def report(self, r: dict) -> str:
        L = [f"TemporalLeakageDetector: "
             f"{len(r['flagged'])}/{r['n_features']} features suspeitas"]
        if not r['flagged']:
            L.append("  ✅ Nenhum sinal de vazamento temporal.")
            return "\n".join(L)
        for row in r['features']:
            if not row['alerts']:
                continue
            L.append(f"  ⚠ '{row['feature']}': {', '.join(row['alerts'])}"
                     f"  ρ_presente={row['rho_presente']}"
                     f"  ρ_futuro={row['rho_futuro']}"
                     f"  ρ_defasado={row['rho_defasado']}")
        return "\n".join(L)

    def check_availability(self, data: pd.DataFrame,
                           availability_lag: dict,
                           target: str,
                           horizon: int = 1,
                           time_col: Optional[str] = None) -> dict:
        """
        Verificação por METADADOS — o caminho robusto.

        A detecção puramente estatística tem limite conhecido (ver
        LIMITATIONS): em problemas autorregressivos, uma feature legitimamente
        coincide com o target em algum lag, o que confunde qualquer assinatura
        de correlação cruzada. Medido: recall 3/5, e a assinatura de assimetria
        de correlação cruzada teve especificidade 0%.

        A solução correta é declarar, por feature, com quantos períodos de
        atraso ela fica disponível — é o que feature stores chamam de
        point-in-time correctness.

            availability_lag = {
                'ENA_MLT': 0,        # conhecida no próprio instante
                'ena_media_mes': 30, # só no fim do mês
                'ma7_centrada': -3,  # NEGATIVO: depende do futuro -> inválida
            }

        Regra: a feature é válida se `lag >= 0` e `lag <= horizon` não é
        exigido (features antigas são sempre válidas). Lag negativo significa
        dependência de dados futuros: vazamento certo.
        """
        results, alerts = [], []
        for f, lag in availability_lag.items():
            if f not in data.columns:
                results.append({'feature': f, 'lag': lag,
                                'status': 'COLUNA_AUSENTE',
                                'alerts': ['SCHEMA_MISMATCH']})
                alerts.append('SCHEMA_MISMATCH')
                continue
            fa = []
            if lag < 0:
                fa.append('FUTURE_DEPENDENT')
            elif lag == 0:
                fa.append('SAME_INSTANT_CHECK')
            results.append({'feature': f, 'lag': lag,
                            'status': ('INVÁLIDA' if lag < 0 else
                                       'VERIFICAR' if lag == 0 else 'OK'),
                            'alerts': fa})
            alerts.extend(fa)
        nao_declaradas = [c for c in data.columns
                          if c not in availability_lag and c != target
                          and c != time_col]
        if nao_declaradas:
            alerts.append('UNDECLARED_FEATURES')
        invalidas = [r['feature'] for r in results
                     if 'FUTURE_DEPENDENT' in r['alerts']]
        return {'features': results, 'invalid': invalidas,
                'undeclared': nao_declaradas,
                'alerts': sorted(set(alerts)),
                'leak_confirmed': bool(invalidas),
                'method': 'METADATA'}

    def plain(self, r: dict) -> str:
        if not r['flagged']:
            return ("Nenhuma variável parece usar informação que só estará "
                    "disponível depois do momento da previsão.")
        nomes = ", ".join(f"'{f}'" for f in r['flagged'][:4])
        return (f"Possível vazamento temporal em {nomes}. Essas variáveis "
                f"parecem carregar informação do futuro em relação ao momento "
                f"da previsão. Confirme se elas realmente estarão disponíveis "
                f"em produção no instante da inferência — se não, o "
                f"desempenho do backtest está superestimado.")
