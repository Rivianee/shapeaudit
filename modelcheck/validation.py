"""
validation.py — 4ª camada: consistência treino × inferência

Fecha a lacuna identificada empiricamente (v1.29): todos os outros
detectores comparam dois MODELOS no mesmo X_eval. Se as duas versões são
internamente coerentes mas o que cada uma espera receber difere do que
recebe, nenhuma comparação modelo-vs-modelo captura o problema.

Classe de falhas coberta:
  - train/serving skew (feature calculada de forma diferente)
  - mudança de escala/unidade entre treino e inferência
  - imputação diferente nos dois momentos
  - janela de agregação diferente
  - níveis categóricos inéditos
  - variação de taxa de ausentes

Custo computacional: apenas estatísticas descritivas e PSI. Sem SHAP,
sem ALE, sem bootstrap. Ordem de milissegundos.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Dict


def psi(ref: np.ndarray, cur: np.ndarray,
        n_bins: int = 10) -> Optional[float]:
    """Population Stability Index entre duas amostras contínuas."""
    ref = np.asarray(ref, dtype=float); cur = np.asarray(cur, dtype=float)
    ref = ref[np.isfinite(ref)]; cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return None
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(ref, bins=edges)
    b, _ = np.histogram(cur, bins=edges)
    a = np.clip(a / max(a.sum(), 1), 1e-6, None)
    b = np.clip(b / max(b.sum(), 1), 1e-6, None)
    return float(((b - a) * np.log(b / a)).sum())


class TrainServingValidator:
    """
    Compara a distribuição das features usadas no TREINO com a do conjunto
    de inferência. Detecta skew que os detectores modelo-vs-modelo não veem.

        v = TrainServingValidator()
        r = v.check(X_train, X_serving)
        print(v.report(r))

    Limiares (padrão da indústria para PSI):
        PSI < 0.10  estável
        0.10–0.25   deriva moderada
        > 0.25      deriva significativa

    Além do PSI, checa deslocamento de média em unidades de desvio-padrão
    do treino — captura mudança de escala que o PSI pode subestimar quando
    a forma da distribuição se mantém.
    """

    def __init__(self,
                 psi_warn: float = 0.10,
                 psi_alert: float = 0.25,
                 mean_shift_sd: float = 0.50,
                 scale_ratio: float = 1.50,
                 missing_delta: float = 0.05):
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.mean_shift_sd = mean_shift_sd
        self.scale_ratio = scale_ratio
        self.missing_delta = missing_delta

    def check(self, X_train: pd.DataFrame,
              X_serving: pd.DataFrame,
              feature_names: Optional[List[str]] = None) -> dict:
        if feature_names is None:
            feature_names = [c for c in X_train.columns
                             if c in X_serving.columns]

        results, alerts = [], []
        for f in feature_names:
            if f not in X_train.columns or f not in X_serving.columns:
                results.append({'feature': f, 'status': 'MISSING_COLUMN',
                                'alert': 'SCHEMA_MISMATCH'})
                alerts.append('SCHEMA_MISMATCH')
                continue

            tr, sv = X_train[f], X_serving[f]
            num = (pd.api.types.is_numeric_dtype(tr)
                   and pd.api.types.is_numeric_dtype(sv))

            miss_tr = float(tr.isna().mean())
            miss_sv = float(sv.isna().mean())
            miss_shift = abs(miss_sv - miss_tr)

            row = {'feature': f, 'missing_train': round(miss_tr, 4),
                   'missing_serving': round(miss_sv, 4),
                   'missing_shift': round(miss_shift, 4)}
            feat_alerts = []

            # BUG 39 (v1.0.1): binária rara acusava "dados mudaram" em todo
            # caso, inclusive nos saudáveis. No AI4I, `RNF` (falha aleatória,
            # ~0,2% de positivos) aparecia em 100% das auditorias. O PSI
            # compara bins de quantil; com dois valores e um deles raro, cada
            # oscilação de poucas linhas vira PSI alto. Para essas colunas a
            # pergunta certa é outra: a PROPORÇÃO de positivos mudou?
            bin_rara = False
            if num and tr.dropna().nunique() <= 2:
                p_tr = float(tr.dropna().astype(float).mean())
                p_sv = float(sv.dropna().astype(float).mean())
                if min(p_tr, 1 - p_tr) < .05:      # minoria abaixo de 5%
                    bin_rara = True
                    n_tr, n_sv = len(tr.dropna()), len(sv.dropna())
                    dif = abs(p_sv - p_tr)
                    pool = (p_tr * n_tr + p_sv * n_sv) / max(n_tr + n_sv, 1)
                    se = (pool * (1 - pool) * (1 / max(n_tr, 1)
                                               + 1 / max(n_sv, 1))) ** .5
                    z = dif / se if se > 1e-12 else 0.0
                    # exige diferença de proporção relevante E significativa
                    relevante = dif >= max(.01, .5 * min(p_tr, p_sv))
                    row.update({'kind': 'binaria_rara',
                                'taxa_train': round(p_tr, 5),
                                'taxa_serving': round(p_sv, 5),
                                'z': round(float(z), 2)})
                    if relevante and z >= 3.0:
                        feat_alerts.append('RATE_SHIFT')
                    # nível que a inferência traz e o treino nunca viu: é
                    # recodificação ou escala, não raridade amostral. Pega o
                    # caso em que o treino tem {0, 3} e a inferência {0, 1}.
                    niveis_tr = set(np.round(tr.dropna().astype(float), 6))
                    niveis_sv = set(np.round(sv.dropna().astype(float), 6))
                    if niveis_sv - niveis_tr:
                        feat_alerts.append('NEW_LEVEL')

            if num:
                a = tr.to_numpy(dtype=float); b = sv.to_numpy(dtype=float)
                af = a[np.isfinite(a)]; bf = b[np.isfinite(b)]
                sd = float(af.std()) if len(af) else 0.0
                m_tr = float(af.mean()) if len(af) else np.nan
                m_sv = float(bf.mean()) if len(bf) else np.nan
                s_tr = sd
                s_sv = float(bf.std()) if len(bf) else np.nan

                p = psi(af, bf)
                shift_sd = (abs(m_sv - m_tr) / sd) if sd > 1e-12 else 0.0
                ratio = (s_sv / s_tr) if s_tr > 1e-12 else 1.0
                # cobertura de faixa: fração do serving fora do range do treino
                lo, hi = (af.min(), af.max()) if len(af) else (np.nan, np.nan)
                oor = float(((bf < lo) | (bf > hi)).mean()) if len(bf) else 0.0

                row.update({'psi': None if p is None else round(p, 4),
                            'mean_train': round(m_tr, 4),
                            'mean_serving': round(m_sv, 4),
                            'mean_shift_sd': round(shift_sd, 3),
                            'std_ratio': round(ratio, 3),
                            'out_of_range': round(oor, 4),
                            'kind': 'numeric'})

                # v0.12.0: piso de ruído do PSI. Com poucas amostras por bin
                # o PSI oscila só por amostragem: num split aleatório do Air
                # Passengers (132 linhas) a camada acusava 5 de 5 variáveis.
                # Estimamos o piso dividindo o TREINO ao meio 20 vezes e
                # tomando o percentil 95 — não há modelo envolvido, só a
                # distribuição, então o nulo é legítimo.
                piso = 0.0
                if p is not None and len(af) >= 20:
                    r_ = np.random.RandomState(0)
                    nulos = []
                    for _ in range(20):
                        idx = r_.permutation(len(af)); meio = len(af) // 2
                        pn = psi(af[idx[:meio]], af[idx[meio:]])
                        if pn is not None:
                            nulos.append(pn)
                    if nulos:
                        piso = float(np.quantile(nulos, .95))
                row['psi_piso_ruido'] = round(piso, 4)
                if p is not None and p <= piso:
                    p = None            # dentro do ruído: não é sinal

                if bin_rara:
                    # BUG 39: numa binária rara o PSI não se aplica, e a
                    # variância da amostra de inferência pode ser ZERO só
                    # porque não caiu nenhum positivo nela — foi assim que
                    # `RNF` (0,15% de positivos) aparecia como "mudança de
                    # escala" em toda auditoria. Para essas colunas vale só
                    # o teste de proporção, feito acima.
                    p = None
                    if s_tr > 1e-12:          # o treino ainda varia
                        ratio = 1.0
                        shift_sd = 0.0
                        oor = 0.0
                if p is not None and p > self.psi_alert:
                    feat_alerts.append('PSI_ALERT')
                elif p is not None and p > self.psi_warn:
                    feat_alerts.append('PSI_WARN')
                if shift_sd > self.mean_shift_sd:
                    feat_alerts.append('MEAN_SHIFT')
                if ratio > self.scale_ratio or ratio < 1 / self.scale_ratio:
                    feat_alerts.append('SCALE_CHANGE')
                if oor > 0.05:
                    feat_alerts.append('OUT_OF_RANGE')
                if s_tr <= 1e-12 < (s_sv if s_sv == s_sv else 0):
                    # o treino virou constante e a inferência não: é o padrão
                    # de coluna perdida, e vale mesmo em binária rara
                    feat_alerts.append('CONSTANT_IN_TRAINING')
            else:
                lv_tr = set(pd.Series(tr.dropna().unique()).astype(str))
                lv_sv = set(pd.Series(sv.dropna().unique()).astype(str))
                novos = sorted(lv_sv - lv_tr)[:6]
                jac = (len(lv_tr & lv_sv) / max(len(lv_tr | lv_sv), 1))
                row.update({'kind': 'categorical', 'jaccard': round(jac, 3),
                            'novos_niveis': novos})
                if novos:
                    feat_alerts.append('NEW_LEVELS')
                if jac < 0.7:
                    feat_alerts.append('LEVEL_SHIFT')

            if miss_shift > self.missing_delta:
                feat_alerts.append('MISSING_SHIFT')

            row['alerts'] = feat_alerts
            results.append(row)
            alerts.extend(feat_alerts)

        # colunas presentes no serving mas ausentes no treino
        extra = [c for c in X_serving.columns if c not in X_train.columns]
        if extra:
            alerts.append('EXTRA_COLUMNS')

        flagged = [r['feature'] for r in results if r.get('alerts')]
        severe = [a for a in alerts if a in
                  ('SCHEMA_MISMATCH', 'PSI_ALERT', 'SCALE_CHANGE',
                   'MEAN_SHIFT', 'NEW_LEVELS')]
        skew_score = max(0.0, 100 - len(set(alerts)) * 12
                         - len(flagged) * 5)

        return {'features': results, 'flagged': flagged,
                'alerts': sorted(set(alerts)),
                'extra_columns': extra,
                'severe': bool(severe),
                'skew_score': round(skew_score, 1),
                'n_features': len(feature_names)}

    def report(self, r: dict) -> str:
        L = [f"TrainServingValidator: skew_score={r['skew_score']}/100",
             f"  Features com alerta: {len(r['flagged'])}/{r['n_features']}"]
        if not r['flagged'] and not r['extra_columns']:
            L.append("  ✅ Distribuição de inferência compatível com o treino.")
            return "\n".join(L)
        for row in r['features']:
            if not row.get('alerts'):
                continue
            if row.get('kind') == 'numeric':
                L.append(f"  ⚠ '{row['feature']}': {', '.join(row['alerts'])}"
                         f"  PSI={row.get('psi')}"
                         f"  média {row['mean_train']}→{row['mean_serving']}"
                         f"  (Δ={row['mean_shift_sd']} sd)"
                         f"  escala×{row['std_ratio']}")
            else:
                L.append(f"  ⚠ '{row['feature']}': {', '.join(row['alerts'])}"
                         f"  jaccard={row.get('jaccard')}"
                         f"  novos={row.get('novos_niveis')}")
        if r['extra_columns']:
            L.append(f"  ⚠ Colunas extras no serving: {r['extra_columns'][:5]}")
        return "\n".join(L)


def plain_report(r: dict) -> str:
    """Relatório em linguagem de negócio."""
    if not r['flagged'] and not r['extra_columns']:
        return ("Os dados de inferência têm a mesma distribuição dos dados "
                "de treino. Nenhum sinal de inconsistência de pipeline.")
    parts = []
    for row in r['features']:
        al = row.get('alerts') or []
        if 'SCALE_CHANGE' in al or 'MEAN_SHIFT' in al:
            parts.append(f"'{row['feature']}' chega numa escala diferente da "
                         f"que o modelo viu no treino")
        elif 'PSI_ALERT' in al:
            parts.append(f"'{row['feature']}' tem distribuição bem diferente "
                         f"da do treino")
        elif 'NEW_LEVELS' in al:
            parts.append(f"'{row['feature']}' traz categorias que não "
                         f"existiam no treino")
        elif 'MISSING_SHIFT' in al:
            parts.append(f"'{row['feature']}' está vindo com muito mais "
                         f"valores ausentes que no treino")
        elif 'SCHEMA_MISMATCH' in al:
            parts.append(f"'{row['feature']}' não existe nos dados de "
                         f"inferência")
    if not parts:
        return "Pequenas diferenças de distribuição, sem gravidade aparente."
    txt = "Inconsistência entre treino e inferência: " + "; ".join(parts[:4]) + "."
    return (txt + "\n\nIsso indica problema no pipeline de dados, não no "
            "modelo. Verifique como essas variáveis são calculadas nos dois "
            "momentos (agregação, unidade, imputação).")
