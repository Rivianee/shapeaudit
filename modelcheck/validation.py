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
from scipy import stats
from typing import List, Optional, Dict


def psi(ref: np.ndarray, cur: np.ndarray,
        n_bins: int = 10) -> Optional[float]:
    """Population Stability Index entre duas amostras contínuas."""
    ref = np.asarray(ref, dtype=float); cur = np.asarray(cur, dtype=float)
    ref = ref[np.isfinite(ref)]; cur = cur[np.isfinite(cur)]
    if len(ref) < 10 or len(cur) < 10:
        return None
    niveis = np.unique(ref)
    if len(niveis) < 2:
        return None
    if len(niveis) <= n_bins:
        # BUG 46: em feature discreta os quantis repetem valores e o
        # `edges[-1] = inf` fundia o nível mais alto com o anterior —
        # recodificar 4→3 num 0..4 uniforme dava PSI 0,0008. Com poucos
        # níveis, compara a frequência de cada nível (união dos dois lados).
        uni = np.union1d(niveis, np.unique(cur))
        a = np.array([(ref == v).sum() for v in uni], dtype=float)
        b = np.array([(cur == v).sum() for v in uni], dtype=float)
    else:
        edges = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
        if len(edges) < 3:
            return None
        # BUG 46: o topo só é sobrescrito por inf quando o máximo não é um
        # nível com massa; se for, ganha bin próprio [max, inf] em vez de
        # ser fundido com o penúltimo.
        massa_topo = float((ref == edges[-1]).mean())
        edges[0] = -np.inf
        if massa_topo >= 1.0 / (2 * n_bins):
            edges = np.append(edges, np.inf)
        else:
            edges[-1] = np.inf
        a, _ = np.histogram(ref, bins=edges)
        b, _ = np.histogram(cur, bins=edges)
    return _psi_de_contagens(a, b)


def _psi_de_contagens(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = np.clip(a / max(a.sum(), 1), 1e-6, None)
    b = np.clip(b / max(b.sum(), 1), 1e-6, None)
    return float(((b - a) * np.log(b / a)).sum())


def psi_categorico(ref, cur) -> Optional[float]:
    """PSI sobre frequências de categorias (união dos níveis, com epsilon)."""
    ref = pd.Series(ref).astype(str).to_numpy()
    cur = pd.Series(cur).astype(str).to_numpy()
    if len(ref) < 10 or len(cur) < 10:
        return None
    cod, _ = pd.factorize(np.concatenate([ref, cur]))
    return _psi_codigos(cod[:len(ref)], cod[len(ref):])


def _psi_codigos(a, b) -> Optional[float]:
    """PSI entre dois vetores de códigos inteiros de categoria."""
    if len(a) < 10 or len(b) < 10:
        return None
    k = int(max(a.max(), b.max())) + 1
    return _psi_de_contagens(np.bincount(a, minlength=k),
                             np.bincount(b, minlength=k))


def _piso_ruido(ref, n_cur: int, fn, n_rep: int = 50) -> float:
    """
    BUG 49: o piso de ruído do PSI era estimado com metades do treino
    (n/2 × n/2), mas a inferência costuma ser bem menor; com 200 linhas
    contra 10k o piso saía baixo demais e 17,5% das auditorias saudáveis
    acusavam PSI_WARN. O nulo agora sorteia uma amostra do TAMANHO da
    inferência e compara contra o resto do treino. E como o percentil 95
    empírico de 20 sorteios deixava ~5% de falso alarme POR feature (numa
    auditoria de 10 features, ~1 em 3 acusava algo), o piso é o quantil
    99,5% de uma gama ajustada por momentos aos sorteios — sob H0 o PSI é
    aproximadamente um qui-quadrado escalado.
    """
    n = len(ref)
    k = min(int(n_cur), n // 2)
    if k < 10 or n - k < 10:
        return 0.0
    r_ = np.random.RandomState(0)
    nulos = []
    for _ in range(n_rep):
        idx = r_.permutation(n)
        pn = fn(ref[idx[k:]], ref[idx[:k]])
        if pn is not None:
            nulos.append(pn)
    if len(nulos) < 5:
        return 0.0
    nulos = np.asarray(nulos)
    media, var = float(nulos.mean()), float(nulos.var(ddof=1))
    if media <= 1e-12 or var <= 1e-18:
        return float(nulos.max())
    return float(stats.gamma.ppf(.995, a=media ** 2 / var, scale=var / media))


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

            if num:
                a = tr.to_numpy(dtype=float, na_value=np.nan)
                b = sv.to_numpy(dtype=float, na_value=np.nan)
                af = a[np.isfinite(a)]; bf = b[np.isfinite(b)]
                # BUG 43: `isna()` não conta ±inf, e todo o caminho numérico
                # descarta não-finitos antes de cada estatística — uma coluna
                # de inferência inteira em inf não gerava alerta nenhum. Em
                # coluna numérica, não-finito conta como ausente.
                miss_tr = 1 - len(af) / len(a) if len(a) else 0.0
                miss_sv = 1 - len(bf) / len(b) if len(b) else 0.0
                inf_tr = float(np.isinf(a).mean()) if len(a) else 0.0
                inf_sv = float(np.isinf(b).mean()) if len(b) else 0.0
            else:
                miss_tr = float(tr.isna().mean())
                miss_sv = float(sv.isna().mean())
            miss_shift = abs(miss_sv - miss_tr)

            row = {'feature': f, 'missing_train': round(miss_tr, 4),
                   'missing_serving': round(miss_sv, 4),
                   'missing_shift': round(miss_shift, 4)}
            feat_alerts = []
            if num:
                row.update({'inf_train': round(inf_tr, 4),
                            'inf_serving': round(inf_sv, 4)})
                if (inf_sv - inf_tr > self.missing_delta
                        or (len(b) and not len(bf) and len(af))):
                    feat_alerts.append('NONFINITE')

            # BUG 39 (v1.0.1): binária rara acusava "dados mudaram" em todo
            # caso, inclusive nos saudáveis. No AI4I, `RNF` (falha aleatória,
            # ~0,2% de positivos) aparecia em 100% das auditorias. O PSI
            # compara bins de quantil; com dois valores e um deles raro, cada
            # oscilação de poucas linhas vira PSI alto. Para essas colunas a
            # pergunta certa é outra: a PROPORÇÃO de positivos mudou?
            # BUG 40 (v1.0.3): o teste de proporção acima era aplicado a
            # qualquer coluna de dois níveis, inclusive quando os valores não
            # eram uma codificação 0/1. Se a inferência traz um valor fora de
            # [0, 1] — uma coluna {0,1} que passou a chegar constante em 3,0,
            # ou em 6,5 — então `pool` sai de [0, 1], `pool * (1 - pool)` fica
            # negativo, e a raiz de um negativo em float devolve um COMPLEXO.
            # A comparação seguinte levantava
            #   TypeError: '>' not supported between 'complex' and 'float'
            # e a checagem treino × uso inteira morria, justamente no caso que
            # ela existe para pegar: coluna constante por pipeline quebrado.
            #
            # A causa não era a raiz quadrada, era o teste de proporção rodar
            # sobre algo que não é proporção. Agora:
            #   * valor de inferência fora dos níveis do treino é
            #     recodificação — vira NEW_LEVEL, sem passar pelo z;
            #   * a variância é limitada em zero de todo jeito, para nunca
            #     mais existir caminho que produza complexo.
            # BUG 45: binária comum (minoria 5–50%) nunca tinha teste de
            # taxa — o PSI devolvia None (<3 bordas) e média/desvio quase
            # nunca disparavam: 30%→50% passava. Toda coluna numérica com
            # ≤2 níveis no treino entra no teste de proporção; níveis que
            # não são {0,1} viram 0/1 pelo treino (nível maior = 1).
            binaria = False
            if num and len(af) and len(np.unique(af)) <= 2:
                binaria = True
                niveis_tr = sorted(set(np.round(af, 6)))
                niveis_sv = set(np.round(bf, 6))
                if len(niveis_tr) == 2:
                    alto = niveis_tr[1]
                    vt = (np.round(af, 6) == alto).astype(float)
                    vs = (np.round(bf, 6) == alto).astype(float)
                else:                      # treino constante
                    v0 = niveis_tr[0]
                    base = v0 if v0 in (0.0, 1.0) else 0.0
                    vt = np.full(len(af), base); vs = np.full(len(bf), base)
                p_tr = float(vt.mean())
                p_sv = float(vs.mean()) if len(vs) else float('nan')
                minoria = min(p_tr, 1 - p_tr)
                row.update({'kind': 'binaria_rara' if minoria < .05
                            else 'binaria',
                            'taxa_train': round(p_tr, 5),
                            'taxa_serving': round(p_sv, 5)})

                # nível que a inferência traz e o treino nunca viu: é
                # recodificação ou escala, não raridade amostral. Pega o
                # caso em que o treino tem {0, 1} e a inferência {3}.
                novos = niveis_sv - set(niveis_tr)
                if novos:
                    feat_alerts.append('NEW_LEVEL')
                    row['niveis_novos'] = sorted(novos)[:5]
                    # p_sv não é proporção: comparar taxas aqui não tem
                    # sentido. NEW_LEVEL já reportou o que importa.
                    row['z'] = None
                elif not len(vs):
                    row['z'] = None        # sem valor finito: BUG 43 cobre
                else:
                    n_tr, n_sv = len(vt), len(vs)
                    dif = abs(p_sv - p_tr)
                    pool = ((p_tr * n_tr + p_sv * n_sv)
                            / max(n_tr + n_sv, 1))
                    var = pool * (1 - pool) * (1 / max(n_tr, 1)
                                               + 1 / max(n_sv, 1))
                    se = max(var, 0.0) ** .5     # nunca complexo
                    # correção de continuidade (Yates): com poucas
                    # ocorrências (9 em 10k × 2 em 300) a aproximação normal
                    # exagera o z; sem ela a binária rara saudável voltava a
                    # alarmar (regressão do bug 39 ao tirar o piso de 0,01).
                    cc = .5 * (1 / max(n_tr, 1) + 1 / max(n_sv, 1))
                    z = max(dif - cc, 0.0) / se if se > 1e-12 else 0.0
                    # BUG 44: a relevância era `dif >= max(.01, .5*min(p))`;
                    # com taxa de inferência 0 sobrava só o piso de 0,01 e
                    # uma flag de 0,8% indo a 0% (z=12,8) passava. Agora é
                    # relativa à MAIOR taxa do lado minoritário (mudança de
                    # ~2× é relevante), com teto de 5 p.p. para binária comum.
                    if p_tr <= .5:
                        m_tr, m_sv = p_tr, p_sv
                    else:
                        m_tr, m_sv = 1 - p_tr, 1 - p_sv
                    relevante = dif >= min(.5 * max(m_tr, m_sv), .05)
                    row['z'] = round(float(z), 2)
                    if relevante and z >= 3.0:
                        feat_alerts.append('RATE_SHIFT')

            if num:
                sd = float(af.std()) if len(af) else 0.0
                m_tr = float(af.mean()) if len(af) else np.nan
                m_sv = float(bf.mean()) if len(bf) else np.nan
                s_tr = sd
                s_sv = float(bf.std()) if len(bf) else np.nan

                p = psi(af, bf)
                shift_sd = (abs(m_sv - m_tr) / sd) if sd > 1e-12 else 0.0
                # BUG 48: a razão de desvios-padrão disparava SCALE_CHANGE em
                # 35% dos splits saudáveis de lognormal(0; 1,5) (10k × 200):
                # o desvio de cauda pesada é dominado por poucos pontos. Em
                # coluna contínua (> 10 níveis) usa a razão de IQR; se o IQR
                # do treino é nulo, volta ao desvio-padrão.
                ratio = (s_sv / s_tr) if s_tr > 1e-12 else 1.0
                if len(np.unique(af)) > 10 and len(bf):
                    iqr_tr = float(np.subtract(*np.percentile(af, [75, 25])))
                    iqr_sv = float(np.subtract(*np.percentile(bf, [75, 25])))
                    if iqr_tr > 1e-12:
                        ratio = iqr_sv / iqr_tr
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
                # Estimamos o piso com splits aleatórios do TREINO (BUG 49:
                # do tamanho da inferência) e tomamos o percentil 95 — não há
                # modelo envolvido, só a distribuição, então o nulo é legítimo.
                piso = 0.0
                if p is not None and len(af) >= 20 and not binaria:
                    piso = _piso_ruido(af, len(bf), psi)
                row['psi_piso_ruido'] = round(piso, 4)
                if p is not None and p <= piso:
                    p = None            # dentro do ruído: não é sinal

                if binaria:
                    # BUG 39: numa binária o PSI não se aplica, e a
                    # variância da amostra de inferência pode ser ZERO só
                    # porque não caiu nenhum positivo nela — foi assim que
                    # `RNF` (0,15% de positivos) aparecia como "mudança de
                    # escala" em toda auditoria. Para essas colunas vale só
                    # o teste de proporção, feito acima (BUG 45: vale para
                    # binária comum também).
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
                ct = pd.Series(tr.dropna().to_numpy(dtype=object)).astype(str)
                cs = pd.Series(sv.dropna().to_numpy(dtype=object)).astype(str)
                lv_tr, lv_sv = set(ct.unique()), set(cs.unique())
                novos = sorted(lv_sv - lv_tr)[:6]
                jac = (len(lv_tr & lv_sv) / max(len(lv_tr | lv_sv), 1))
                # BUG 47: categórica só comparava o CONJUNTO de níveis —
                # 80/15/5 → 5/15/80, ou trocar rótulos a↔c, não alertava.
                # PSI sobre as frequências (união dos níveis, com epsilon) e
                # o mesmo piso de ruído do caminho numérico.
                p = psi_categorico(ct, cs)
                piso = 0.0
                if p is not None and len(ct) >= 20:
                    cod_tr, _ = pd.factorize(ct.to_numpy())
                    piso = _piso_ruido(cod_tr, len(cs), _psi_codigos)
                row.update({'kind': 'categorical', 'jaccard': round(jac, 3),
                            'novos_niveis': novos,
                            'psi': None if p is None else round(p, 4),
                            'psi_piso_ruido': round(piso, 4)})
                if novos:
                    feat_alerts.append('NEW_LEVELS')
                if jac < 0.7:
                    feat_alerts.append('LEVEL_SHIFT')
                if p is not None and p > piso:
                    if p > self.psi_alert:
                        feat_alerts.append('PSI_ALERT')
                    elif p > self.psi_warn:
                        feat_alerts.append('PSI_WARN')

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
                   'MEAN_SHIFT', 'NEW_LEVELS', 'NONFINITE')]
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
