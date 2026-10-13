"""
importance.py — ImportanceCollapseDetector

Detecta o tipo de falha que o SDD NÃO detecta:
uma feature que TINHA peso no modelo v1 e PERDEU peso no v2
(ou ganhou peso do nada), sem que a "forma" mude de classe.

Motivação empírica (v1.18):
  RF vs RF_alcohol_neutralizado → SDD detectou 0 divergências de forma,
  mas 'alcohol' foi literalmente zerada no treino de v2.
  A forma continuou classificável, só a MAGNITUDE colapsou.

Três alertas:
  IMPORTANCE_COLLAPSE  — feature perdeu >collapse_frac da importância
  IMPORTANCE_EMERGENCE — feature ganhou importância vindo de ~0
  RANK_SHUFFLE         — ranking global de importâncias se reordenou

Estatística: bootstrap sobre amostras + IC percentil, para não
alertar por variância de seed.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats


def _shap_importance(model, X: pd.DataFrame,
                     n_sample: int = 200,
                     seed: int = 42) -> Optional[np.ndarray]:
    """Importância = média de |SHAP| por feature. None se indisponível."""
    try:
        import shap
        Xs = X.sample(min(n_sample, len(X)), random_state=seed)
        sv = shap.TreeExplainer(model).shap_values(Xs)
        # BUG 68 (v1.0.4): multiclasse usava só a classe 1; uma variável que
        # separa as classes 0 e 2 ficava invisível. Agora soma as classes
        # (binário continua igual: as duas fatias são espelho).
        from modelcheck.frechet import importancia_media_abs
        return importancia_media_abs(sv)
    except Exception:
        return None


def _arvores_independentes(model):
    """Árvores de um ensemble por bagging (RandomForest/ExtraTrees), ou None."""
    est = getattr(model, 'estimators_', None)
    nome = type(model).__name__
    if est is None or not isinstance(est, list) or len(est) < 5:
        return None
    if not any(k in nome for k in ('RandomForest', 'ExtraTrees')):
        return None       # boosting: árvores sequenciais, não são réplicas
    return est


def _shap_por_arvore(arvores, Xs):
    """Matriz (T, n, p[, K]) com a atribuição de cada árvore em Xs."""
    import shap
    from modelcheck.frechet import como_array_shap
    Xv = np.asarray(Xs, dtype=float)
    return np.stack([como_array_shap(shap.TreeExplainer(t).shap_values(Xv))
                     .astype(np.float32) for t in arvores])


def _imp_lote(M, K):
    """Importância média |φ| para um lote (B, n, p[, K]) -> (B, p)."""
    if M.ndim == 4:
        M = M[..., 1] if K == 2 else M
        return np.abs(M).mean(axis=(1, 3)) if M.ndim == 4 else np.abs(M).mean(axis=1)
    return np.abs(M).mean(axis=1)


def colapso_confirmado_por_arvores(model_v1, model_v2, X, nomes,
                                   n_perm: int = 300, n_sample: int = 150,
                                   seed: int = 0):
    """
    BUG 81 (v1.0.4): BLOQUEADO falso em alta dimensão.

    Com 200 variáveis e uma floresta de 100 árvores, metade dos retreinos
    SAUDÁVEIS (só a semente muda) saía BLOQUEADO, com 6 a 9 variáveis de
    ruído acusadas. Variável de ruído ganha peso minúsculo que oscila muito
    entre sementes: 0,8% -> 0,2% parece colapso. O piso adaptativo do bug 37
    (1/p) deixa essas variáveis entrarem, e o bootstrap de linhas do detector
    mede só ruído de amostragem — não o ruído do retreino, que é o que conta.

    Numa floresta por bagging cada árvore é uma réplica independente do
    procedimento de treino. Se a mudança fosse só acaso, as árvores das duas
    versões seriam intercambiáveis. Teste de permutação: junta as árvores,
    sorteia divisões em dois grupos do mesmo tamanho das florestas, e mede a
    razão de peso em cada divisão. Colapso real (v2 nunca usa a coluna, v1
    sempre usa) é extremo nessa distribuição; ruído usado em 2 de 40 árvores
    não é.

    (Uma primeira versão reamostrava as árvores de cada versão
    separadamente. Falhava: variável de ruído ausente de todas as árvores da
    v2 continuava ausente em toda reamostragem e parecia colapso certo.)

    Estatística: log(peso_v1) - log(peso_v2), padronizada pela média e pelo
    desvio da distribuição nula (escore z). Um p-valor por sorteio não serve:
    com 300 sorteios o menor p possível é 0,003, e depois da correção para
    200 variáveis nada poderia ser confirmado — nem o colapso real.

    Devolve {variavel: z} ou None quando não se aplica (modelo que não é
    floresta por bagging).
    """
    a1, a2 = _arvores_independentes(model_v1), _arvores_independentes(model_v2)
    if a1 is None or a2 is None or not nomes:
        return None
    Xs = X.sample(min(n_sample, len(X)), random_state=seed)
    cols = list(X.columns)
    idx = [cols.index(f) for f in nomes]
    P = np.concatenate([_shap_por_arvore(a1, Xs), _shap_por_arvore(a2, Xs)])
    T1, T = len(a1), len(P)
    K = P.shape[3] if P.ndim == 4 else 0
    forma = P.shape[1:]
    flat = P.reshape(T, -1)

    def razao(W1, W2):
        # log(peso_v1) - log(peso_v2); eps relativo ao peso total
        I1 = _imp_lote((W1 @ flat).reshape((-1,) + forma), K)
        I2 = _imp_lote((W2 @ flat).reshape((-1,) + forma), K)
        eps = 1e-4 * (I1.sum(axis=1, keepdims=True) + I2.sum(axis=1, keepdims=True)) / 2
        return np.log(I1[:, idx] + eps) - np.log(I2[:, idx] + eps)

    w1 = np.zeros((1, T), np.float32); w1[0, :T1] = 1.0 / T1
    w2 = np.zeros((1, T), np.float32); w2[0, T1:] = 1.0 / (T - T1)
    obs = razao(w1, w2)[0]

    rng = np.random.RandomState(seed)
    nulo = []
    for _ in range(0, n_perm, 100):
        W1 = np.zeros((100, T), np.float32); W2 = np.zeros((100, T), np.float32)
        for k in range(100):
            perm = rng.permutation(T)
            W1[k, perm[:T1]] = 1.0 / T1
            W2[k, perm[T1:]] = 1.0 / (T - T1)
        nulo.append(razao(W1, W2))
    nulo = np.concatenate(nulo)[:n_perm]
    z = (obs - nulo.mean(axis=0)) / (nulo.std(axis=0) + 1e-9)
    return {f: float(v) for f, v in zip(nomes, z)}


def _perturbation_importance(model, X: pd.DataFrame,
                             feature_names: List[str],
                             n_repeat: int = 5,
                             seed: int = 42,
                             subset: Optional[List[str]] = None,
                             n_denom: Optional[int] = None) -> np.ndarray:
    """
    Fallback model-agnostic: importância por permutação da feature.
    Mede quanto a predição muda ao embaralhar a coluna.
    Não precisa de y — mede sensibilidade, não perda de acurácia.
    """
    rng = np.random.RandomState(seed)
    # classificação: usa probabilidade da classe positiva.
    # BUG 70 (v1.0.4): com 3+ classes caía em predict() e media |rótulo
    # permutado - rótulo|: trocar 0->2 contava o dobro de 0->1, e renomear
    # as classes mudava a importância. Agora usa a distância de variação
    # total entre os vetores de probabilidade, que não depende do rótulo.
    def _pred(Z):
        if hasattr(model, 'predict_proba'):
            try:
                p = np.asarray(model.predict_proba(Z), dtype=float)
                if p.ndim == 2 and p.shape[1] == 2:
                    return p[:, 1]
                if p.ndim == 2 and p.shape[1] > 2:
                    return p
            except Exception:
                pass    # noqa: lacuna-nao-aplicavel — cai no predict()
        return np.asarray(model.predict(Z), dtype=float)

    def _dist(a, b):
        d = np.abs(a - b)
        return 0.5 * d.sum(axis=1) if d.ndim == 2 else d
    base = _pred(X)
    imps = np.zeros(len(feature_names))

    # ── Modo lazy: permuta só as monitoradas + uma amostra das demais ──
    # Necessário porque a importância é RELATIVA (precisa de denominador).
    # Custo O(len(subset) + n_denom) em vez de O(n_features).
    if subset:
        rs = np.random.RandomState(seed)
        # n_denom adaptativo: com 200 features, 12 amostras dão um denominador
        # ruidoso e geram falso positivo (medido: 2/3 em P=200).
        if n_denom is None:
            n_denom = max(15, int(0.25 * len(feature_names)))
        others = [f for f in feature_names if f not in set(subset)]
        sample = list(rs.choice(others, min(n_denom, len(others)),
                                replace=False)) if others else []
        active = list(subset) + sample
        idx_active = [feature_names.index(f) for f in active
                      if f in feature_names]
    else:
        active = feature_names
        idx_active = list(range(len(feature_names)))

    for j in idx_active:
        f = feature_names[j]
        deltas = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            deltas.append(_dist(_pred(Xp), base).mean())
        imps[j] = float(np.mean(deltas))

    # No modo lazy, escala o denominador para estimar a soma total
    if subset and len(active) < len(feature_names):
        n_sampled = len(active) - len(subset)
        if n_sampled > 0:
            others_sum = sum(imps[feature_names.index(f)]
                             for f in active if f not in set(subset))
            n_others = len(feature_names) - len(subset)
            est = others_sum / n_sampled * n_others
            # distribui a estimativa nas não medidas para o denominador
            not_measured = [i for i in range(len(feature_names))
                            if i not in idx_active]
            if not_measured:
                per = (est - others_sum) / len(not_measured)
                for i in not_measured:
                    imps[i] = max(per, 0.0)
    return imps


class ImportanceCollapseDetector:
    """
    Detecta colapso, emergência e reordenação de importância de features.

    Complementa o SDD:
      SDD  → "a FORMA da relação mudou" (crescente → u)
      ICD  → "a feature deixou de importar" (peso 0.15 → 0.00)

    Uso:
        det = ImportanceCollapseDetector()
        res = det.compare(model_v1, model_v2, X_eval, feature_names)
        print(det.report(res))
    """

    def __init__(self,
                 collapse_frac:   float = 0.60,
                 emergence_frac:  float = 2.50,
                 min_rel_weight:  float = 0.02,
                 n_boot:          int   = 8,
                 n_sample:        int   = 200,
                 rank_alpha:      float = 0.05,
                 per_feature_threshold: Optional[dict] = None):
        self.per_feature_threshold = per_feature_threshold or {}
        self.collapse_frac  = collapse_frac
        self.emergence_frac = emergence_frac
        self.min_rel_weight = min_rel_weight
        self.n_boot         = n_boot
        self.n_sample       = n_sample
        self.rank_alpha     = rank_alpha

    def _importances(self, model, X, feature_names, seed, subset=None):
        imp = _shap_importance(model, X, self.n_sample, seed)
        if imp is not None:
            # BUG 69 (v1.0.4): o SHAP vem na ordem das colunas de X; o resto
            # do detector indexa pela posição em feature_names. Alinha pelo
            # nome para nomes em outra ordem (ou subconjunto) não trocarem
            # a culpa de variável.
            cols = list(X.columns)
            if len(imp) == len(cols) and list(feature_names) != cols:
                pos = {c: i for i, c in enumerate(cols)}
                imp = np.array([imp[pos[f]] for f in feature_names], float)
            return imp, 'SHAP'
        return (_perturbation_importance(model, X, feature_names, seed=seed,
                                         subset=subset),
                'PERMUTATION_LAZY' if subset else 'PERMUTATION')

    def calibrate(self, model_factory, data: pd.DataFrame,
                  target: str, feature_names: List[str],
                  window: int, n_pairs: int = 8,
                  percentile: float = 5.0) -> dict:
        """
        Calibra o limiar de colapso por feature usando retreinos SAUDÁVEIS.

        Necessário porque a variação natural de importância difere por
        feature. Medido no ENA (8 retreinos disjuntos de 365 dias):
            ENA_MLT  p5=0.70   ENA_ARM p5=0.55
            mes      p5=0.63   doy     p5=0.64
            EAR      p5=0.31  ← abaixo do limiar padrão 0.40 → falso positivo

        model_factory(X, y) deve devolver um modelo treinado.
        Retorna dict feature -> limiar (razão abaixo da qual é colapso).
        """
        ratios = {f: [] for f in feature_names}
        n = len(data)
        step = max((n - 2 * window) // max(n_pairs, 1), 1)
        ev = data.sample(min(300, n), random_state=0)
        for k in range(n_pairs):
            s = window + k * step
            if s + window > n:
                break
            A = data.iloc[s - window:s]
            B = data.iloc[s:s + window]
            if len(A) < window or len(B) < window:
                break
            m1 = model_factory(A[feature_names], A[target])
            m2 = model_factory(B[feature_names], B[target])
            res = self.compare(m1, m2, ev[feature_names], feature_names)
            for r in res['features']:
                ratios[r['feature']].append(r['ratio'])
        thresholds, stats = {}, {}
        n_calibradas = 0
        for f, v in ratios.items():
            if len(v) < 3:
                thresholds[f] = 1 - self.collapse_frac
                stats[f] = {'n': len(v), 'calibrada': False}
                continue
            n_calibradas += 1
            arr = np.array(v, dtype=float)
            p = float(np.percentile(arr, percentile))
            # limiar = metade do percentil observado, nunca acima do padrão
            thresholds[f] = round(min(1 - self.collapse_frac, p * 0.5), 4)
            stats[f] = {'n': len(arr), 'mean': round(float(arr.mean()), 3),
                        'min': round(float(arr.min()), 3),
                        f'p{int(percentile)}': round(p, 3)}
        # BUG 18 (v1.48): quando NENHUMA feature reunia pares suficientes, o
        # método devolvia os thresholds PADRÃO sem sinalizar — o usuário
        # recebia um dict que parecia calibração válida e seguia com o limiar
        # fixo, achando que tinha calibrado. Só o campo `n: 0` denunciava,
        # enterrado em `stats`.
        ok = n_calibradas > 0
        if not ok:
            self.per_feature_threshold = {}
            return {'ok': False,
                    'reason': ('nenhuma feature reuniu pares suficientes '
                               '(mínimo 3). Verifique se `window` cabe nos '
                               'dados: são necessários pelo menos '
                               '2*window + n_pairs registros.'),
                    'n_calibradas': 0,
                    'n_features': len(ratios),
                    'thresholds': thresholds, 'stats': stats,
                    'default': 1 - self.collapse_frac}
        self.per_feature_threshold = thresholds
        return {'ok': True,
                'n_calibradas': n_calibradas,
                'n_features': len(ratios),
                'thresholds': thresholds, 'stats': stats,
                'default': 1 - self.collapse_frac}

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None,
                monitored: Optional[List[str]] = None) -> dict:
        """
        monitored: lista de features cuja integridade é crítica (ex: as que
        vêm de um join frágil ou de fonte externa). Quando informada, o
        campo `monitored_alert` só é True se o colapso ocorrer NELAS.

        Medido em 108 pares reais (ENA, 3 subsistemas, 4 níveis de
        sobreposição de janela de treino):
            regra 'n_col > 0'                → recall 97%, especificidade 69%
            regra 'colapso em monitored'     → recall 97%, especificidade 100%
        Controles nunca acusaram a feature monitorada em nenhum nível de
        sobreposição; retreino temporal legítimo desloca a importância de
        OUTRAS features, não da que foi corrompida.
        """
        from modelcheck._validate import validate_inputs
        X_eval, feature_names = validate_inputs(
            model_v1, model_v2, X_eval, feature_names,
            caller='ImportanceCollapseDetector')

        if feature_names is None:
            feature_names = list(X_eval.columns)

        # bootstrap de importâncias para separar sinal de variância
        I1, I2, method = [], [], None
        for b in range(self.n_boot):
            idx = np.random.RandomState(b).choice(
                len(X_eval), len(X_eval), replace=True)
            Xb = X_eval.iloc[idx].reset_index(drop=True)
            i1, m1 = self._importances(model_v1, Xb, feature_names, seed=b,
                                       subset=monitored)
            i2, m2 = self._importances(model_v2, Xb, feature_names, seed=b,
                                       subset=monitored)
            # BUG 33 (v0.9.1): o método do v2 era descartado. Quando o SHAP
            # funcionava para um modelo e falhava para o outro (ex.: shap
            # 0.48 não lê o base_score do XGBoost 3.x), a comparação
            # misturava importância SHAP de um com importância por
            # permutação do outro — escalas diferentes. Resultado medido:
            # RF vs XGBoost treinados nos MESMOS dados davam BLOQUEADO,
            # apontando uma variável que nenhum dos dois tinha perdido.
            #
            # Correção: se os métodos divergem, recalcula os DOIS pelo
            # método comum (permutação), que funciona para qualquer modelo.
            if m1 != m2:
                i1 = _perturbation_importance(model_v1, Xb, feature_names,
                                              seed=b, subset=monitored)
                i2 = _perturbation_importance(model_v2, Xb, feature_names,
                                              seed=b, subset=monitored)
                m1 = 'PERMUTATION (unificado: SHAP falhou em um dos modelos)'
                self._metodo_unificado = True
            method = method or m1
            I1.append(i1); I2.append(i2)
        I1 = np.array(I1); I2 = np.array(I2)

        # normaliza para peso relativo (soma 1) em cada réplica
        R1 = I1 / (I1.sum(axis=1, keepdims=True) + 1e-12)
        R2 = I2 / (I2.sum(axis=1, keepdims=True) + 1e-12)

        results, alerts = [], []
        for j, f in enumerate(feature_names):
            a, b = R1[:, j], R2[:, j]
            ratios = b / (a + 1e-12)
            # RAZÃO ABSOLUTA (v1.38): a razão RELATIVA é confundida pelo
            # denominador em alta dimensionalidade. Medido com P=200 e sinal
            # concentrado em f0/f1: ao zerar f0, o coeficiente de f1 quase não
            # muda (+1.995 -> +1.947) mas o peso relativo cai de 0.32 para
            # 0.11 (razão 0.38) porque a soma se redistribui entre 198
            # features de ruído. Isso marcava f1 como colapso (2/3 dos casos).
            # A razão absoluta não sofre desse artefato.
            aa, bb = I1[:, j], I2[:, j]
            ratios_abs = bb / (aa + 1e-12)
            ratio_abs_med = float(np.median(ratios_abs))
            ratio_med = float(np.median(ratios))
            lo, hi = np.percentile(ratios, [5, 95])

            w1, w2 = float(a.mean()), float(b.mean())
            # BUG 37 (v0.19.0): o piso de relevância era fixo em 2% do total.
            # Com muitas variáveis isso desliga a camada inteira: no SECOM
            # (475 variáveis) a MAIS importante pesa 0.3%, e uma falha total
            # (0.3% -> 0.0%) passava sem exame. O piso não pode exigir mais que
            # a fatia uniforme (1/p): uma variável que carrega ao menos a média
            # não é desprezível.
            piso = min(self.min_rel_weight, 1.0 / max(len(feature_names), 1))
            was_relevant = w1 >= piso
            is_relevant  = w2 >= piso

            thr = getattr(self, 'per_feature_threshold', {}).get(
                f, 1 - self.collapse_frac)
            # Exige colapso nas DUAS escalas: relativa (o peso caiu no
            # conjunto) E absoluta (a feature de fato perdeu efeito).
            collapse = (was_relevant and
                        ratio_med < thr and
                        ratio_abs_med < thr and
                        hi < 1.0)               # IC não cruza "sem mudança"
            emergence = (not was_relevant and is_relevant and
                         ratio_med > self.emergence_frac and lo > 1.0)

            alert = ('IMPORTANCE_COLLAPSE'  if collapse  else
                     'IMPORTANCE_EMERGENCE' if emergence else None)
            if alert:
                alerts.append(alert)

            results.append({
                'feature':      f,
                'weight_v1':    round(w1, 5),
                'weight_v2':    round(w2, 5),
                'ratio':        round(ratio_med, 3),
                'ratio_abs':    round(ratio_abs_med, 3),
                'ratio_ci':     (round(float(lo), 3), round(float(hi), 3)),
                'alert':        alert,
            })

        # BUG 81: confirma os colapsos contra o ruído do próprio retreino
        suspeitos = [r['feature'] for r in results
                     if r['alert'] == 'IMPORTANCE_COLLAPSE']
        if suspeitos:
            try:
                zs = colapso_confirmado_por_arvores(model_v1, model_v2,
                                                    X_eval, suspeitos)
            except Exception:
                zs = None  # noqa: lacuna-nao-aplicavel — fica o critério antigo
            if zs is None:
                # BUG 82 (v1.0.4): fora de floresta (boosting estocástico,
                # p=200) o mesmo falso BLOQUEADO aparecia em 3 de 3 retreinos
                # saudáveis, e as árvores do boosting não são réplicas para o
                # teste de permutação. Critério de massa de ruído: quando o
                # peso está concentrado em poucas variáveis (número efetivo
                # < p/4), a cauda é ruído, e uma variável só "colapsa" se
                # antes pesava claramente acima dela. Medido (GBM estocástico,
                # p=200 e 500, 700 variáveis de ruído): peso/máximo da cauda
                # chega a 2,3 no ruído e começa em 4,5 numa variável real
                # fraca. Teto: 3x o máximo da cauda. Em modelo denso não há
                # cauda de ruído e nada muda.
                w = R1.mean(0)
                p_ = len(w)
                n_eff = 1.0 / float((w ** 2).sum() + 1e-12)
                if p_ >= 20 and n_eff < 0.25 * p_:
                    k = int(np.ceil(n_eff))
                    cauda = np.sort(w)[:-k] if k < p_ else w
                    teto = 3.0 * float(cauda.max())
                    for r in results:
                        if r['feature'] in suspeitos and r['weight_v1'] < teto:
                            r['alert'] = None
                            r['colapso_nao_confirmado'] = True
                            r['peso_na_massa_de_ruido'] = round(teto, 5)
                            alerts.remove('IMPORTANCE_COLLAPSE')
            if zs:
                # correção pelo número de variáveis TESTADAS (todas acima do
                # piso), não só pelas suspeitas: a suspeita já é uma seleção.
                piso_ = min(self.min_rel_weight, 1.0 / max(len(feature_names), 1))
                m = max(1, sum(1 for r in results if r['weight_v1'] >= piso_))
                # A nula tem cauda pesada (variável de ruído alterna entre
                # "usada por poucas árvores" e "nenhuma"), então a cauda
                # normal subestima o limiar. Medido em 810 variáveis de ruído
                # (p=60-500, 10-100 árvores, retreinos saudáveis): z máximo
                # 4,9. Colapso real: 17 a 42 (forte), >= 6 (fraco) exceto em
                # floresta de 10-30 árvores com 200+ variáveis. Piso 6.
                z_crit = max(6.0, float(scipy_stats.norm.ppf(1 - 0.05 / m)))
                for r in results:
                    f = r['feature']
                    if f not in zs:
                        continue
                    r['z_permutacao_arvores'] = round(zs[f], 2)
                    if zs[f] < z_crit:
                        r['alert'] = None
                        r['colapso_nao_confirmado'] = True
                        alerts.remove('IMPORTANCE_COLLAPSE')

        # reordenação global do ranking
        rho, p_rank = scipy_stats.spearmanr(R1.mean(0), R2.mean(0))
        rank_shuffle = bool(p_rank > self.rank_alpha or rho < 0.7)
        if rank_shuffle:
            alerts.append('RANK_SHUFFLE')

        n_col = sum(1 for r in results if r['alert'] == 'IMPORTANCE_COLLAPSE')
        n_eme = sum(1 for r in results if r['alert'] == 'IMPORTANCE_EMERGENCE')
        score = max(0.0, 100 - n_col * 25 - n_eme * 15
                    - (15 if rank_shuffle else 0))

        monitored_hits = []
        if monitored:
            mset = set(monitored)
            monitored_hits = [r['feature'] for r in results
                              if r['alert'] == 'IMPORTANCE_COLLAPSE'
                              and r['feature'] in mset]

        return {
            'method':          method,
            'monitored':       list(monitored) if monitored else None,
            'monitored_hits':  monitored_hits,
            'monitored_alert': bool(monitored_hits),
            'features':        results,
            'rank_rho':        round(float(rho), 4),
            'rank_p':          round(float(p_rank), 4),
            'rank_shuffle':    rank_shuffle,
            'alerts':          alerts,
            'n_collapse':      n_col,
            'n_emergence':     n_eme,
            'importance_score': round(score, 1),
        }

    def report(self, res: dict) -> str:
        lines = [f"ImportanceCollapseDetector ({res['method']}, "
                 f"n_boot={self.n_boot}):",
                 f"  Score: {res['importance_score']}/100",
                 f"  Ranking ρ={res['rank_rho']:.3f} "
                 f"{'⚠ RANK_SHUFFLE' if res['rank_shuffle'] else '✅ estável'}"]
        flagged = [r for r in res['features'] if r['alert']]
        if not flagged:
            lines.append("  ✅ Nenhum colapso ou emergência de importância.")
        else:
            for r in flagged:
                lines.append(
                    f"  ⚠ {r['alert']}: '{r['feature']}'  "
                    f"peso {r['weight_v1']:.4f} → {r['weight_v2']:.4f}  "
                    f"({r['ratio']:.2f}x, IC90 {r['ratio_ci']})")
        return "\n".join(lines)
