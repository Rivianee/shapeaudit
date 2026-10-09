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
        # classificação: shap pode devolver lista [classe0, classe1] ou 3D
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        return np.abs(sv).mean(axis=0)
    except Exception:
        return None


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
    # classificação: usa probabilidade da classe positiva
    def _pred(Z):
        if hasattr(model, 'predict_proba'):
            try:
                p = model.predict_proba(Z)
                if p.ndim == 2 and p.shape[1] == 2:
                    return np.asarray(p[:, 1], dtype=float)
            except Exception:
                pass
        return np.asarray(model.predict(Z), dtype=float)
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
            deltas.append(np.abs(_pred(Xp) - base).mean())
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
