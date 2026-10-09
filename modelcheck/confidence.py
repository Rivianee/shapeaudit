"""
confidence.py — Intervalos de confiança para scores do ModelGuard

Motivação (Revisão Crítica, lacuna A):
  Nenhum score tinha IC. Não era possível afirmar que score 58.2 e 71.8
  são estatisticamente distintos. Sem isso, o benchmark não é publicável.

Método: bootstrap não-paramétrico sobre X_eval (reamostragem com
reposição), recalculando o score em cada réplica. Retorna média, mediana,
IC percentil e desvio.

CALIBRAÇÃO MEDIDA (v1.35) — leia antes de usar o IC em decisão:

Cobertura empírica com nominal de 95%, contra o score da população
(5.000 amostras), 18 repetições por condição, n_eval=150:

    par divergente:     cobertura 100%   largura média 23.1 pts
    par de controle:    cobertura 100%   largura média 44.5 pts

O IC é CONSERVADOR (sobre-cobertura), não anticonservador. Não há risco de
falsa precisão — o risco é o oposto: intervalos largos demais, com pouco
poder de discriminação.

Causa: o Health Score é DISCRETO (múltiplos de ~5 e ~15 pontos, conforme a
contagem de alertas). O bootstrap percentil sobre uma estatística discreta
produz intervalos que saltam de degrau em degrau, sempre englobando o valor
verdadeiro. Um IC de 44.5 pontos numa escala de 0-100 é quase metade da
escala.

CONSEQUÊNCIA PRÁTICA: use o IC para afirmar que dois cenários NÃO são
distinguíveis (a sobreposição é conclusiva). Não o use para afirmar que
são distinguíveis com 95% de confiança — a sobre-cobertura torna o teste
excessivamente permissivo nessa direção. Para comparação, prefira
`compare_scores_ci`, que usa o teste bootstrap da diferença (`boot_p`)
em vez da sobreposição de intervalos.

Uso:
    from modelcheck.confidence import score_with_ci, compare_scores_ci
    r = score_with_ci(audit, model_v1, model_v2, X_eval, n_boot=50)
    print(r['summary'])
"""
import numpy as np
import pandas as pd
from typing import Callable, Optional, List


def score_with_ci(score_fn: Callable,
                  model_v1, model_v2,
                  X_eval: pd.DataFrame,
                  y_eval: Optional[pd.Series] = None,
                  feature_names: Optional[List[str]] = None,
                  n_boot: int = 50,
                  alpha: float = 0.05,
                  score_key: str = 'health_score',
                  seed: int = 0) -> dict:
    """
    Calcula IC bootstrap para qualquer função de score.

    score_fn deve aceitar (m1, m2, X, y, feature_names) ou (m1, m2, X)
    e retornar dict contendo score_key.
    """
    if feature_names is None:
        feature_names = list(X_eval.columns)

    rng = np.random.RandomState(seed)
    scores, n_fail = [], 0

    for b in range(n_boot):
        idx = rng.choice(len(X_eval), len(X_eval), replace=True)
        Xb = X_eval.iloc[idx].reset_index(drop=True)
        yb = (y_eval.iloc[idx].reset_index(drop=True)
              if y_eval is not None else None)
        try:
            try:
                out = score_fn(model_v1, model_v2, Xb, yb, feature_names)
            except TypeError:
                out = score_fn(model_v1, model_v2, Xb)
            s = out.get(score_key)
            scores.append(np.nan if s is None else float(s))
        except Exception:
            n_fail += 1
            scores.append(np.nan)

    arr = np.array(scores, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {'error': 'todas as réplicas falharam', 'n_fail': n_fail}

    lo = float(np.percentile(valid, 100 * alpha / 2))
    hi = float(np.percentile(valid, 100 * (1 - alpha / 2)))
    mean = float(valid.mean())

    return {
        'mean':       round(mean, 2),
        'median':     round(float(np.median(valid)), 2),
        'std':        round(float(valid.std()), 2),
        'ci_low':     round(lo, 2),
        'ci_high':    round(hi, 2),
        'ci_width':   round(hi - lo, 2),
        'n_boot':     n_boot,
        'n_valid':    int(len(valid)),
        'n_fail':     n_fail,
        'alpha':      alpha,
        'scores':     valid.tolist(),
        'summary':    (f"{mean:.1f} "
                       f"[IC{int((1-alpha)*100)}: {lo:.1f}–{hi:.1f}]"),
    }


def compare_scores_ci(ci_a: dict, ci_b: dict,
                      label_a: str = "A", label_b: str = "B") -> dict:
    """
    Testa se dois scores com IC são estatisticamente distintos.

    Dois critérios:
      overlap      — ICs se sobrepõem? (critério conservador, informal)
      boot_p       — proporção de réplicas em que a >= b (teste bootstrap)
    """
    a = np.array(ci_a.get('scores', []), dtype=float)
    b = np.array(ci_b.get('scores', []), dtype=float)
    overlap = not (ci_a['ci_high'] < ci_b['ci_low'] or
                   ci_b['ci_high'] < ci_a['ci_low'])

    boot_p = None
    if len(a) > 0 and len(b) > 0:
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        m = float(diff.mean())
        # BUG CORRIGIDO (v1.35): a versão anterior usava
        #     p = 2 * min(frac(diff>=0), 1-frac(diff>=0))
        # que devolve p=0 quando TODAS as diferenças são zero — declarando
        # "altamente significativo" exatamente no caso de nenhuma diferença.
        # Pego pelo teste de dois controles idênticos.
        if np.allclose(diff, 0.0):
            boot_p = 1.0
        else:
            # proporção de réplicas com sinal oposto à diferença média,
            # contando empates como metade; correção de continuidade (+1)
            oposto = float((np.sign(diff) != np.sign(m)).sum()
                           - 0.5 * float((diff == 0).sum()))
            oposto = max(oposto, 0.0)
            boot_p = round(min(1.0, 2 * (oposto + 1) / (n + 1)), 4)

    return {
        'label_a':      label_a,
        'label_b':      label_b,
        'mean_a':       ci_a['mean'],
        'mean_b':       ci_b['mean'],
        'delta':        round(ci_a['mean'] - ci_b['mean'], 2),
        'ci_overlap':   overlap,
        'boot_p':       boot_p,
        'distinct':     bool((not overlap) or
                             (boot_p is not None and boot_p < 0.05)),
        'summary':      (f"{label_a}={ci_a['summary']} vs "
                         f"{label_b}={ci_b['summary']} → "
                         f"{'DISTINTOS' if ((not overlap) or (boot_p is not None and boot_p < 0.05)) else 'NÃO distinguíveis'}"
                         + (f" (p={boot_p})" if boot_p is not None else "")),
    }
