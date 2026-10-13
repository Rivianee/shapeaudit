"""
backends.py — Backends de importância para o detector de colapso

Motivação: o SHAP só é exato e rápido em árvores (TreeSHAP). Fora delas,
o fallback atual é permutação sobre a predição. Este módulo expõe quatro
backends com a MESMA interface, para que o `ImportanceCollapseDetector`
possa trocar de representação e para que a escolha seja decidida por
benchmark, não por preferência.

Backends implementados (todos métodos estabelecidos e verificáveis):

  'shap'         TreeSHAP — média de |SHAP| por feature. Exato em árvores.
  'perm_pred'    permutação medindo mudança na PREDIÇÃO (não usa y).
                 É o fallback atual.
  'perm_loss'    permutação medindo aumento da PERDA (usa y). É a forma
                 clássica de permutation importance (Breiman) e a base
                 conceitual da importância global tipo SAGE.
  'pdp_range'    amplitude da curva de dependência parcial. Barato, mas
                 assume independência entre features.
  'ice_var'      variância entre curvas ICE individuais — capta
                 heterogeneidade de efeito, não só o efeito médio.

Nota sobre o PDP: ele assume independência e pode gerar perfis enganosos
com features correlacionadas. Está aqui para o benchmark, não como padrão.
"""
import numpy as np
import pandas as pd
from typing import List, Optional

BACKENDS = ('shap', 'perm_pred', 'perm_loss', 'pdp_range', 'ice_var',
            'kernel_shap')


def _pred(model, X):
    if hasattr(model, 'predict_proba'):
        try:
            p = model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] == 2:
                return np.asarray(p[:, 1], dtype=float)
            if p.ndim == 2 and p.shape[1] > 2:
                # BUG 70 (v1.0.4): com 3+ classes caía em predict() e tratava
                # o rótulo como número (0->2 valia o dobro de 0->1). Não há
                # curva única honesta aqui: falha explícita, e a camada vira
                # lacuna declarada em vez de número errado.
                raise ValueError('curva de efeito para classificador com '
                                 f'{p.shape[1]} classes não é suportada')
        except ValueError:
            raise
        except Exception:
            pass    # noqa: lacuna-nao-aplicavel — sem predict_proba utilizável
    return np.asarray(model.predict(X), dtype=float)


def imp_shap(model, X, feature_names, y=None, seed=0, **kw):
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return np.abs(sv).mean(axis=0)


def imp_kernel_shap(model, X, feature_names, y=None, seed=0,
                    nsamples=100, n_background=50, **kw):
    """
    KernelSHAP — média de |SHAP| por feature, model-agnostic.

    BACKEND OPCIONAL, não padrão. Medido (v1.51) em Ridge e MLP com 6
    features, corrompendo uma:

        detecção do colapso : idêntica a 'perm_pred' e ao ALE
        divergências de forma: idêntica ao ALE (2 no Ridge, 0 no MLP)
        tempo                : 12-13x mais lento que 'perm_pred'
        determinismo         : sim (o shap fixa a semente interna)

    Escalabilidade medida (Ridge, p de 6 a 50): a razão contra 'perm_pred'
    CAI de 8x para 2x, ou seja o custo relativo melhora em alta
    dimensionalidade — mas nunca compensa, porque a resposta é a mesma.

    Use apenas se precisar de valores de Shapley propriamente ditos por
    exigência externa (auditoria que pede SHAP nominalmente). Para detectar
    colapso de importância, 'perm_pred' dá o mesmo resultado mais rápido.
    """
    import shap as _shap
    bg = _shap.sample(X[feature_names], min(n_background, len(X)),
                      random_state=seed)
    expl = _shap.KernelExplainer(model.predict, bg)
    sv = expl.shap_values(X[feature_names], nsamples=nsamples, silent=True)
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
    return np.abs(sv).mean(axis=0)


def imp_perm_pred(model, X, feature_names, y=None, seed=0,
                  n_repeat=5, **kw):
    rng = np.random.RandomState(seed)
    base = _pred(model, X)
    out = np.zeros(len(feature_names))
    for j, f in enumerate(feature_names):
        d = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            d.append(np.abs(_pred(model, Xp) - base).mean())
        out[j] = float(np.mean(d))
    return out


def imp_perm_loss(model, X, feature_names, y=None, seed=0,
                  n_repeat=5, **kw):
    """Permutation importance clássica: aumento do erro ao embaralhar."""
    if y is None:
        return imp_perm_pred(model, X, feature_names, seed=seed,
                             n_repeat=n_repeat)
    rng = np.random.RandomState(seed)
    yv = np.asarray(y, dtype=float)
    base = float(((_pred(model, X) - yv) ** 2).mean())
    out = np.zeros(len(feature_names))
    for j, f in enumerate(feature_names):
        d = []
        for _ in range(n_repeat):
            Xp = X.copy()
            Xp[f] = rng.permutation(Xp[f].values)
            d.append(float(((_pred(model, Xp) - yv) ** 2).mean()) - base)
        out[j] = max(float(np.mean(d)), 0.0)
    return out


def imp_pdp_range(model, X, feature_names, y=None, seed=0,
                  n_grid=8, n_sample=150, **kw):
    """Amplitude da curva PDP. Assume independência entre features."""
    Xs = X.sample(min(n_sample, len(X)), random_state=seed)
    out = np.zeros(len(feature_names))
    for j, f in enumerate(feature_names):
        xv = X[f].to_numpy(dtype=float)
        if np.std(xv) < 1e-12:
            continue
        grid = np.unique(np.percentile(xv, np.linspace(5, 95, n_grid)))
        means = []
        for g in grid:
            Xg = Xs.copy(); Xg[f] = g
            means.append(_pred(model, Xg).mean())
        out[j] = float(np.ptp(means)) if len(means) > 1 else 0.0
    return out


def imp_ice_var(model, X, feature_names, y=None, seed=0,
                n_grid=6, n_sample=60, **kw):
    """
    Variância entre curvas ICE. Captura heterogeneidade: features cujo
    efeito difere entre instâncias pontuam alto mesmo com PDP plano.
    """
    Xs = X.sample(min(n_sample, len(X)), random_state=seed)
    out = np.zeros(len(feature_names))
    for j, f in enumerate(feature_names):
        xv = X[f].to_numpy(dtype=float)
        if np.std(xv) < 1e-12:
            continue
        grid = np.unique(np.percentile(xv, np.linspace(10, 90, n_grid)))
        curves = []
        for g in grid:
            Xg = Xs.copy(); Xg[f] = g
            curves.append(_pred(model, Xg))
        if len(curves) < 2:
            continue
        C = np.vstack(curves)                 # (n_grid, n_amostras)
        amp = C.max(axis=0) - C.min(axis=0)   # amplitude por instância
        out[j] = float(amp.mean())
    return out


_FN = {'shap': imp_shap, 'perm_pred': imp_perm_pred,
       'perm_loss': imp_perm_loss, 'pdp_range': imp_pdp_range,
       'ice_var': imp_ice_var, 'kernel_shap': imp_kernel_shap}


def importance(model, X: pd.DataFrame, feature_names: List[str],
               backend: str = 'shap', y=None, seed: int = 0,
               **kw) -> Optional[np.ndarray]:
    """
    Importância por feature no backend escolhido. `None` se falhar.
    Todos devolvem vetor não-negativo de tamanho len(feature_names).
    """
    if backend not in _FN:
        raise ValueError(f"backend deve ser um de {BACKENDS}")
    try:
        v = _FN[backend](model, X, feature_names, y=y, seed=seed, **kw)
        return np.asarray(v, dtype=float)
    except Exception:
        return None
