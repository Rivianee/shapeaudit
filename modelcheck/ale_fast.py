"""
ale_fast.py — ALE vetorizado + embedding de modelo

Duas contribuições:

1. ale_curve_fast — ALE com 1 chamada a predict() por bin (em vez de
   uma por amostra), tratamento correto de bins vazios e suporte a
   features categóricas/binárias.

2. model_embedding_ale — representação vetorial do modelo construída a
   partir dos perfis ALE de todas as features. Permite usar os detectores
   geométricos (Fréchet, Mantel, eigenvalue) em QUALQUER arquitetura,
   sem depender de shap.TreeExplainer.
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def feature_kind(X: pd.DataFrame, feature: str) -> str:
    """Classifica a feature: binary, categorical ou continuous."""
    nu = X[feature].nunique(dropna=True)
    if nu <= 1:  return 'constant'
    if nu == 2:  return 'binary'
    if nu <= 10: return 'categorical'
    return 'continuous'


def _predict(model, X: pd.DataFrame) -> np.ndarray:
    """predict robusto: usa proba da classe positiva se classificador."""
    if hasattr(model, 'predict_proba'):
        try:
            p = model.predict_proba(X)
            if p.ndim == 2 and p.shape[1] == 2:
                return np.asarray(p[:, 1], dtype=float)
        except Exception:
            pass
    return np.asarray(model.predict(X), dtype=float)


def ale_curve_fast(model, X: pd.DataFrame, feature: str,
                   n_bins: int = 5,
                   min_bin_n: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    ALE vetorizado. Retorna (centros, ale_acumulado_centrado) ou None.

    Ganho de performance: monta os DataFrames deslocados UMA vez por bin
    e faz duas chamadas a predict por bin, em vez de chamadas por amostra.
    Para features categóricas usa efeito médio por categoria.
    """
    kind = feature_kind(X, feature)
    if kind == 'constant':
        return None

    x = X[feature].values

    # ── Categóricas/binárias: efeito médio por categoria ─────────────
    if kind in ('binary', 'categorical'):
        cats = np.sort(pd.unique(x))
        base = _predict(model, X)
        centers, effects = [], []
        for c in cats:
            Xc = X.copy()
            Xc[feature] = c
            effects.append(float((_predict(model, Xc) - base).mean()))
            centers.append(float(c))
        if len(effects) < 2:
            return None
        arr = np.array(effects, dtype=float)
        return np.array(centers, dtype=float), arr - arr.mean()

    # ── Contínuas: ALE clássico, vetorizado ──────────────────────────
    x = np.asarray(x, dtype=float)
    edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        return None

    # digitize uma única vez para todas as amostras
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    centers, effects = [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < min_bin_n:
            continue
        sub = X[mask]
        Xlo = sub.copy(); Xlo[feature] = edges[b]
        Xhi = sub.copy(); Xhi[feature] = edges[b + 1]
        # 2 chamadas a predict por bin, vetorizadas sobre o bin inteiro
        d = _predict(model, Xhi) - _predict(model, Xlo)
        effects.append(float(d.mean()))
        centers.append(float((edges[b] + edges[b + 1]) / 2))

    if len(effects) < 3:
        return None
    cum = np.cumsum(effects)
    return np.array(centers), cum - cum.mean()


def model_embedding_ale(model, X: pd.DataFrame,
                        feature_names: Optional[List[str]] = None,
                        n_bins: int = 5,
                        n_points: int = 20,
                        normalize: bool = True) -> Optional[np.ndarray]:
    """
    Embedding do modelo a partir dos perfis ALE.

    Cada feature contribui com n_points valores interpolados do seu perfil
    ALE. O vetor final tem dimensão n_features * n_points e representa o
    "comportamento funcional" do modelo — comparável entre arquiteturas.

    normalize=True escala cada perfil pela soma de |valores|, isolando a
    FORMA da magnitude. normalize=False preserva magnitude (útil para
    detectar colapso de importância).
    """
    if feature_names is None:
        feature_names = list(X.columns)
    parts = []
    for f in feature_names:
        out = ale_curve_fast(model, X, f, n_bins=n_bins)
        if out is None:
            parts.append(np.zeros(n_points))
            continue
        centers, vals = out
        if len(centers) < 2:
            parts.append(np.zeros(n_points))
            continue
        xi = np.linspace(centers[0], centers[-1], n_points)
        yi = np.interp(xi, centers, vals)
        if normalize:
            s = np.abs(yi).sum()
            yi = yi / s if s > 1e-12 else yi
        parts.append(yi)
    if not parts:
        return None
    return np.concatenate(parts)


def sample_embeddings_ale(model, X: pd.DataFrame,
                          feature_names: Optional[List[str]] = None,
                          n_replicas: int = 30,
                          frac: float = 0.7,
                          n_bins: int = 5,
                          n_points: int = 8,
                          seed: int = 0) -> Optional[np.ndarray]:
    """
    Matriz de embeddings (n_replicas × dim) por subamostragem.

    Substitui a matriz de SHAP values como "nuvem de pontos" que representa
    o modelo, permitindo Fréchet / Mantel / eigenvalue drift sem SHAP.
    """
    if feature_names is None:
        feature_names = list(X.columns)
    rng = np.random.RandomState(seed)
    n = max(int(len(X) * frac), 30)
    rows = []
    for r in range(n_replicas):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        Xs = X.iloc[idx].reset_index(drop=True)
        emb = model_embedding_ale(model, Xs, feature_names,
                                  n_bins=n_bins, n_points=n_points,
                                  normalize=False)
        if emb is not None:
            rows.append(emb)
    if len(rows) < 3:
        return None
    return np.vstack(rows)
