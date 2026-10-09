"""
shapedist.py — Distância graduada entre formas

Motivação (Revisão Crítica, lacunas F e G):
  O SDD tratava divergência como binária. `u → u_invertido` (oposição real)
  pesava igual a `u → irregular` (perda de estrutura). Perde informação e
  o threshold 0.30 continuava sendo magic number.

Duas contribuições:

  1. SHAPE_COST — matriz de custo entre as 5 classes de forma,
     justificada semanticamente. Normalizada para [0, 1].

  2. profile_distance — distância CONTÍNUA entre dois perfis binados,
     independente da classificação em classes. Combina:
       - divergência de forma (perfis normalizados por amplitude)
       - divergência de magnitude (razão de amplitudes)
     Isso resolve o caso `alcohol neutralizado`, onde a forma se mantém
     mas a magnitude colapsa — invisível para a distância de forma pura.
"""
import numpy as np
from typing import Optional

SHAPES = ['crescente', 'decrescente', 'u', 'u_invertido', 'irregular']

# Custo semântico entre formas (0 = idêntico, 4 = oposição total)
#              cresc  decr  u   u_inv  irreg
_COST = np.array([
    [0,     4,    3,    3,     2],   # crescente
    [4,     0,    3,    3,     2],   # decrescente
    [3,     3,    0,    1,     1],   # u
    [3,     3,    1,    0,     1],   # u_invertido
    [2,     2,    1,    1,     0],   # irregular
], dtype=float)

SHAPE_COST = {a: {b: _COST[i, j] for j, b in enumerate(SHAPES)}
              for i, a in enumerate(SHAPES)}


def shape_cost(shape_a: str, shape_b: str,
               normalize: bool = True) -> Optional[float]:
    """
    Custo entre duas classes de forma. None se alguma for desconhecida.
    normalize=True devolve em [0, 1] (dividido por 4).
    """
    if shape_a not in SHAPE_COST or shape_b not in SHAPE_COST:
        return None
    c = SHAPE_COST[shape_a][shape_b]
    return float(c / 4.0) if normalize else float(c)


def severity(shape_a: str, shape_b: str) -> str:
    """Rótulo interpretável da severidade da mudança de forma."""
    c = shape_cost(shape_a, shape_b, normalize=False)
    if c is None:
        return 'UNKNOWN'
    if c == 0:  return 'IDENTICAL'
    if c == 1:  return 'MINOR'        # u <-> u_invertido, ou <-> irregular
    if c == 2:  return 'STRUCTURE_LOSS'
    if c == 3:  return 'MAJOR'
    return 'OPPOSITE'                  # crescente <-> decrescente


def _binned(x: np.ndarray, v: np.ndarray, n_bins: int = 8):
    """Perfil binado quantílico: retorna (centros, médias)."""
    x = np.asarray(x, dtype=float); v = np.asarray(v, dtype=float)
    edges = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        return None, None
    cs, ms = [], []
    for i in range(len(edges) - 1):
        m = (x >= edges[i]) & (x <= edges[i + 1])
        if m.sum() < 2:
            continue
        cs.append((edges[i] + edges[i + 1]) / 2)
        ms.append(v[m].mean())
    if len(ms) < 3:
        return None, None
    return np.array(cs), np.array(ms)


def profile_distance(x: np.ndarray,
                     v1: np.ndarray, v2: np.ndarray,
                     n_bins: int = 8,
                     w_shape: float = 0.6,
                     w_magnitude: float = 0.4) -> Optional[dict]:
    """
    Distância contínua entre dois perfis de atribuição na MESMA feature.

    Decompõe em duas componentes ortogonais:
      shape_div     — divergência de FORMA (perfis centrados e escalados
                      pela própria amplitude; ignora magnitude)
      magnitude_div — divergência de MAGNITUDE (razão de amplitudes)

    Ambas em [0, 1]. `distance` é a média ponderada.
    Vantagem sobre o threshold 0.30: escala interpretável e contínua.
    """
    _, p1 = _binned(x, v1, n_bins)
    _, p2 = _binned(x, v2, n_bins)
    if p1 is None or p2 is None or len(p1) != len(p2):
        return None

    a1 = float(np.ptp(p1)); a2 = float(np.ptp(p2))

    # componente de FORMA: normaliza cada perfil por sua amplitude
    if a1 < 1e-12 or a2 < 1e-12:
        shape_div = 0.0 if (a1 < 1e-12 and a2 < 1e-12) else 1.0
    else:
        n1 = (p1 - p1.mean()) / a1
        n2 = (p2 - p2.mean()) / a2
        # L1 média normalizada: perfis normalizados vivem em ~[-1, 1]
        shape_div = float(np.clip(np.abs(n1 - n2).mean(), 0, 1))

    # componente de MAGNITUDE: 0 se amplitudes iguais, 1 se uma colapsou
    denom = max(a1, a2)
    magnitude_div = 0.0 if denom < 1e-12 else float(abs(a1 - a2) / denom)

    dist = float(w_shape * shape_div + w_magnitude * magnitude_div)
    return {
        'distance':      round(dist, 4),
        'shape_div':     round(shape_div, 4),
        'magnitude_div': round(magnitude_div, 4),
        'amplitude_v1':  round(a1, 6),
        'amplitude_v2':  round(a2, 6),
        'n_bins_used':   int(len(p1)),
        'interpretation': ('estável'          if dist < 0.15 else
                           'divergência leve' if dist < 0.35 else
                           'divergência moderada' if dist < 0.55 else
                           'divergência severa'),
    }
