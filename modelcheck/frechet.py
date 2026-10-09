"""
frechet.py — detecção de mudança de forma por distância (v0.10.0)

POR QUE ISTO EXISTE
-------------------
Comparamos duas formas de ler os perfis de atribuição entre versões: classificar
cada um em cinco formas, ou medir a distância de Fréchet entre eles. Medido em
duas bancadas independentes (paper-dados/tudo.json):

    bancada 1 (formas grosseiras)  classe+confirmação 94%/9%  | Fréchet 100%/2%
    bancada 2 (formas sutis)       classe+confirmação 30%/4%  | Fréchet 99%/11%

A distância é o detector mais sensível. A classe continua útil para NOMEAR a
mudança ("u_invertido -> u"). Este módulo faz as duas coisas: a distância
decide, a classe explica.

LIMIAR
------
O perfil é normalizado nos dois eixos (posição do bin em [0, 1], atribuição
padronizada), então a distância não depende da escala da variável nem do alvo.
O limiar padrão, 0.33, é o percentil 95 dos controles da bancada 1. Na bancada
2 ele deu 11% de falso positivo, não 5%: transfere só em parte. Calibre no seu
domínio com retreinos saudáveis sempre que puder.

Testamos também autocalibração por reamostragem do conjunto de avaliação: deu
16% a 65% de falso positivo, porque mede só o ruído de amostragem, e a maior
parte da variação real vem do retreino. Por isso não é o padrão.

UMA SÓ IMPLEMENTAÇÃO
--------------------
O orquestrador (Guard) e o audit() chamam esta mesma função. Havia três cópias
do cálculo de forma divergindo entre si (bug 28, classe 4 do catálogo).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

LIMIAR_PADRAO = 0.33
K_PADRAO = 10
# Peso mínimo para a distância avaliar uma variável. Medido em retreinos
# saudáveis (Wine, Concrete, California; 8 pares de sementes cada): acima do
# limiar ficaram 77% das variáveis com peso < 2%, 20% das com 2-5% e 0% das
# com peso >= 5%. O perfil normalizado de uma variável fraca é ruído
# amplificado. Validado depois em outros sete datasets (ver testes).
PESO_MINIMO_PADRAO = 0.05


def perfil(x, s, K: int = K_PADRAO) -> Optional[np.ndarray]:
    """Média da atribuição por bin de quantil de x. None se não houver bins."""
    x = np.asarray(x, dtype=float)
    s = np.asarray(s, dtype=float)
    ok = np.isfinite(x) & np.isfinite(s)
    if ok.sum() < 2 * K:
        return None
    b = pd.qcut(x[ok], K, labels=False, duplicates="drop")
    m = pd.Series(s[ok]).groupby(np.asarray(b)).mean().to_numpy()
    return m if len(m) >= 3 else None


def perfil_com_bordas(x, s, K: int = K_PADRAO):
    """
    Como `perfil`, mas devolve também as bordas dos bins em unidades da
    variável — é o que permite dizer "entre 9,4 e 10,2 de álcool" em vez de
    "no bin 3" (v0.13.0).
    """
    x = np.asarray(x, dtype=float); s = np.asarray(s, dtype=float)
    ok = np.isfinite(x) & np.isfinite(s)
    if ok.sum() < 2 * K:
        return None, None
    xb, sb = x[ok], s[ok]
    b, bordas = pd.qcut(xb, K, labels=False, duplicates="drop", retbins=True)
    m = pd.Series(sb).groupby(np.asarray(b)).mean().to_numpy()
    if len(m) < 3:
        return None, None
    return m, np.asarray(bordas, float)


def _norm(m: np.ndarray) -> np.ndarray:
    sd = float(m.std())
    return (m - m.mean()) / sd if sd > 1e-12 else np.zeros_like(m)


def distancia(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fréchet discreta entre dois perfis, após normalizar os dois eixos.
    Perfis com número de bins diferente são aceitos.
    """
    a, b = _norm(np.asarray(a, float)), _norm(np.asarray(b, float))
    P = np.c_[np.linspace(0, 1, len(a)), a]
    Q = np.c_[np.linspace(0, 1, len(b)), b]
    n, k = len(P), len(Q)
    C = np.empty((n, k))
    for i in range(n):
        for j in range(k):
            d = float(np.linalg.norm(P[i] - Q[j]))
            if i == 0 and j == 0:
                C[i, j] = d
            elif i == 0:
                C[i, j] = max(C[i, j - 1], d)
            elif j == 0:
                C[i, j] = max(C[i - 1, j], d)
            else:
                C[i, j] = max(min(C[i - 1, j], C[i - 1, j - 1], C[i, j - 1]), d)
    return float(C[-1, -1])


def perfis(model, X_eval: pd.DataFrame, feature_names: List[str],
           metodo: str = "treeshap", K: int = K_PADRAO):
    """
    Perfil de efeito por variável, por um de dois geradores (v0.12.2).

    metodo="treeshap": média da atribuição SHAP por bin de quantil.
    metodo="ale":      curva ALE (efeitos locais acumulados), que não exige
                       modelo de árvore.

    Devolve (perfis, pesos): {variavel: array} e {variavel: peso relativo}.
    O peso do ALE é a amplitude da curva normalizada pela soma das
    amplitudes — o análogo da fração de atribuição.
    """
    out, peso = {}, {}
    if metodo == "treeshap":
        import shap
        sv = np.asarray(shap.TreeExplainer(model).shap_values(X_eval), float)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        tot = float(np.abs(sv).mean(axis=0).sum()) + 1e-12
        for i, f in enumerate(feature_names):
            p = perfil(X_eval[f].values, sv[:, i], K)
            if p is not None:
                out[f] = p
                peso[f] = float(np.abs(sv[:, i]).mean()) / tot
        return out, peso
    if metodo != "ale":
        raise ValueError("metodo deve ser 'treeshap' ou 'ale'")
    from modelcheck.ale_fast import ale_curve_fast
    amp = {}
    for f in feature_names:
        try:
            r = ale_curve_fast(model, X_eval, f, n_bins=K)
        except Exception:
            r = None      # noqa: lacuna-nao-aplicavel — variável sem curva
        if r is None:
            continue
        _, curva = r
        if len(curva) < 3:
            continue
        out[f] = np.asarray(curva, float)
        amp[f] = float(np.ptp(curva))
    s = sum(amp.values()) + 1e-12
    peso = {f: a / s for f, a in amp.items()}
    return out, peso


def divergencias_entre(model_v1, model_v2, X_eval: pd.DataFrame,
                       feature_names: List[str], metodo: str = "treeshap",
                       limiar: float = LIMIAR_PADRAO,
                       peso_minimo: float = PESO_MINIMO_PADRAO,
                       K: int = K_PADRAO) -> List[Dict]:
    """
    Compara dois modelos usando o gerador de curva escolhido (v0.12.2).

    O limiar padrão foi calibrado com TreeSHAP. Trocar o gerador muda a
    escala do ruído: recalibre com `calibrar_limiar` antes de confiar no
    veredito com ALE.
    """
    from modelcheck.shape import _shape_profile
    p1, w1 = perfis(model_v1, X_eval, feature_names, metodo, K)
    p2, w2 = perfis(model_v2, X_eval, feature_names, metodo, K)
    out = []
    for f in feature_names:
        if f not in p1 or f not in p2:
            continue
        if max(w1.get(f, 0.0), w2.get(f, 0.0)) < peso_minimo:
            continue
        d = distancia(p1[f], p2[f])
        if d <= limiar:
            continue
        out.append({"feature": f, "frechet": round(d, 3), "metodo": metodo,
                    "peso": round(max(w1.get(f, 0.0), w2.get(f, 0.0)), 3)})
    return out


def _maior_gap(a, b, bordas):
    """Bin em que as duas versões mais discordam, em unidades de atribuição."""
    d = np.asarray(b, float) - np.asarray(a, float)
    k = int(np.argmax(np.abs(d)))
    lo = float(bordas[k]) if k < len(bordas) else float("nan")
    hi = float(bordas[k + 1]) if k + 1 < len(bordas) else float("nan")
    return {"bin": k, "de": round(lo, 4), "ate": round(hi, 4),
            "delta": round(float(d[k]), 4),
            "v1": round(float(a[k]), 4), "v2": round(float(b[k]), 4)}


def divergencias_forma(X_eval: pd.DataFrame, sv1, sv2,
                       feature_names: List[str],
                       min_rel_importance: float = 0.01,
                       limiar: float = LIMIAR_PADRAO,
                       K: int = K_PADRAO,
                       peso_minimo: float = PESO_MINIMO_PADRAO) -> List[Dict]:
    """
    Detecta mudança de forma por distância e nomeia pela classe.

    Uma variável só é avaliada se pesar ao menos `min_rel_importance` em uma
    das versões: o perfil normalizado de uma variável irrelevante é ruído
    amplificado.
    """
    from modelcheck.shape import _shape_profile

    sv1 = np.asarray(sv1, float)
    sv2 = np.asarray(sv2, float)
    tot1 = float(np.abs(sv1).mean(axis=0).sum()) + 1e-12
    tot2 = float(np.abs(sv2).mean(axis=0).sum()) + 1e-12
    out = []
    for i, f in enumerate(feature_names):
        imp1 = float(np.abs(sv1[:, i]).mean())
        imp2 = float(np.abs(sv2[:, i]).mean())
        # mesmo raciocínio do bug 37: com muitas variáveis, o piso fixo de 5%
        # desliga a camada. Nunca exigir mais que o dobro da fatia uniforme.
        piso = min(max(min_rel_importance, peso_minimo),
                   2.0 / max(len(feature_names), 1))
        if max(imp1 / tot1, imp2 / tot2) < piso:
            continue
        x = X_eval[f].values
        a, bordas = perfil_com_bordas(x, sv1[:, i], K)
        b, _ = perfil_com_bordas(x, sv2[:, i], K)
        if a is None or b is None or len(a) != len(b):
            continue
        d = distancia(a, b)
        if d <= limiar:
            continue
        c1 = (_shape_profile(x, sv1[:, i]) or {}).get("shape")
        c2 = (_shape_profile(x, sv2[:, i]) or {}).get("shape")
        out.append({
            "feature": f,
            "shape_v1": c1,
            "shape_v2": c2,
            "frechet": round(d, 3),
            "limiar": limiar,
            # mudança dentro da mesma categoria (ex.: linear -> côncava) é
            # invisível para a classificação e visível para a distância.
            "mesma_categoria": bool(c1 is not None and c1 == c2),
            "spread": round(abs(imp1 - imp2) / (max(imp1, imp2) + 1e-8), 3),
            # perfis normalizados, para o laudo desenhar as duas curvas
            "perfil_v1": [round(v, 4) for v in _norm(a).tolist()],
            "perfil_v2": [round(v, 4) for v in _norm(b).tolist()],
            # em unidades de atribuição (mesma escala do alvo), para o laudo
            # poder dizer quanto e onde — sem isso o analista fica sem saber
            # se a mudança importa
            "efeito_v1": [round(float(v), 4) for v in a.tolist()],
            "efeito_v2": [round(float(v), 4) for v in b.tolist()],
            "bordas": [round(float(v), 4) for v in bordas.tolist()],
            "maior_gap": _maior_gap(a, b, bordas),
        })
    return out


def calibrar_limiar(pares, X_eval, feature_names=None, quantil: float = 0.95,
                    min_rel_importance: float = 0.01,
                    peso_minimo: float = PESO_MINIMO_PADRAO,
                    K: int = K_PADRAO):
    """
    Calibra o limiar da distância no SEU domínio (v0.12.0).

    `pares` é uma lista de pares de modelos SAUDÁVEIS — retreinos que você
    aceitaria promover (por exemplo, o mesmo procedimento com sementes
    diferentes, ou meses consecutivos sem incidente). A função mede a
    distância entre os perfis de cada variável avaliada e devolve o
    percentil pedido: acima dele, a mudança é maior que o ruído do seu
    próprio processo de retreino.

    Por que isto existe: o padrão 0.33 foi calibrado numa bancada sintética
    e deu 11% de falso positivo numa segunda bancada. Ele transfere só em
    parte.

    Devolve {'limiar', 'n_distancias', 'por_variavel', 'quantil'}.
    """
    import shap
    fn = list(feature_names or X_eval.columns)
    dists, por_var = [], {}
    for m1, m2 in pares:
        try:
            s1 = np.asarray(shap.TreeExplainer(m1).shap_values(X_eval), float)
            s2 = np.asarray(shap.TreeExplainer(m2).shap_values(X_eval), float)
        except Exception:
            continue    # noqa: lacuna-nao-aplicavel — par sem TreeSHAP
        if s1.ndim == 3:
            s1, s2 = s1[:, :, 1], s2[:, :, 1]
        t1 = float(np.abs(s1).mean(axis=0).sum()) + 1e-12
        t2 = float(np.abs(s2).mean(axis=0).sum()) + 1e-12
        for i, f in enumerate(fn):
            w = max(float(np.abs(s1[:, i]).mean()) / t1,
                    float(np.abs(s2[:, i]).mean()) / t2)
            if w < max(min_rel_importance, peso_minimo):
                continue
            a = perfil(X_eval[f].values, s1[:, i], K)
            b = perfil(X_eval[f].values, s2[:, i], K)
            if a is None or b is None:
                continue
            d = distancia(a, b)
            dists.append(d); por_var.setdefault(f, []).append(round(d, 3))
    if len(dists) < 5:
        return {"limiar": None, "n_distancias": len(dists),
                "reason": ("poucos pares saudáveis para calibrar; use ao menos "
                           "5 distâncias (mais pares ou mais variáveis)"),
                "por_variavel": por_var, "quantil": quantil}
    return {"limiar": round(float(np.quantile(dists, quantil)), 3),
            "n_distancias": len(dists), "por_variavel": por_var,
            "quantil": quantil,
            "note": ("limiar do seu domínio: acima dele a mudança é maior que "
                     "o ruído do seu próprio retreino")}


def diff_influencia(sv1, sv2, feature_names: List[str]) -> List[Dict]:
    """
    Quanto cada variável pesa em cada versão, em fração do total (v0.14.0).

    Serve para o "model diff": o que ganhou influência, o que perdeu, e o que
    manteve o peso. A terceira categoria é a que só faz sentido junto com a
    forma — peso igual e comportamento diferente.
    """
    sv1 = np.asarray(sv1, float); sv2 = np.asarray(sv2, float)
    m1 = np.abs(sv1).mean(axis=0); m2 = np.abs(sv2).mean(axis=0)
    t1 = float(m1.sum()) + 1e-12; t2 = float(m2.sum()) + 1e-12
    out = []
    for i, f in enumerate(feature_names):
        p1, p2 = float(m1[i]) / t1, float(m2[i]) / t2
        out.append({"feature": f, "peso_v1": round(p1, 4), "peso_v2": round(p2, 4),
                    "delta_pp": round((p2 - p1) * 100, 1)})
    return sorted(out, key=lambda d: -max(d["peso_v1"], d["peso_v2"]))
