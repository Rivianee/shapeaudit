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

# BUG 66 (v1.0.4): variável cujo efeito vem de INTERAÇÃO tem curva média
# quase plana. Com y = 3*x1*x2, x1 pesa ~20% (passa no piso), mas a média da
# atribuição por faixa de x1 é ruído de ±0,1. `_norm` divide pelo próprio
# desvio e infla esse ruído até a escala de um sinal real; dois retreinos
# saudáveis (só muda a semente) davam REVISAR em 6 de 6 pares.
#
# A comparação de forma só faz sentido quando existe uma forma: a amplitude
# da curva média tem de ser comparável à atribuição típica da variável.
# Efeito linear dá razão ~4,4; em U, ~3,9; interação pura, ~0,2. Pulamos o
# par quando AS DUAS versões ficam abaixo de 0,5 — se só uma tem forma, a
# mudança é real e continua sendo comparada.
RAZAO_MIN_EFEITO_PRINCIPAL = 0.5


def tem_efeito_principal(perfil_atrib, s_col) -> bool:
    """A curva média tem amplitude relevante frente à atribuição típica?"""
    a = np.asarray(perfil_atrib, float)
    tip = float(np.mean(np.abs(np.asarray(s_col, float))))
    if a.size < 2 or not np.isfinite(a).all() or tip <= 1e-12:
        return False
    return float(np.ptp(a)) / tip >= RAZAO_MIN_EFEITO_PRINCIPAL


def piso_relevancia(n_features: int, min_rel_importance: float = 0.01,
                    peso_minimo: float = PESO_MINIMO_PADRAO) -> float:
    """
    Piso de peso para a camada de forma avaliar uma variável.

    BUG 67 (v1.0.4): `calibrar_limiar` usava max(min_rel, peso_minimo) e o
    detector usava min(..., 2/p). Com p > 40 a calibração nunca via as
    variáveis fracas que o detector avalia — justamente as mais ruidosas.
    Uma regra só, usada nos dois lugares.
    """
    return min(max(min_rel_importance, peso_minimo),
               2.0 / max(int(n_features), 1))


def como_array_shap(sv) -> np.ndarray:
    """
    Normaliza a saída do TreeExplainer para (n, p) ou (n, p, classes).

    shap antigo devolve lista [classe0, classe1, ...]; np.asarray nessa lista
    dá (classes, n, p), e `sv[:, :, 1]` pegava o eixo errado em silêncio.
    """
    if isinstance(sv, list):
        sv = np.stack([np.asarray(x, float) for x in sv], axis=-1)
    sv = np.asarray(sv, float)
    if sv.ndim == 3 and sv.shape[2] == 1:
        sv = sv[:, :, 0]
    return sv


def fatias_por_classe(sv) -> list:
    """
    Lista de matrizes (n, p) a comparar.

    BUG 68 (v1.0.4): multiclasse usava só a classe 1. Uma variável que separa
    a classe 0 da 2 ficava com peso quase nulo e o colapso dela passava. Em
    binário as duas fatias são espelho uma da outra, então mantemos a
    classe 1 (resultado idêntico ao de antes); com 3+ classes, cada classe é
    comparada.
    """
    sv = como_array_shap(sv)
    if sv.ndim == 2:
        return [sv]
    if sv.shape[2] == 2:
        return [sv[:, :, 1]]
    return [sv[:, :, k] for k in range(sv.shape[2])]


def importancia_media_abs(sv) -> np.ndarray:
    """Média de |atribuição| por variável, somando todas as classes."""
    sv = como_array_shap(sv)
    if sv.ndim == 3:
        if sv.shape[2] == 2:
            return np.abs(sv[:, :, 1]).mean(axis=0)
        return np.abs(sv).mean(axis=(0, 2))
    return np.abs(sv).mean(axis=0)


def indices_por_nome(X_eval, feature_names, n_cols: int) -> list:
    """
    Posição de cada variável na matriz de atribuição.

    BUG 69 (v1.0.4): a matriz SHAP segue a ordem das colunas de X_eval, mas
    era rotulada pela POSIÇÃO em feature_names. Com nomes em outra ordem, o
    colapso de `a` era atribuído a `d`. Agora o rótulo vem do nome.
    """
    cols = list(getattr(X_eval, "columns", []))
    if cols and len(cols) == n_cols:
        pos = {c: i for i, c in enumerate(cols)}
        faltam = [f for f in feature_names if f not in pos]
        if faltam:
            raise ValueError(f"variáveis não encontradas em X_eval: {faltam}")
        return [pos[f] for f in feature_names]
    if len(feature_names) != n_cols:
        raise ValueError(
            f"feature_names tem {len(feature_names)} nomes, mas a atribuição "
            f"tem {n_cols} colunas")
    return list(range(n_cols))


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
           metodo: str = "treeshap", K: int = K_PADRAO, _tipico=None):
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
        sv = fatias_por_classe(shap.TreeExplainer(model).shap_values(X_eval))[0]
        idx = indices_por_nome(X_eval, feature_names, sv.shape[1])
        tot = float(np.abs(sv).mean(axis=0).sum()) + 1e-12
        for i, f in zip(idx, feature_names):
            p = perfil(X_eval[f].values, sv[:, i], K)
            if p is not None:
                out[f] = p
                peso[f] = float(np.abs(sv[:, i]).mean()) / tot
                if _tipico is not None:
                    _tipico[f] = sv[:, i]
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
    s1, s2 = ({}, {}) if metodo == "treeshap" else (None, None)
    p1, w1 = perfis(model_v1, X_eval, feature_names, metodo, K, _tipico=s1)
    p2, w2 = perfis(model_v2, X_eval, feature_names, metodo, K, _tipico=s2)
    out = []
    for f in feature_names:
        if f not in p1 or f not in p2:
            continue
        if max(w1.get(f, 0.0), w2.get(f, 0.0)) < peso_minimo:
            continue
        # BUG 66: no TreeSHAP, pular variável sem efeito principal nas duas
        # versões. O ALE já é curva de efeito principal; não precisa.
        if s1 is not None and not (tem_efeito_principal(p1[f], s1[f])
                                   or tem_efeito_principal(p2[f], s2[f])):
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
                       peso_minimo: float = PESO_MINIMO_PADRAO,
                       _pulados: Optional[list] = None,
                       _totais: Optional[tuple] = None) -> List[Dict]:
    """
    Detecta mudança de forma por distância e nomeia pela classe.

    Uma variável só é avaliada se pesar ao menos `min_rel_importance` em uma
    das versões: o perfil normalizado de uma variável irrelevante é ruído
    amplificado.
    """
    from modelcheck.shape import _shape_profile

    f1, f2 = fatias_por_classe(sv1), fatias_por_classe(sv2)
    if len(f1) != len(f2):
        raise ValueError("as duas versões têm número de classes diferente")
    if len(f1) > 1:
        # multiclasse: compara classe a classe e fica com a maior distância
        # de cada variável (BUG 68).
        #
        # BUG 94 (v1.0.4): a primeira versão media o peso de cada variável
        # DENTRO da classe. Classe rara tem atribuição pequena e ruidosa, mas
        # dentro dela qualquer variável "pesa"; com 4 classes, o vinho deu
        # REVISAR em 6 de 6 retreinos saudáveis (1.0.3: 1 de 6), com 7 a 9
        # variáveis por retreino. O peso agora é medido contra o total de
        # TODAS as classes: só entra o par (classe, variável) que pesa no
        # modelo inteiro.
        T1 = sum(float(np.abs(a).mean(axis=0).sum()) for a in f1) + 1e-12
        T2 = sum(float(np.abs(a).mean(axis=0).sum()) for a in f2) + 1e-12
        melhor = {}
        for k, (a1, a2) in enumerate(zip(f1, f2)):
            for d in divergencias_forma(X_eval, a1, a2, feature_names,
                                        min_rel_importance, limiar, K,
                                        peso_minimo, _pulados,
                                        _totais=(T1, T2)):
                d["classe"] = k
                if d["feature"] not in melhor or d["frechet"] > melhor[d["feature"]]["frechet"]:
                    melhor[d["feature"]] = d
        return list(melhor.values())
    sv1, sv2 = f1[0], f2[0]
    idx = indices_por_nome(X_eval, feature_names, sv1.shape[1])
    tot1 = float(np.abs(sv1).mean(axis=0).sum()) + 1e-12
    tot2 = float(np.abs(sv2).mean(axis=0).sum()) + 1e-12
    if _totais is not None:              # multiclasse: peso no modelo inteiro
        tot1, tot2 = _totais
    # mesmo raciocínio do bug 37: com muitas variáveis, o piso fixo de 5%
    # desliga a camada. Nunca exigir mais que o dobro da fatia uniforme.
    piso = piso_relevancia(len(feature_names), min_rel_importance, peso_minimo)
    out = []
    for i, f in zip(idx, feature_names):
        imp1 = float(np.abs(sv1[:, i]).mean())
        imp2 = float(np.abs(sv2[:, i]).mean())
        if max(imp1 / tot1, imp2 / tot2) < piso:
            continue
        x = X_eval[f].values
        a, bordas = perfil_com_bordas(x, sv1[:, i], K)
        b, _ = perfil_com_bordas(x, sv2[:, i], K)
        if a is None or b is None or len(a) != len(b):
            continue
        if not (tem_efeito_principal(a, sv1[:, i])
                or tem_efeito_principal(b, sv2[:, i])):
            # BUG 66: sem forma nas duas versões. Registra, para o laudo não
            # dizer que a forma foi comparada quando não havia forma.
            if _pulados is not None and f not in _pulados:
                _pulados.append(f)
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
    dists, por_var, pulados = [], {}, 0
    for m1, m2 in pares:
        try:
            s1 = como_array_shap(shap.TreeExplainer(m1).shap_values(X_eval))
            s2 = como_array_shap(shap.TreeExplainer(m2).shap_values(X_eval))
        except Exception:
            pulados += 1   # par sem TreeSHAP (modelo que não é árvore)
            continue
        # BUG 90 (v1.0.4): só media a classe 0, enquanto o Guard compara
        # todas e fica com a maior distância. Com 3 classes o limiar saía
        # baixo demais para o que o Guard de fato mede.
        f1, f2 = fatias_por_classe(s1), fatias_por_classe(s2)
        # mesma regra de peso do Guard em multiclasse (BUG 94)
        tot = None if len(f1) == 1 else (
            sum(float(np.abs(a).mean(axis=0).sum()) for a in f1) + 1e-12,
            sum(float(np.abs(a).mean(axis=0).sum()) for a in f2) + 1e-12)
        for a1, a2 in zip(f1, f2):
            _calibra_par(a1, a2, X_eval, fn, min_rel_importance, peso_minimo,
                         K, dists, por_var, tot)
    if len(dists) < 5:
        # BUG 90: devolvia limiar=None, e Guard(frechet_threshold=None)
        # usava o padrão sem avisar — o fluxo do README calibrava "em vão".
        motivo = (f"{pulados} de {len(pares)} pares não são modelos de árvore "
                  "(a calibração usa TreeSHAP)" if pulados else
                  "poucos pares saudáveis")
        raise ValueError(
            f"calibrar_limiar: só {len(dists)} distâncias medidas ({motivo}); "
            "são necessárias ao menos 5. Use mais pares ou mais variáveis.")
    return {"limiar": round(float(np.quantile(dists, quantil)), 3),
            "n_distancias": len(dists), "por_variavel": por_var,
            "quantil": quantil,
            "note": ("limiar do seu domínio: acima dele a mudança é maior que "
                     "o ruído do seu próprio retreino")}


def _calibra_par(s1, s2, X_eval, fn, min_rel_importance, peso_minimo, K,
                 dists, por_var, totais=None):
    idx = indices_por_nome(X_eval, fn, s1.shape[1])
    t1 = float(np.abs(s1).mean(axis=0).sum()) + 1e-12
    t2 = float(np.abs(s2).mean(axis=0).sum()) + 1e-12
    if totais is not None:
        t1, t2 = totais
    piso = piso_relevancia(len(fn), min_rel_importance, peso_minimo)
    for i, f in zip(idx, fn):
        w = max(float(np.abs(s1[:, i]).mean()) / t1,
                float(np.abs(s2[:, i]).mean()) / t2)
        if w < piso:
            continue
        a = perfil(X_eval[f].values, s1[:, i], K)
        b = perfil(X_eval[f].values, s2[:, i], K)
        if a is None or b is None:
            continue
        if not (tem_efeito_principal(a, s1[:, i])
                or tem_efeito_principal(b, s2[:, i])):
            continue
        d = distancia(a, b)
        dists.append(d); por_var.setdefault(f, []).append(round(d, 3))


def diff_influencia(sv1, sv2, feature_names: List[str],
                    X_eval=None) -> List[Dict]:
    """
    Quanto cada variável pesa em cada versão, em fração do total (v0.14.0).

    Serve para o "model diff": o que ganhou influência, o que perdeu, e o que
    manteve o peso. A terceira categoria é a que só faz sentido junto com a
    forma — peso igual e comportamento diferente.
    """
    m1 = importancia_media_abs(sv1); m2 = importancia_media_abs(sv2)
    t1 = float(m1.sum()) + 1e-12; t2 = float(m2.sum()) + 1e-12
    idx = indices_por_nome(X_eval, feature_names, len(m1)) if X_eval is not None \
        else list(range(len(feature_names)))
    out = []
    for i, f in zip(idx, feature_names):
        p1, p2 = float(m1[i]) / t1, float(m2[i]) / t2
        out.append({"feature": f, "peso_v1": round(p1, 4), "peso_v2": round(p2, 4),
                    "delta_pp": round((p2 - p1) * 100, 1)})
    return sorted(out, key=lambda d: -max(d["peso_v1"], d["peso_v2"]))
