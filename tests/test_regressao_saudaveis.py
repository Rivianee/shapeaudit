"""
Regressão permanente: retreino saudável não pode ser bloqueado (v1.0.4).

Por que existe: correção que amplia gatilho virou classe de bug. Nesta
própria versão, três correções criaram falso alarme e só foram pegas
rodando retreinos reais:
  - bug 83: a correção da sentinela (73) bloqueou 3 retreinos saudáveis
    do ONS;
  - bug 94: a correção de multiclasse (68) pôs o vinho em REVISAR em 6 de
    6 retreinos saudáveis;
  - bug 81/82: alta dimensão bloqueava metade dos retreinos saudáveis.

Os retreinos do ONS são os do artigo (mesmas variáveis, janelas de 24
meses com passo de 3, floresta de 80 árvores). Por padrão roda uma amostra
(uma janela a cada 8, ~2 min). A bateria completa — 157 retreinos distintos — roda
com `pytest -m slow` e deve rodar antes de toda publicação:

    pytest tests/test_regressao_saudaveis.py -m slow
"""
import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from shapeaudit import Guard

D = os.path.join(os.path.dirname(__file__), "dados")


def _ena():
    d = pd.read_csv(os.path.join(D, "ons_ena_se_diaria.csv.gz"), parse_dates=["data"])
    d["lag1"] = d.ena.shift(1); d["lag7"] = d.ena.shift(7); d["lag30"] = d.ena.shift(30)
    d["ma30"] = d.ena.rolling(30).mean().shift(1)
    d["mes"] = d.data.dt.month; d["doy"] = d.data.dt.dayofyear
    return d.dropna().reset_index(drop=True), "ena", ["lag1", "lag7", "lag30", "ma30", "mes", "doy"]


def _carga():
    d = pd.read_csv(os.path.join(D, "ons_carga_se_diaria.csv.gz"), parse_dates=["data"])
    d["lag1"] = d.carga.shift(1); d["lag7"] = d.carga.shift(7); d["lag365"] = d.carga.shift(365)
    d["ma30"] = d.carga.rolling(30).mean().shift(1)
    d["dia_sem"] = d.data.dt.dayofweek; d["mes"] = d.data.dt.month; d["doy"] = d.data.dt.dayofyear
    return d.dropna().reset_index(drop=True), "carga", ["lag1", "lag7", "lag365", "ma30", "dia_sem", "mes", "doy"]


def _janelas(d):
    out, t, fim = [], d.data.min(), d.data.max()
    while t + pd.DateOffset(months=24) <= fim:
        w = d[(d.data >= t) & (d.data < t + pd.DateOffset(months=24))]
        if len(w) > 500:
            out.append((t, w))
        t += pd.DateOffset(months=3)
    return out


def _retreinos(serie, protocolo, passo):
    d, alvo, fe = serie()
    mk = lambda w: RandomForestRegressor(n_estimators=80, max_depth=8,
                                         random_state=42, n_jobs=1).fit(w[fe], w[alvo])
    jan = _janelas(d)
    ev_fixo = d.sample(400, random_state=0)
    casos = []
    for i in range(1, len(jan), passo):
        (_, w0), (t1, w1) = jan[i - 1], jan[i]
        if protocolo == "limpo":
            fut = d[d.data > w1.data.max()]
            if len(fut) < 400:
                continue
            ev = fut.sample(min(1500, len(fut)), random_state=0)
        else:
            ev = ev_fixo
        casos.append((str(t1.date()), mk(w0), mk(w1), ev[fe], ev[alvo], w1[fe]))
    return casos


def _bloqueios(casos):
    ruins = []
    for nome, v1, v2, X, y, Xt in casos:
        r = Guard().audit(v1, v2, X, y, X_train=Xt)
        if r.verdict == "BLOQUEADO":
            ruins.append((nome, r.root_cause_features))
    return ruins


@pytest.mark.parametrize("serie,protocolo", [(_ena, "publicado"), (_carga, "publicado"),
                                             (_ena, "limpo")])
def test_ons_amostra_sem_bloqueado(serie, protocolo):
    casos = _retreinos(serie, protocolo, passo=8)
    assert len(casos) >= 5
    assert _bloqueios(casos) == []


@pytest.mark.slow
# 55 + 51 + 51 = 157 retreinos distintos. Os scripts originais do artigo
# rodavam em lotes que se sobrepunham na fronteira (lote 0 = janelas 1-12,
# lote 1 = 12-24...) e contavam a janela da fronteira duas vezes: daí os
# "59" do ENA e "54" da carga.
@pytest.mark.parametrize("serie,protocolo,n_esperado", [(_ena, "publicado", 55),
                                                        (_carga, "publicado", 51),
                                                        (_ena, "limpo", 51)])
def test_ons_157_retreinos_sem_bloqueado(serie, protocolo, n_esperado):
    casos = _retreinos(serie, protocolo, passo=1)
    assert len(casos) == n_esperado
    assert _bloqueios(casos) == []


# ── tipos de variável que o ONS não cobre (bugs 44, 45, 47, 68, 94) ───
def _churn():
    raiz = os.path.dirname(os.path.dirname(__file__))
    d = pd.read_excel(os.path.join(raiz, "exemplo_churn.xlsx"))
    y = (d.pop("churn") == "Yes").astype(int)
    X = d.apply(lambda c: c if c.dtype.kind in "if" else
                pd.Series(pd.factorize(c)[0], index=c.index)).astype(float)
    return X, y


def _vinho():
    w = pd.read_csv(os.path.join(D, "winequality-red.csv"), sep=None, engine="python")
    return w, w.pop("quality").clip(4, 7).astype(int)


@pytest.mark.parametrize("dados,nome", [(_churn, "binárias e categóricas"),
                                        (_vinho, "multiclasse")])
def test_retreino_saudavel_nao_alarma(dados, nome):
    X, y = dados()
    vereditos = []
    for s in range(3):
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=.4, random_state=s, stratify=y)
        mk = lambda k: RandomForestClassifier(n_estimators=60, max_depth=8,
                                              random_state=k, n_jobs=1).fit(Xa, ya)
        vereditos.append(Guard().audit(mk(s), mk(s + 100), Xb, yb, X_train=Xa).verdict)
    assert "BLOQUEADO" not in vereditos, (nome, vereditos)
    assert "ATENCAO" not in vereditos, (nome, vereditos)     # mesmo dado: treino = uso
    assert vereditos.count("REVISAR") <= 1, (nome, vereditos)
