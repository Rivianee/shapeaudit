"""
Todo sinal chega a toda saída (v1.0.4).

Classe de bug que apareceu várias vezes: o sinal existia e não chegava ao
veredito por um dos caminhos. `pipeline.combinar` jogava fora o ATENÇÃO do
Guard (bug 76); a CLI tratava INDETERMINADO como sucesso (85) e ignorava
--monitor (86); o snapshot usava uma cópia antiga da lógica e via mudança de
forma como APROVADO (93); `combinar` ignorava o modo exploratório (92).

Para cada tipo de veredito, monta um cenário e exige que API, dicionário,
laudo, texto, CLI, código de saída e `combinar` digam a mesma coisa.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from shapeaudit import Guard

CABECALHO = {"BLOQUEADO": "Não promova", "REVISAR": "Revise antes de promover",
             "ATENCAO": "Promova com atenção", "APROVADO": "Pode promover"}


def _rf(X, y, s=0):
    return RandomForestRegressor(n_estimators=30, max_depth=6,
                                 random_state=s, n_jobs=1).fit(X, y)


def _base():
    r = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame(r.normal(size=(n, 4)), columns=list("abcd"))
    X["a2"] = X.a + r.normal(0, .02, n)            # quase-cópia de a
    y = 3 * X.a + 2 * X.b + X.c + r.normal(0, .3, n)
    y_rel = 2 * X.b + X.c + np.random.default_rng(9).normal(0, .3, n)
    y_forma = 3 * X.a + 2 * (X.b ** 2 - 1) + X.c
    return X, y, y_rel, y_forma


X, Y, Y_REL, Y_FORMA = _base()
CENARIOS = {
    "saudavel": dict(v2=lambda: _rf(X, Y, 1), esperado="APROVADO"),
    "colapso": dict(v2=lambda: _rf(X.assign(a=0.0, a2=0.0), Y, 1), esperado="BLOQUEADO"),
    "treino_x_uso": dict(v2=lambda: _rf(X, Y, 1), X_train=X.assign(b=X.b * 100), esperado="ATENCAO"),
    "clone": dict(v2=lambda: _rf(X.assign(a=0.0), Y, 1), esperado="ATENCAO"),
    "reaprendizado": dict(v2=lambda: _rf(X, Y_REL, 1), y=Y_REL, X_train=X, esperado="REVISAR"),
    "forma": dict(v2=lambda: _rf(X, Y_FORMA, 1), esperado="REVISAR"),
    "modelo_constante": dict(v2=lambda: _rf(X, Y * 0 + Y.mean(), 1), X_train=X, esperado="BLOQUEADO"),
    "integridade_falha": dict(v2=lambda: _rf(X.assign(a=0.0, a2=0.0), Y, 1), linhas=9, esperado="INDETERMINADO"),
    "exploratorio": dict(v2=lambda: _rf(X, Y_FORMA, 1), modo="exploratory", esperado="APROVADO"),
}


@pytest.fixture(scope="module")
def v1():
    return _rf(X, Y)


@pytest.mark.parametrize("nome", list(CENARIOS))
def test_todas_as_saidas_concordam(nome, v1, tmp_path):
    from modelcheck.cli import _codigo, main
    from modelcheck.pipeline import _SEV_COMB, combinar, exit_code
    c = CENARIOS[nome]
    v2 = c["v2"]()
    y = c.get("y", Y)
    Xe, ye = X, y
    if "linhas" in c:
        Xe, ye = X.iloc[:c["linhas"]], y.iloc[:c["linhas"]]
    modo = c.get("modo", "retraining")
    res = Guard(mode=modo).audit(v1, v2, Xe, ye, X_train=c.get("X_train"))
    v = res.verdict
    assert v == c["esperado"], (nome, v, res.verdict_driver)

    d = res.to_dict()
    assert d["verdict"] == v
    assert d["root_cause_features"] == res.root_cause_features
    assert d["skew_features"] == res.skew_features
    assert d["clone_redistribution"] == res.clone_redistribution
    assert sorted(d["relearning"]) == sorted(res.relearning)
    assert len(d["divergences"]) == len(res.shape_divergences)
    if v in CABECALHO:
        assert CABECALHO[v] in res.to_html()
    assert v in res.plain

    # CLI, com os mesmos dados (inclusive X_train e modo)
    joblib.dump(v1, tmp_path / "a.pkl")
    joblib.dump(v2, tmp_path / "b.pkl")
    Xe.assign(alvo=np.asarray(ye)).to_csv(tmp_path / "e.csv", index=False)
    args = ["compare", str(tmp_path / "a.pkl"), str(tmp_path / "b.pkl"),
            str(tmp_path / "e.csv"), "--target", "alvo", "--format", "json",
            "-o", str(tmp_path / "o.json"), "--fail-on", "low", "--mode", modo]
    if c.get("X_train") is not None:
        c["X_train"].to_csv(tmp_path / "t.csv", index=False)
        args += ["--train", str(tmp_path / "t.csv")]
    cod = main(args)
    vc = json.loads((tmp_path / "o.json").read_text(encoding="utf-8"))["verdict"]
    assert vc == v
    assert cod == _codigo(v, "low") == exit_code({"verdict": v}, "low")

    # combinar acrescenta sinal externo; nunca fica mais brando que o Guard
    comb = combinar(res, drifted_columns=[])["verdict"]
    assert _SEV_COMB[comb] >= _SEV_COMB[v], (comb, v)
    if nome == "exploratorio":
        assert comb == "APROVADO"


@pytest.mark.parametrize("nome", ["saudavel", "colapso", "forma"])
def test_snapshot_concorda_com_o_guard(nome, v1, tmp_path):
    from modelcheck import snapshot
    v2 = CENARIOS[nome]["v2"]()
    p = str(tmp_path / "s.json")
    snapshot.save(v1, X, p)
    s = snapshot.diff(p, v2, X)
    g = Guard().audit(v1, v2, X, Y)
    assert s["verdict"] == g.verdict, (s["verdict"], g.verdict)
    assert s["root_cause_features"] == g.root_cause_features
    assert sorted(x["feature"] for x in s["divergences"]) == \
        sorted(x["feature"] for x in g.shape_divergences)
