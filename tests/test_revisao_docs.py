"""
Documentação contra realidade (v1.0.4).

Cada teste cobra uma promessa que a documentação faz e que o código não
cumpria: parâmetro aceito sem efeito, código de saída que deixava passar
modelo quebrado, visto sobre variável que nem existia.
"""
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from shapeaudit import Guard


def _rf(X, y, s=0):
    return RandomForestRegressor(n_estimators=30, max_depth=6,
                                 random_state=s, n_jobs=1).fit(X, y)


@pytest.fixture(scope="module")
def dados():
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.normal(size=(500, 4)), columns=list("abcd"))
    y = 3 * X.a + 2 * X.b + X.c + r.normal(0, .3, 500)
    return X, y


@pytest.fixture(scope="module")
def forma_mudou(dados):
    """b muda de linear para U: só a forma muda."""
    X, y = dados
    y2 = 3 * X.a + 2 * (X.b ** 2 - 1) + X.c
    return _rf(X, y), _rf(X, y2, 1)


# BUG 87 — mode='exploratory' não fazia nada ─────────────────────────
def test_exploratorio_nao_alerta_mudanca_de_forma(dados, forma_mudou):
    X, y = dados
    v1, v2 = forma_mudou
    assert Guard().audit(v1, v2, X, y).verdict == "REVISAR"
    r = Guard(mode="exploratory").audit(v1, v2, X, y)
    assert r.verdict == "APROVADO" and r.verdict_driver == "forma_esperada"
    assert r.to_dict()["mode"] == "exploratory"
    assert "modo exploratório" in r.to_html()


def test_exploratorio_continua_bloqueando_colapso(dados):
    X, y = dados
    r = Guard(mode="exploratory").audit(_rf(X, y), _rf(X.assign(a=0.0), y, 1), X, y)
    assert r.verdict == "BLOQUEADO"


def test_modo_com_erro_de_digitacao_e_recusado():
    with pytest.raises(ValueError, match="mode"):
        Guard(mode="exploratorio")


# BUG 84 — multiclasse fora de árvore dava APROVADO sem lacuna ─────────
def test_multiclasse_fora_de_arvore_nao_aprova():
    r_ = np.random.default_rng(1)
    X = pd.DataFrame(r_.normal(size=(400, 3)), columns=list("abc"))
    y = np.where(X.a > .5, 1, np.where(X.b > 0, 2, 0))
    m = LogisticRegression(max_iter=500).fit(X, y)
    r = Guard().audit(m, m, X, pd.Series(y))
    assert r.verdict == "INDETERMINADO", r.verdict
    assert any("shape" in x.get("layer", "") for x in r.not_evaluated)


# BUG 89 — nome monitorado inexistente virava "continua influenciando" ──
def test_monitorada_inexistente_e_erro(dados):
    X, y = dados
    v = _rf(X, y)
    with pytest.raises(ValueError, match="não existem"):
        Guard().check_features(v, v, X, monitored=["A"])


# BUG 90 — calibração que não calibra ─────────────────────────────────
def test_calibrar_com_poucos_pares_e_erro(dados):
    from shapeaudit.frechet import calibrar_limiar
    X, y = dados
    with pytest.raises(ValueError, match="distâncias"):
        calibrar_limiar([(_rf(X[["a"]], y), _rf(X[["a"]], y, 1))], X[["a"]])


def test_calibrar_fora_de_arvore_diz_o_motivo(dados):
    from shapeaudit.frechet import calibrar_limiar
    from sklearn.linear_model import Ridge
    X, y = dados
    m = Ridge().fit(X, y)
    with pytest.raises(ValueError, match="não são modelos de árvore"):
        calibrar_limiar([(m, m)] * 3, X)


def test_calibrar_multiclasse_mede_todas_as_classes():
    from shapeaudit.frechet import calibrar_limiar
    r_ = np.random.default_rng(2)
    X = pd.DataFrame(r_.normal(size=(600, 4)), columns=list("wxyz"))
    y = np.where(X.w > .3, 1, np.where(X.z > 0, 2, 0))
    m = lambda s: RandomForestClassifier(n_estimators=30, max_depth=5,
                                         random_state=s, n_jobs=1).fit(X, y)
    cal = calibrar_limiar([(m(s), m(s + 50)) for s in range(3)], X)
    assert "z" in cal["por_variavel"], cal["por_variavel"].keys()


def test_limiar_em_modelo_fora_de_arvore_avisa(dados):
    from sklearn.linear_model import Ridge
    X, y = dados
    m = Ridge().fit(X, y)
    with pytest.warns(UserWarning, match="frechet_threshold"):
        Guard(frechet_threshold=0.5).audit(m, m, X, y)


# BUG 88 — rótulos aceitos e nunca mostrados ──────────────────────────
def test_rotulos_aparecem_no_laudo(dados):
    X, y = dados
    v = _rf(X, y)
    r = Guard().audit(v, v, X, y, label_v1="PRODUCAO_JAN", label_v2="CANDIDATA_FEV")
    h = r.to_html()
    assert "PRODUCAO_JAN" in h and "CANDIDATA_FEV" in h


# BUG 85 / 86 — CLI ───────────────────────────────────────────────────
@pytest.fixture(scope="module")
def arquivos(tmp_path_factory, dados):
    X, y = dados
    d = tmp_path_factory.mktemp("cli")
    joblib.dump(_rf(X, y), d / "v1.pkl")
    joblib.dump(_rf(X.assign(a=0.0), y, 1), d / "quebrado.pkl")
    joblib.dump(_rf(X, y, 1), d / "saudavel.pkl")
    X.assign(alvo=y).to_csv(d / "eval.csv", index=False)
    return d


def _cli(*args):
    from modelcheck.cli import main
    return main([str(a) for a in args] + ["--format", "json", "-o", os.devnull])


def test_cli_bloqueado_sai_1_por_padrao(arquivos):
    d = arquivos
    assert _cli("compare", d / "v1.pkl", d / "quebrado.pkl", d / "eval.csv", "--target", "alvo") == 1


def test_cli_saudavel_sai_0(arquivos):
    d = arquivos
    assert _cli("compare", d / "v1.pkl", d / "saudavel.pkl", d / "eval.csv", "--target", "alvo") == 0


def test_cli_fail_on_none_continua_disponivel(arquivos):
    d = arquivos
    assert _cli("compare", d / "v1.pkl", d / "quebrado.pkl", d / "eval.csv",
                "--target", "alvo", "--fail-on", "none") == 0


def test_cli_indeterminado_sai_2():
    from modelcheck.cli import _codigo
    for nivel in ("none", "low", "medium", "high"):
        assert _codigo("INDETERMINADO", nivel) == 2


def test_cli_compare_monitor_tem_efeito(arquivos, tmp_path):
    """--monitor em compare era só copiado para a saída."""
    import json
    from modelcheck.cli import main
    d = arquivos
    out = tmp_path / "r.json"
    main(["compare", str(d / "v1.pkl"), str(d / "quebrado.pkl"), str(d / "eval.csv"),
          "--target", "alvo", "--monitor", "a", "--format", "json", "-o", str(out),
          "--fail-on", "none"])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res.get("monitored_hits") == ["a"]
