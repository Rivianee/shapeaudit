"""
Testes da revisão do núcleo do veredito (v1.0.4).

Cada teste reproduz um defeito confirmado na revisão de outubro/2026 e falha
na 1.0.3. Os modelos são pequenos de propósito: o que se testa é o caminho
do veredito, não a qualidade do ajuste.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from shapeaudit import Guard


def _dados(n=600, s=0):
    r = np.random.default_rng(s)
    X = pd.DataFrame(r.normal(size=(n, 4)), columns=list("abcd"))
    y = 3 * X.a + 2 * X.b + X.c + .5 * X.d + r.normal(0, .3, n)
    return X, y


def _rf(X, y, s=0):
    return RandomForestRegressor(n_estimators=30, max_depth=6,
                                 random_state=s, n_jobs=1).fit(X, y)


@pytest.fixture(scope="module")
def quebrado():
    """v2 treinada com `a` constante: falha de pipeline clássica."""
    X, y = _dados()
    Xe, ye = _dados(300, 1)
    v1 = _rf(X, y)
    Xb = X.assign(a=0.0)
    return dict(X=X, y=y, Xe=Xe, ye=ye, v1=v1, v2=_rf(Xb, y, 1), Xb=Xb)


# BUG 69 ─────────────────────────────────────────────────────────────
def test_nomes_fora_de_ordem_nao_trocam_a_culpa(quebrado):
    q = quebrado
    r = Guard().audit(q["v1"], q["v2"], q["Xe"], q["ye"], ["d", "c", "b", "a"])
    assert r.verdict == "BLOQUEADO"
    assert r.root_cause_features == ["a"], r.root_cause_features


def test_nome_inexistente_e_erro_claro(quebrado):
    q = quebrado
    with pytest.raises(ValueError, match="não estão em X_eval"):
        Guard().audit(q["v1"], q["v2"], q["Xe"], q["ye"], ["a", "b", "zz", "d"])


# BUG 74 ─────────────────────────────────────────────────────────────
def test_numpy_com_nomes_detecta_a_quebra(quebrado):
    q = quebrado
    v1 = _rf(q["X"].values, q["y"])
    v2 = _rf(q["Xb"].values, q["y"], 1)
    r = Guard().audit(v1, v2, q["Xe"].values, q["ye"].values, list("abcd"))
    assert r.verdict == "BLOQUEADO"
    assert r.root_cause_features == ["a"]


def test_numpy_sem_nomes_nao_quebra(quebrado):
    q = quebrado
    v1 = _rf(q["X"].values, q["y"])
    v2 = _rf(q["Xb"].values, q["y"], 1)
    r = Guard().audit(v1, v2, q["Xe"].values, q["ye"].values)
    assert r.verdict == "BLOQUEADO"
    assert r.root_cause_features == ["f0"]


# BUG 75 ─────────────────────────────────────────────────────────────
def test_y_com_tamanho_errado_e_erro(quebrado):
    q = quebrado
    with pytest.raises(ValueError, match="y_eval tem 50"):
        Guard().audit(q["v1"], q["v2"], q["Xe"], q["ye"].iloc[:50])


def test_y_com_nan_e_erro(quebrado):
    q = quebrado
    y = q["ye"].copy()
    y.iloc[:3] = np.nan
    with pytest.raises(ValueError, match="ausentes"):
        Guard().audit(q["v1"], q["v2"], q["Xe"], y)


# BUG 71 ─────────────────────────────────────────────────────────────
def test_integridade_que_falha_nao_vira_aprovado(quebrado):
    """Com 9 linhas o detector de integridade recusa (mínimo 10). Antes: APROVADO."""
    q = quebrado
    r = Guard().audit(q["v1"], q["v2"], q["Xe"].iloc[:9], q["ye"].iloc[:9])
    assert r.verdict == "INDETERMINADO", r.verdict
    camadas = [x.get("layer", "") for x in r.not_evaluated]
    assert "feature integrity" in camadas


# BUG 73 ─────────────────────────────────────────────────────────────
def test_sentinela_no_treino_nao_vira_reaprendizado(quebrado):
    q = quebrado
    Xt = q["X"].copy()
    m = np.random.RandomState(5).rand(len(Xt)) < .97
    Xt.loc[m, "a"] = -999.0
    r = Guard().audit(q["v1"], _rf(Xt, q["y"], 1), q["Xe"], q["ye"], X_train=Xt)
    assert r.verdict == "BLOQUEADO", (r.verdict, r.relearning)
    assert "a" in r.root_cause_features
    assert "a" not in r.relearning


def test_reaprendizado_legitimo_continua_reconhecido():
    """A coluna chega normal, só o alvo deixou de depender dela."""
    X, _ = _dados()
    Xe, _ = _dados(300, 1)
    r_ = np.random.default_rng(9)
    y1 = 3 * X.a + 2 * X.b + X.c + r_.normal(0, .3, len(X))
    y2 = 2 * X.b + X.c + r_.normal(0, .3, len(X))
    ye = 2 * Xe.b + Xe.c
    r = Guard().audit(_rf(X, y1), _rf(X, y2, 1), Xe, ye, X_train=X)
    assert r.verdict == "REVISAR", r.verdict
    assert "a" in r.relearning


# BUG 72 ─────────────────────────────────────────────────────────────
def test_modelo_constante_bloqueia_mesmo_com_x_train(quebrado):
    q = quebrado
    v2 = _rf(q["X"], pd.Series(np.full(len(q["y"]), q["y"].mean())), 1)
    r = Guard().audit(q["v1"], v2, q["Xe"], q["ye"], X_train=q["X"])
    assert r.verdict == "BLOQUEADO"
    assert r.raw.get("model_collapsed") is True


def test_classificador_desbalanceado_nao_e_colapso():
    """Prever sempre a maioria com `predict` não é degenerado se a probabilidade varia."""
    r_ = np.random.default_rng(2)
    X = pd.DataFrame(r_.normal(size=(800, 3)), columns=list("abc"))
    y = (X.a + r_.normal(0, 2, 800) > 3.2).astype(int)
    m = lambda s: RandomForestClassifier(n_estimators=30, max_depth=4,
                                         random_state=s, n_jobs=1).fit(X, y)
    res = Guard().audit(m(0), m(1), X, y)
    assert not res.raw.get("model_collapsed")


# BUG 68 ─────────────────────────────────────────────────────────────
def test_multiclasse_ve_variavel_que_separa_classes_0_e_2():
    """
    `a` decide classe 1 contra o resto; `b` decide 0 contra 2. Na fatia da
    classe 1, `b` pesa ~1,7% — abaixo do piso — e o colapso dela sumia.
    Cenário do revisor (t2.py).
    """
    from modelcheck.importance import ImportanceCollapseDetector
    rng = np.random.RandomState(0)
    n = 800
    X = pd.DataFrame({k: rng.randn(n) for k in "abcdefg"})
    y = np.where(X.a > 0, 1, np.where(X.b < 0, 0, 2))
    m1 = RandomForestClassifier(100, random_state=0, min_samples_leaf=5,
                                n_jobs=1).fit(X, y)
    m2 = RandomForestClassifier(100, random_state=1, min_samples_leaf=5,
                                n_jobs=1).fit(X.assign(b=0.0), y)
    r = ImportanceCollapseDetector().compare(m1, m2, X)
    b = [f for f in r["features"] if f["feature"] == "b"][0]
    assert b.get("alert") == "IMPORTANCE_COLLAPSE", b


# BUG 76 / 77 ────────────────────────────────────────────────────────
def test_combinar_nunca_e_mais_brando_que_o_guard(quebrado):
    from modelcheck.pipeline import combinar
    q = quebrado
    Xe2 = q["Xe"].copy()
    Xe2["b"] *= 100
    r = Guard().audit(q["v1"], _rf(q["X"], q["y"], 1), Xe2, q["ye"], X_train=q["X"])
    assert r.verdict == "ATENCAO"
    assert combinar(r, drifted_columns=[])["verdict"] == "ATENCAO"


def test_exit_code():
    from modelcheck.pipeline import exit_code
    assert exit_code({"verdict": "INDETERMINADO"}) == 2
    assert exit_code({"verdict": "BLOQUEADO"}, "HIGH") == 1
    assert exit_code({"verdict": "APROVADO"}) == 0
    with pytest.raises(ValueError):
        exit_code({"verdict": "BLOQUEADO"}, "altissimo")


# BUG 80 ─────────────────────────────────────────────────────────────
def test_limiar_de_frechet_chega_na_camada_de_forma():
    """Antes, Guard(threshold=...) morria no orquestrador e a forma usava 0,33."""
    r_ = np.random.default_rng(3)
    n = 600
    X = pd.DataFrame(r_.normal(size=(n, 3)), columns=["x1", "x2", "x3"])
    y = X.x1 + (X.x3 ** 2 - 1)
    y2 = X.x1 + X.x3          # x3 muda de U para linear: mudança real
    v1, v2 = _rf(X, y), _rf(X, y2, 1)
    baixo = Guard(frechet_threshold=0.33).audit(v1, v2, X, y)
    alto = Guard(frechet_threshold=50.0).audit(v1, v2, X, y)
    assert any(d["feature"] == "x3" for d in baixo.shape_divergences)
    assert alto.shape_divergences == []


def test_padrao_do_limiar_nao_mudou():
    from modelcheck.frechet import LIMIAR_PADRAO
    assert LIMIAR_PADRAO == 0.33
    assert Guard().frechet_threshold is None


# BUG 66 ─────────────────────────────────────────────────────────────
def test_curva_plana_nas_duas_versoes_nao_e_comparada():
    from modelcheck.frechet import divergencias_forma
    r_ = np.random.default_rng(0)
    n = 1000
    X = pd.DataFrame({"x": r_.normal(size=n), "z": r_.normal(size=n)})
    # atribuição grande e simétrica: média por faixa de x ~0, sem forma
    s1 = np.c_[r_.normal(0, 1, n) * np.sign(X.z), X.z * 2]
    s2 = np.c_[r_.normal(0, 1, n) * np.sign(X.z), X.z * 2]
    pulados = []
    d = divergencias_forma(X, s1, s2, ["x", "z"], _pulados=pulados)
    assert "x" not in [v["feature"] for v in d]
    assert pulados == ["x"]


def test_forma_real_continua_detectada():
    from modelcheck.frechet import divergencias_forma
    r_ = np.random.default_rng(0)
    n = 1000
    x = r_.normal(size=n)
    X = pd.DataFrame({"x": x, "z": r_.normal(size=n)})
    s1 = np.c_[x, X.z * .5]                 # linear
    s2 = np.c_[x ** 2 - 1, X.z * .5]        # U
    d = divergencias_forma(X, s1, s2, ["x", "z"])
    assert [v["feature"] for v in d] == ["x"]


# BUG 70 ─────────────────────────────────────────────────────────────
def test_permutacao_multiclasse_nao_depende_do_rotulo():
    from sklearn.linear_model import LogisticRegression
    from modelcheck.importance import _perturbation_importance
    r_ = np.random.default_rng(1)
    X = pd.DataFrame(r_.normal(size=(600, 2)), columns=["a", "b"])
    y = np.where(X.a > .4, 1, np.where(X.b > 0, 2, 0))
    rotulo = {0: 1, 1: 0, 2: 2}
    y_ren = np.vectorize(rotulo.get)(y)
    i1 = _perturbation_importance(LogisticRegression(max_iter=500).fit(X, y), X, ["a", "b"])
    i2 = _perturbation_importance(LogisticRegression(max_iter=500).fit(X, y_ren), X, ["a", "b"])
    p1, p2 = i1 / i1.sum(), i2 / i2.sum()
    assert np.allclose(p1, p2, atol=.02), (p1, p2)


def test_to_dict_e_o_que_o_laudo_recebe(quebrado):
    q = quebrado
    r = Guard().audit(q["v1"], q["v2"], q["Xe"], q["ye"])
    d = r.to_dict()
    for k in ("verdict", "skew_status", "clone_redistribution", "relearning",
              "model_diff", "not_evaluated", "verdict_driver", "ood_features"):
        assert k in d, k
    assert "BLOQUEADO" == d["verdict"]


# BUG 81 / 82 — BLOQUEADO falso em alta dimensão ─────────────────────
def _alta_dim(s, p=200, n=400):
    r_ = np.random.default_rng(s)
    Z = pd.DataFrame(r_.normal(size=(n, p)), columns=[f"c{i}" for i in range(p)])
    return Z, 3 * Z.c0 + .8 * Z.c1 + r_.normal(0, 2.0, n)


@pytest.mark.parametrize("s", [0, 1])
def test_floresta_alta_dimensao_saudavel_nao_bloqueia(s):
    """
    1.0.3: com 200 variáveis e floresta de 20 árvores, 4 de 4 retreinos
    saudáveis (só a semente muda) saíam BLOQUEADO, com 6 a 9 variáveis de
    ruído acusadas. Mesmo com 100 árvores, 2 de 4.
    """
    Z, y = _alta_dim(s)
    m = lambda k: RandomForestRegressor(n_estimators=20, max_depth=5,
                                        random_state=k, n_jobs=1).fit(Z, y)
    r = Guard().audit(m(0), m(1), Z, y)
    assert r.verdict != "BLOQUEADO", r.root_cause_features


def test_floresta_alta_dimensao_quebra_real_aponta_so_a_variavel():
    Z, y = _alta_dim(0)
    m = lambda X_, k: RandomForestRegressor(n_estimators=20, max_depth=5,
                                            random_state=k, n_jobs=1).fit(X_, y)
    r = Guard().audit(m(Z, 0), m(Z.assign(c0=0.0), 1), Z, y)
    assert r.verdict == "BLOQUEADO"
    assert r.root_cause_features == ["c0"]


def test_boosting_alta_dimensao_saudavel_nao_bloqueia():
    from sklearn.ensemble import GradientBoostingRegressor
    Z, y = _alta_dim(1)
    m = lambda k: GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                            subsample=.8, random_state=k).fit(Z, y)
    r = Guard().audit(m(0), m(1), Z, y)
    assert r.verdict != "BLOQUEADO", r.root_cause_features


def test_boosting_alta_dimensao_quebra_fraca_continua_detectada():
    from sklearn.ensemble import GradientBoostingRegressor
    Z, y = _alta_dim(0)
    m = lambda X_, k: GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                                subsample=.8, random_state=k).fit(X_, y)
    r = Guard().audit(m(Z, 0), m(Z.assign(c1=0.0), 1), Z, y)
    assert r.root_cause_features == ["c1"], r.root_cause_features


# BUG 83 — deriva natural + reaprendizado não é falha de pipeline ─────
def test_reaprendizado_com_deriva_natural_nao_bloqueia():
    """
    A primeira correção do bug 73 negava reaprendizado a qualquer coluna
    acusada pela checagem treino × uso. Em série temporal toda coluna deriva
    um pouco: dois retreinos saudáveis do ENA viraram BLOQUEADO. Deriva só de
    forma (PSI) com a coluna normal é reaprendizado; corrupção traz alerta de
    escala/média/faixa (teste da sentinela acima).
    """
    r_ = np.random.default_rng(4)
    n = 800
    Xtr = pd.DataFrame(r_.normal(size=(n, 3)), columns=list("abc"))
    Xtr["a"] = Xtr["a"] + 0.45                      # deriva natural leve
    Xe = pd.DataFrame(r_.normal(size=(400, 3)), columns=list("abc"))
    y1 = 3 * Xtr.a + 2 * Xtr.b + r_.normal(0, .3, n)
    y2 = 2 * Xtr.b + Xtr.c + r_.normal(0, .3, n)     # o alvo deixou de depender de a
    res = Guard().audit(_rf(Xtr, y1), _rf(Xtr, y2, 1), Xe, 2 * Xe.b + Xe.c,
                        X_train=Xtr)
    assert res.verdict != "BLOQUEADO", (res.verdict, res.root_cause_features)
    assert "a" in res.relearning
