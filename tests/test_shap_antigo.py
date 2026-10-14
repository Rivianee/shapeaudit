"""
BUG 95 (v1.0.5): shap < 0.45 devolve, em classificação, uma LISTA com uma
matriz por classe; o shap novo devolve um array (n, p, classes). Os módulos
públicos que recebem SHAP do usuário precisam dar o mesmo resultado nos dois
formatos. Apareceu no CI com as versões mínimas (shap 0.44): o fairness lia a
lista como (classes, n, p) e o universal trocava SHAP por outro método em
silêncio.

Aqui a lista é simulada a partir do array, então o teste roda em qualquer
versão do shap.
"""
import numpy as np
import pandas as pd
import pytest
import shap
from sklearn.ensemble import RandomForestClassifier

from modelcheck.fairness import FairnessChecker
from modelcheck.frechet import matriz_shap
from modelcheck.shape import ShapeDivergenceDetector
from modelcheck.universal import _repr_matrix


@pytest.fixture(scope="module")
def caso():
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.normal(size=(300, 4)), columns=["a", "b", "sexo", "d"])
    y = ((X.a + X.b + .3 * X.sexo) > 0).astype(int)
    m = RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0,
                               n_jobs=1).fit(X, y)
    s = shap.TreeExplainer(m).shap_values(X)
    arr = s if not isinstance(s, list) else np.stack(s, axis=-1)
    lista = [arr[:, :, k] for k in range(arr.shape[2])]   # formato do shap antigo
    return X, m, arr, lista


def test_matriz_shap_igual_nos_dois_formatos(caso):
    X, _, arr, lista = caso
    np.testing.assert_allclose(matriz_shap(arr), matriz_shap(lista))
    assert matriz_shap(lista).shape == X.shape


def test_fairness_aceita_lista(caso):
    X, _, arr, lista = caso
    fc = FairnessChecker(["sexo"], max_weight=.01)
    assert fc.check(arr, X) == fc.check(lista, X)


def test_fairness_nome_errado_continua_levantando(caso):
    X, _, _, lista = caso
    with pytest.raises(ValueError, match="ausentes"):
        FairnessChecker(["raca"]).check(lista, X)


def test_shape_aceita_lista(caso):
    X, _, arr, lista = caso
    d = ShapeDivergenceDetector()
    a = d.check({"v1": arr, "v2": arr}, X)
    b = d.check({"v1": lista, "v2": lista}, X)
    assert str(a) == str(b)


def test_universal_nao_troca_shap_em_silencio(caso, monkeypatch):
    X, m, arr, lista = caso

    class FakeExplainer:
        def __init__(self, *_a, **_k): pass
        def shap_values(self, Xs): return [x[:len(Xs)] for x in lista]

    monkeypatch.setattr(shap, "TreeExplainer", FakeExplainer)
    mat, metodo = _repr_matrix(m, X, list(X.columns), use_shap=True)
    assert metodo == "SHAP"
    assert mat.shape[1] == X.shape[1]
