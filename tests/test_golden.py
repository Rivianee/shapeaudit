"""
Testes golden: o veredito de casos fixos não pode mudar sem que alguém
mude este arquivo. Cenários sintéticos e rápidos, sem dado externo.

Se um destes falhar, o comportamento mudou: confirme que é intencional e
atualize o esperado JUNTO com a explicação no CHANGELOG.
"""
import numpy as np, pandas as pd, pytest
from conftest import RF
import modelcheck as M

ESPERADO = {
    'identicos':            ('APROVADO',  'nenhum_sinal'),
    'semente':              ('APROVADO',  'nenhum_sinal'),
    'coluna_constante':     ('BLOQUEADO', 'feature_integrity'),
    'clone_assumiu':        ('ATENCAO',   'clone_redistribution'),
    'modelo_deixou_de_usar': ('REVISAR',  'relearning'),
    'dados_mudaram':        ('ATENCAO',   'train_serving'),
}


def _base(n=900, seed=0):
    r = np.random.RandomState(seed)
    a = r.uniform(0, 1, n)
    X = pd.DataFrame({'a': a, 'a_clone': a + r.randn(n) * .01,
                      'b': r.uniform(0, 1, n), 'c': r.uniform(0, 1, n)})
    y = pd.Series(3 * X.a + 2 * X.b + r.randn(n) * .2)
    return X, y


def _caso(nome):
    X, y = _base()
    m1 = RF(0, n=50).fit(X, y)
    if nome == 'identicos':
        return M.Guard().audit(m1, m1, X, y, list(X.columns), X_train=X)
    if nome == 'semente':
        return M.Guard().audit(m1, RF(9, n=50).fit(X, y), X, y, list(X.columns), X_train=X)
    if nome == 'coluna_constante':
        Xq = X.copy(); Xq['a'] = Xq['a'].mean(); Xq['a_clone'] = Xq['a_clone'].mean()
        return M.Guard().audit(m1, RF(0, n=50).fit(Xq, y), X, y, list(X.columns), X_train=Xq)
    if nome == 'clone_assumiu':
        Xq = X.copy(); Xq['a'] = Xq['a'].mean()
        return M.Guard().audit(m1, RF(0, n=50).fit(Xq, y), X, y, list(X.columns), X_train=Xq)
    if nome == 'modelo_deixou_de_usar':
        Xq = X.copy(); Xq['a'] = Xq['a'].mean()
        # X_train com a variável presente: o modelo escolheu não usar
        return M.Guard().audit(m1, RF(0, n=50).fit(Xq, y), X, y, list(X.columns), X_train=X)
    if nome == 'dados_mudaram':
        Xs = X.copy(); Xs['b'] = Xs['b'] * 2.5 + 1
        return M.Guard().audit(m1, RF(9, n=50).fit(X, y), Xs, y, list(X.columns), X_train=X)
    raise KeyError(nome)


@pytest.mark.parametrize('nome', list(ESPERADO))
def test_veredito_golden(nome):
    r = _caso(nome)
    esperado = ESPERADO[nome]
    assert (r.verdict, r.verdict_driver.split('+')[0]) == esperado, (
        f"{nome}: esperado {esperado}, obtido "
        f"({r.verdict}, {r.verdict_driver}). Se a mudança é intencional, "
        f"atualize ESPERADO e o CHANGELOG.")
