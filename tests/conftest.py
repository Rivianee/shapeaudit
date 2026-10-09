"""Fixtures compartilhadas da suite de regressão do ModelGuard."""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer, make_friedman1

RF = lambda s=0, **kw: RandomForestRegressor(
    n_estimators=kw.pop('n', 50), max_depth=5, random_state=s, n_jobs=1, **kw)
RFC = lambda s=0: RandomForestClassifier(
    n_estimators=50, max_depth=5, random_state=s, n_jobs=1)


def _split(X, y, ts=0.4):
    return train_test_split(X, y, test_size=ts, random_state=42)


@pytest.fixture(scope='session')
def diabetes():
    o = load_diabetes()
    return pd.DataFrame(o.data, columns=o.feature_names), pd.Series(o.target)


@pytest.fixture(scope='session')
def breast():
    o = load_breast_cancer()
    X = pd.DataFrame(o.data, columns=[c.replace(' ', '_') for c in o.feature_names])
    return X.iloc[:, :10], pd.Series(o.target)


@pytest.fixture(scope='session')
def friedman():
    Xa, ya = make_friedman1(n_samples=800, n_features=8, noise=1.0, random_state=0)
    return pd.DataFrame(Xa, columns=[f'x{i}' for i in range(8)]), pd.Series(ya)


@pytest.fixture(scope='session')
def toy():
    """Dataset pequeno e determinístico para testes de contrato."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.randn(400), 'b': rng.randn(400),
                      'c': rng.randn(400)})
    y = X.a * 3 + X.b * 1.5 + rng.randn(400) * 0.3
    return X, y


def make_pair(X, y, corrupt=None, kind='neutralize', seed2=0, clf=False):
    """
    Devolve (m1, m2, Xte, yte, alvo). corrupt=None → controle.
    kind: 'neutralize' | 'shuffle' | 'seed' | 'poison'
    """
    Xtr, Xte, ytr, yte = _split(X, y)
    M = RFC if clf else RF
    m1 = M(0).fit(Xtr, ytr)
    if corrupt is None:
        m2 = M(seed2).fit(Xtr, ytr)
        return m1, m2, Xte, yte, None
    tgt = (corrupt if isinstance(corrupt, str)
           else pd.Series(m1.feature_importances_, index=X.columns).idxmax())
    Xc, yc = Xtr.copy(), ytr.copy()
    if kind == 'neutralize':
        Xc[tgt] = Xc[tgt].mean()
    elif kind == 'shuffle':
        Xc[tgt] = np.random.RandomState(0).permutation(Xc[tgt].values)
    elif kind == 'poison':
        msk = Xtr[tgt] > Xtr[tgt].quantile(0.75)
        yc.loc[Xtr[msk].index] = (1 - yc.loc[Xtr[msk].index] if clf
                                  else ytr.max() + ytr.min() - yc.loc[Xtr[msk].index])
    m2 = M(0).fit(Xc, yc)
    return m1, m2, Xte, yte, tgt
