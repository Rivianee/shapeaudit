"""
Etapa 4 — Performance e escalabilidade.

Limites derivados das medições da v1.28. Marcados como `slow` para não
travar o ciclo rápido de CI: rodar com `pytest -m slow`.
"""
import warnings, time, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
from conftest import RF
from modelcheck.audit import audit
from modelcheck.api import Guard as ModelGuard


def _synth(n, p, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.randn(n, p), columns=[f'f{i}' for i in range(p)])
    y = X.f0 * 3 + X.f1 * 2 + rng.randn(n) * .3
    return X, y


@pytest.mark.slow
@pytest.mark.parametrize('p,limite_s', [(50, 30), (200, 60)])
def test_audit_shap_escala(p, limite_s):
    X, y = _synth(3000, p)
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['f0'] = Xc['f0'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    t = time.time()
    audit(m1, m2, X.sample(400, random_state=1), verbose=False)
    dt = time.time() - t
    assert dt < limite_s, f"audit SHAP com p={p}: {dt:.1f}s > {limite_s}s"


@pytest.mark.slow
@pytest.mark.parametrize('p', [50, 200])
def test_check_features_lazy_tempo_constante(p):
    """
    check_features monitorando 2 features deve ter tempo ~constante em p.
    Medido na v1.28: 25s (eager) -> 1.9s (lazy) com p=200.
    """
    X, y = _synth(1200, p)
    Xc = X.copy(); Xc['f0'] = Xc['f0'].mean()
    g = ModelGuard()
    t = time.time()
    g.check_features(Ridge().fit(X, y), Ridge().fit(Xc, y),
                     X.sample(300, random_state=1),
                     monitored=['f0'], feature_names=list(X.columns))
    dt = time.time() - t
    assert dt < 15, f"check_features lazy com p={p}: {dt:.1f}s (esperado <15s)"


@pytest.mark.slow
def test_muitas_amostras():
    X, y = _synth(50000, 20)
    m1 = RF(0, n=30).fit(X, y)
    t = time.time()
    r = audit(m1, m1, X.sample(500, random_state=1), verbose=False)
    dt = time.time() - t
    assert r['health_score'] == 100
    assert dt < 60, f"50k amostras: {dt:.1f}s"
