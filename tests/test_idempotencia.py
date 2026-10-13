"""
Mesma entrada, mesma saída — byte a byte (v1.0.4).

A biblioteca não tem parâmetro de semente: todo sorteio interno usa semente
fixa. Se algum estado escondido entrar (cache, sorteio sem semente, ordem
de dicionário), estes testes quebram. Verificado também entre processos
separados, que é onde estado global costuma aparecer.
"""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from shapeaudit import Guard


def _caso():
    r = np.random.default_rng(0)
    X = pd.DataFrame(r.normal(size=(400, 4)), columns=list("abcd"))
    y = 3 * X.a + 2 * X.b + X.c + r.normal(0, .3, 400)
    m = lambda X_, s: RandomForestRegressor(n_estimators=30, max_depth=6,
                                            random_state=s, n_jobs=1).fit(X_, y)
    Xq = X.assign(a=0.0)
    return m(X, 0), m(Xq, 1), X, y, Xq


def _assinatura():
    v1, v2, X, y, Xq = _caso()
    r = Guard().audit(v1, v2, X, y, X_train=Xq)
    d = r.to_dict()
    return json.dumps(d, sort_keys=True, default=str), r.to_html()


def test_duas_vezes_no_mesmo_processo():
    a, b = _assinatura(), _assinatura()
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_entre_processos():
    codigo = ("import sys, hashlib; sys.path.insert(0, %r); "
              "from test_idempotencia import _assinatura; "
              "d, h = _assinatura(); "
              "print(hashlib.sha256(d.encode()).hexdigest(), "
              "hashlib.sha256(h.encode()).hexdigest())") % __import__("os").path.dirname(__file__)
    saidas = {subprocess.run([sys.executable, "-W", "ignore", "-c", codigo],
                             capture_output=True, text=True, check=True).stdout.strip()
              for _ in range(2)}
    assert len(saidas) == 1, saidas
