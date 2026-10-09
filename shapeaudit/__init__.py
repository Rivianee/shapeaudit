"""
shapeaudit — o pacote publicado no PyPI como `shapeaudit`.

Reexporta `modelcheck`, o nome interno de implementação. Os três funcionam:

    import shapeaudit      # nome atual
    import modelguard      # nome anterior, mantido
    import modelcheck      # nome original, mantido

A divergência é histórica: a biblioteca nasceu como `modelcheck`, foi
publicada como `modelguard-ml` e hoje se chama ShapeAudit. Renomear o módulo
interno quebraria quem já instalou, então os três apontam para a mesma
implementação.
"""
import sys as _sys

import modelcheck as _mc
from modelcheck import *              # noqa: F401,F403
from modelcheck import (              # noqa: F401
    Guard, snapshot, laudo, pipeline,
)

__all__ = getattr(_mc, "__all__", []) + [
    "Guard", "snapshot", "laudo", "pipeline",
]
try:
    from importlib.metadata import version as _v
    __version__ = _v("shapeaudit")
except Exception:
    __version__ = getattr(_mc, "__version__", "0.0.0+dev")

for _nome in ("snapshot", "laudo", "pipeline", "cli", "audit", "shape",
              "frechet", "importance", "validation", "fairness"):
    try:
        _sub = __import__(f"modelcheck.{_nome}", fromlist=[_nome])
        _sys.modules[f"shapeaudit.{_nome}"] = _sub
        globals()[_nome] = _sub
    except ImportError:
        pass                      # noqa: lacuna-nao-aplicavel — opcional
