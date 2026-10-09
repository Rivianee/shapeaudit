"""
modelguard — nome público do pacote distribuído como `modelguard-ml`.

Este módulo reexporta `modelcheck`, que é o nome interno de implementação.
Os dois funcionam:

    import modelguard          # nome do produto
    import modelcheck          # nome histórico, mantido por compatibilidade

A divergência é histórica: a biblioteca nasceu como `modelcheck` e o produto
passou a se chamar ModelGuard. Renomear o módulo interno quebraria código de
quem já usa, então mantemos os dois apontando para a mesma implementação.
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
try:                                   # versão vem do pacote instalado
    from importlib.metadata import version as _v
    __version__ = _v("modelguard-ml")
except Exception:                       # rodando do repositório, sem instalar
    __version__ = getattr(_mc, "__version__", "0.0.0+dev")

# submódulos acessíveis por `modelguard.snapshot`, `from modelguard.laudo
# import render`, etc. — sem isto, só o atributo funcionaria, e o import
# direto falharia.
for _nome in ("snapshot", "laudo", "pipeline", "cli", "audit", "shape",
              "importance", "validation", "fairness"):
    try:
        _sub = __import__(f"modelcheck.{_nome}", fromlist=[_nome])
        _sys.modules[f"modelguard.{_nome}"] = _sub
        globals()[_nome] = _sub
    except ImportError:
        pass                      # noqa: lacuna-nao-aplicavel — opcional
