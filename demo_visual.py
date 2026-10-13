"""
demo_visual.py — ModelGuard do início ao fim, com saída bonita.

    python demo_visual.py

Treina dois modelos nos dados de vinho (UCI, embutidos em demo_vinho.py),
audita três cenários e gera um laudo HTML para cada um.
"""
import io, os, time, warnings, webbrowser
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from shapeaudit import Guard
from shapeaudit.laudo import render

# ── cores de terminal ────────────────────────────────────────────────
USA_COR = os.environ.get("NO_COLOR") is None and os.isatty(1)
def c(txt, cod):
    return f"\033[{cod}m{txt}\033[0m" if USA_COR else txt
NEG, DIM = "1", "2"
TOM = {"BLOQUEADO": "1;31", "REVISAR": "1;33", "ATENCAO": "1;33",
       "APROVADO": "1;32", "INDETERMINADO": "1;35"}
L = 64

def cabecalho(t):
    print("\n" + c("━" * L, DIM)); print(c(f" {t}", NEG)); print(c("━" * L, DIM))

def veredito(r, titulo):
    cor = TOM.get(r.verdict, NEG)
    print(f"\n  {c('▍', cor)} {titulo}")
    print(f"  {c('▍', cor)} {c(r.verdict, cor)}   {c(r.verdict_driver, DIM)}")
    if r.root_cause_features:
        print(f"  {c('▍', cor)} comece por: {c(', '.join(r.root_cause_features), NEG)}")
    for d in r.shape_divergences[:4]:
        a, b = d.get("shape_v1"), d.get("shape_v2")
        seta = "curva diferente, mesma direção" if d.get("mesma_categoria") else f"{a} → {b}"
        print(f"  {c('▍', cor)}   {d['feature']:<22}{seta}  {c('d=' + str(d['frechet']), DIM)}")
    nao = [i["layer"] for i in r.not_evaluated]
    if nao:
        print(f"  {c('▍', cor)} {c('não verificado: ' + ', '.join(nao), DIM)}")

def laudo(r, nome, n):
    # v1.0.4: montava um dicionário próprio e parcial (o padrão do bug 55) —
    # o laudo dizia "não verificado" para uma checagem que rodou e não tinha
    # a seção do model diff. Usa o laudo da própria biblioteca.
    html = r.to_html("pt")
    open(nome, "w", encoding="utf-8").write(html)
    return nome

# ── dados e modelos ──────────────────────────────────────────────────
cabecalho("ShapeAudit — auditoria entre duas versões")
# dados embutidos em demo_vinho.py (só o carregador, sem rodar a demo dele)
_src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "demo_vinho.py"), encoding="utf-8").read()
_ns = {}
exec(_src.split("# ═")[0], _ns)
df = _ns["carregar"]()
X, y = df.drop("quality", axis=1), df["quality"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=42)
print(f"\n  Wine Quality (UCI): {len(df)} vinhos, {X.shape[1]} medidas")
print(f"  treino {len(Xtr)}  |  avaliação {len(Xte)}")

RF = lambda Xd, s=42: RandomForestRegressor(n_estimators=100, max_depth=6,
                                            random_state=s, n_jobs=1).fit(Xd, ytr)
v1 = RF(Xtr)
gb = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42).fit(Xtr, ytr)
Xq = Xtr.copy(); Xq["alcohol"] = Xq["alcohol"].mean()

cenarios = [
    ("Retreino saudável — só muda a semente", RF(Xtr, 7), Xtr, "laudo_saudavel.html"),
    ("Pipeline quebrado — 'alcohol' chega constante", RF(Xq), Xq, "laudo_quebrado.html"),
    ("Outro algoritmo, mesma acurácia (RF x GB)", gb, Xtr, "laudo_outro_modelo.html"),
]
cabecalho("Três cenários")
saidas = []
for titulo, v2, tr, arq in cenarios:
    t0 = time.time()
    r = Guard().audit(v1, v2, Xte, yte, list(X.columns), X_train=tr)
    veredito(r, titulo)
    print(f"  {c('▍', TOM.get(r.verdict, NEG))} {c(f'{time.time()-t0:.1f}s', DIM)}")
    saidas.append(laudo(r, arq, len(Xte)))

cabecalho("Laudos gerados")
for s in saidas:
    print(f"  {c('•', DIM)} {os.path.abspath(s)}")
print(f"\n  {c('Abra qualquer um no navegador: as curvas de antes e depois', DIM)}")
print(f"  {c('estão desenhadas ao lado de cada variável.', DIM)}\n")
if os.environ.get("ABRIR"):
    webbrowser.open("file://" + os.path.abspath(saidas[1]))
