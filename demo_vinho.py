"""
demo_vinho.py — ModelGuard ao vivo, com dados reais.

Wine Quality (UCI): 1.599 vinhos tintos portugueses, 11 medidas
fisico-quimicas, nota de 0 a 10 dada por provadores.

Este e o caso do artigo. Nao precisa de internet nem de arquivo externo:
os dados vao embutidos.

    python demo_vinho.py
"""
import base64, io, time, warnings, zlib
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

L = 66


def titulo(t):
    print("\n" + "=" * L)
    print(f"  {t}")
    print("=" * L)


def passo(n, t):
    print(f"\n  [{n}] {t}")


def pausa(s=0.6):
    time.sleep(s)


# ── os dados, embutidos ──────────────────────────────────────────────
_D = (
"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"
)


def carregar():
    return pd.read_csv(io.StringIO(
        zlib.decompress(base64.b64decode("".join(_D))).decode()))


# ═════════════════════════════════════════════════════════════════════
titulo("MODELGUARD - Wine Quality (UCI), 1.599 vinhos")

passo(1, "Carregando os dados")
df = carregar()
X = df.drop("quality", axis=1)
y = df["quality"]
print(f"      {len(df)} vinhos, {X.shape[1]} medidas fisico-quimicas")
print(f"      alvo: nota de qualidade ({y.min():.0f} a {y.max():.0f})")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=42)
print(f"      treino {len(Xtr)}  |  avaliacao {len(Xte)}")
pausa()

passo(2, "Modelo A: Random Forest")
t0 = time.time()
rf = RandomForestRegressor(n_estimators=100, max_depth=6,
                           random_state=42, n_jobs=1).fit(Xtr, ytr)
ra = rf.score(Xte, yte)
print(f"      R2 = {ra:.4f}   ({time.time()-t0:.1f}s)")
pausa()

passo(3, "Modelo B: Gradient Boosting")
t0 = time.time()
gb = GradientBoostingRegressor(n_estimators=100, max_depth=4,
                               random_state=42).fit(Xtr, ytr)
rb = gb.score(Xte, yte)
print(f"      R2 = {rb:.4f}   ({time.time()-t0:.1f}s)")
pausa()

titulo(f"DIFERENCA DE R2:  {abs(ra-rb):.4f}")
print("\n  Menos de 1%. Em qualquer revisao de codigo os dois")
print("  passariam como equivalentes.")
print("\n  Qual voce colocaria em producao?")
pausa(1.2)

passo(4, "Auditando as duas versoes")
from shapeaudit import Guard

t0 = time.time()
r = Guard().audit(rf, gb, Xte, yte, X_train=Xtr)
dt = time.time() - t0
print(f"      {dt:.1f}s")

divs = r.raw.get("layer1", {}).get("sdd", {}).get("divs", [])

titulo("O QUE OS DOIS MODELOS APRENDERAM DIFERENTE")
if divs:
    print()
    for d in divs:
        a, b = d.get("shape_v1"), d.get("shape_v2")
        nota = "mesma direcao, curva diferente" if d.get("mesma_categoria") \
            else f"{a} -> {b}"
        print(f"    {d['feature']:<22} {nota:<32} distancia {d.get('frechet')}")
    print(f"\n  {len(divs)} de {X.shape[1]} variaveis aprendidas de forma diferente.")
    print("\n  Mesmo R2. Mesmos dados. Curvas diferentes: para o mesmo")
    print("  vinho, os dois modelos reagem de forma diferente a cada medida.")
else:
    print("\n  Nenhuma divergencia de forma.")

print(f"\n  Veredito: {r.verdict}")
pausa()

# ── o contraste: pipeline quebrado ───────────────────────────────────
passo(5, "Agora um caso diferente: o pipeline quebrou")
print("      a coluna `alcohol` virou constante")
Xq = Xtr.copy()
Xq["alcohol"] = Xq["alcohol"].mean()
v2 = RandomForestRegressor(n_estimators=100, max_depth=6,
                           random_state=42, n_jobs=1).fit(Xq, ytr)
print(f"      R2 = {v2.score(Xte, yte):.4f}  (era {ra:.4f})")

rq = Guard().audit(rf, v2, Xte, yte, X_train=Xq)   # treino do modelo NOVO
titulo(f"VEREDITO:  {rq.verdict}")
print(f"\n  Comece por: {rq.root_cause_features}")
print(f"  Decidido por: {getattr(rq, 'verdict_driver', '-')}")
pausa()

# ── e um retreino saudavel ───────────────────────────────────────────
passo(6, "E um retreino saudavel, so mudando a semente")
ok = RandomForestRegressor(n_estimators=100, max_depth=6,
                           random_state=7, n_jobs=1).fit(Xtr, ytr)
ro = Guard().audit(rf, ok, Xte, yte, X_train=Xtr)
print(f"\n      VEREDITO: {ro.verdict}   causa: {ro.root_cause_features}")
print("\n      Uma ferramenta que so sabe reprovar")
print("      e desligada em duas semanas.")
print()
