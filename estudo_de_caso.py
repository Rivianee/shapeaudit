"""
estudo_de_caso.py — uma semana na vida de um modelo de churn.

    python estudo_de_caso.py

Dados reais: 7.043 clientes de telecom (exemplo_churn.xlsx).
Três situações que acontecem de verdade, e o que a auditoria diz em cada uma.
Ao final, três laudos em HTML para abrir no navegador.
"""
import io, os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from shapeaudit import Guard

# ── aparência ────────────────────────────────────────────────────────
COR = os.environ.get("NO_COLOR") is None and os.isatty(1)
c = lambda t, k: f"\033[{k}m{t}\033[0m" if COR else t
DIM, NEG = "2", "1"
TOM = {"BLOQUEADO": "1;31", "REVISAR": "1;33", "ATENCAO": "1;33", "APROVADO": "1;32"}
L = 68
def cena(n, t):
    print("\n" + c("─" * L, DIM))
    print(f" {c('ATO ' + str(n), DIM)}  {c(t, NEG)}")
    print(c("─" * L, DIM))
def fala(t):   print(f"  {t}")
def dado(k, v): print(f"    {c(k + ':', DIM)} {v}")

# ── os dados ─────────────────────────────────────────────────────────
print(c("\n  UMA SEMANA NA VIDA DE UM MODELO DE CHURN", NEG))
d = pd.read_excel(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "exemplo_churn.xlsx"))
y = (d.pop("churn") == "Yes").astype(int)
for col in d.columns:
    if not pd.api.types.is_numeric_dtype(d[col]):
        d[col] = pd.factorize(d[col])[0]
X = d.astype(float)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=42)
RF = lambda Xd, s=0: RandomForestClassifier(n_estimators=100, max_depth=6,
                                            random_state=s, n_jobs=1).fit(Xd, ytr)
auc = lambda m: roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
producao = RF(Xtr)
fala(f"{len(X)} clientes, {X.shape[1]} variáveis. O modelo em produção prevê "
     f"quem vai cancelar.")
dado("AUC em produção", f"{auc(producao):.3f}")
fala("Toda semana ele é retreinado, e alguém decide se a versão nova sobe.")

K = int(.10 * len(Xte))     # a campanha de retenção liga para os 10% de maior risco
def fila(m):
    return set(np.argsort(-m.predict_proba(Xte)[:, 1])[:K])
fila_atual = fila(producao)

def avaliar(titulo, v2, treino, arquivo, texto):
    r = Guard().audit(producao, v2, Xte, yte, list(X.columns), X_train=treino)
    trocados = len(fila_atual - fila(v2))
    cor = TOM.get(r.verdict, NEG)
    print()
    dado("AUC", f"{auc(producao):.3f} → {auc(v2):.3f}   "
                f"(diferença {abs(auc(producao) - auc(v2)):.4f})")
    print(f"\n  {c('▍', cor)} {c(r.verdict, cor)} — {texto}")
    if r.root_cause_features:
        print(f"  {c('▍', cor)} comece por: {c(', '.join(r.root_cause_features), NEG)}")
    mudou = [x for x in r.model_diff if x["categoria"] != "estavel"][:3]
    for x in mudou:
        extra = c("  ← forma mudou", "1;31") if x["forma_mudou"] else ""
        print(f"  {c('▍', cor)}   {x['feature']:<18}"
              f"{x['peso_v1']:>5.0%} → {x['peso_v2']:<5.0%}{x['delta_pp']:+6.1f} pp{extra}")
    print(f"  {c('▍', cor)} na campanha: {c(f'{trocados} de {K} clientes trocam', NEG)}"
          f" ({100 * trocados / K:.1f}%)")
    html = r.to_html(lang="pt", decisao={"trocados": trocados, "selecionados": K,
                                         "pct": round(100 * trocados / K, 1)})
    io.open(arquivo, "w", encoding="utf-8").write(html)
    return arquivo

saidas = []

cena(1, "Segunda: o retreino de rotina")
fala("Mesmo código, mesmos dados, só a semente do treino mudou.")
saidas.append(avaliar("rotina", RF(Xtr, 7), Xtr, "laudo_shapeaudit_1_rotina.html",
                      "nada mudou de forma relevante"))

cena(2, "Quarta: o time trocou o algoritmo")
fala("A mesma base, agora com gradient boosting. A acurácia empata.")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0).fit(Xtr, ytr)
saidas.append(avaliar("algoritmo", gb, Xtr, "laudo_shapeaudit_2_outro_algoritmo.html",
                      "mesma acurácia, comportamento diferente"))

cena(3, "Sexta: alguém mexeu no pipeline")
fala("Uma mudança no ETL fez a coluna `Contract` chegar sempre igual.")
Xq = Xtr.copy(); Xq["Contract"] = Xq["Contract"].mean()
saidas.append(avaliar("pipeline", RF(Xq), Xq, "laudo_shapeaudit_3_pipeline_quebrado.html",
                      "uma variável sumiu do modelo"))

print("\n" + c("─" * L, DIM))
print(f" {c('O QUE A MÉTRICA SOZINHA DIRIA', NEG)}")
print(c("─" * L, DIM))
fala("Nos três casos a AUC ficou praticamente igual. Um portão por métrica")
fala("aprovaria os três — inclusive o da sexta-feira.")
print()
for s in saidas:
    print(f"  {c('•', DIM)} {os.path.abspath(s)}")
print(f"\n  {c('Abra os três no navegador e compare os laudos.', DIM)}\n")
