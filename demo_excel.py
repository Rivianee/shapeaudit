"""
demo_excel.py — ModelGuard com a SUA planilha.

    python demo_excel.py                      # usa exemplo_churn.xlsx
    python demo_excel.py meus_dados.xlsx alvo # usa a sua

O que a planilha precisa ter:
  • uma linha por caso, uma coluna por variável;
  • uma coluna com o que se quer prever (o "alvo");
  • texto é convertido em código automaticamente.

O script treina duas versões do modelo, audita as duas e escreve o laudo
em HTML. Troque a parte marcada por SEUS modelos quando for usar de verdade.
"""
import sys, warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.model_selection import train_test_split

from shapeaudit import Guard

ARQ = sys.argv[1] if len(sys.argv) > 1 else "exemplo_churn.xlsx"
ALVO = sys.argv[2] if len(sys.argv) > 2 else "churn"

print(f"\n1. Lendo {ARQ}")
d = pd.read_excel(ARQ)
if ALVO not in d.columns:
    print(f"   coluna '{ALVO}' não existe. Colunas: {list(d.columns)}"); sys.exit(2)
y = d.pop(ALVO)
print(f"   {len(d)} linhas, {d.shape[1]} variáveis, alvo '{ALVO}'")

# texto vira código; o alvo vira 0/1 quando for Sim/Não
for c in d.columns:
    if not pd.api.types.is_numeric_dtype(d[c]):
        d[c] = pd.factorize(d[c])[0]
classificacao = y.dtype == object or y.nunique() <= 10
if y.dtype == object:
    y = pd.factorize(y)[0]
X = d.astype(float).reset_index(drop=True)
y = pd.Series(y).reset_index(drop=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.4, random_state=42)

print("2. Treinando duas versões  (substitua por SEUS dois modelos)")
RF = RandomForestClassifier if classificacao else RandomForestRegressor
GB = GradientBoostingClassifier if classificacao else GradientBoostingRegressor
v1 = RF(n_estimators=100, max_depth=6, random_state=0, n_jobs=1).fit(Xtr, ytr)
v2 = GB(n_estimators=100, max_depth=4, random_state=0).fit(Xtr, ytr)
print(f"   versão 1: {v1.score(Xte, yte):.3f}   versão 2: {v2.score(Xte, yte):.3f}"
      f"   (diferença {abs(v1.score(Xte, yte) - v2.score(Xte, yte)):.4f})")

print("3. Auditando")
r = Guard().audit(v1, v2, Xte, yte, list(X.columns), X_train=Xtr)
print(f"\n   VEREDITO: {r.verdict}   ({r.verdict_driver})")
if r.root_cause_features:
    print(f"   comece por: {', '.join(r.root_cause_features)}")

mudou = [x for x in r.model_diff if x['categoria'] != 'estavel'][:6]
if mudou:
    print("\n   o que mudou:")
    for x in mudou:
        forma = "  ← forma mudou" if x['forma_mudou'] else ""
        print(f"     {x['feature']:<20}{x['peso_v1']:>5.0%} → {x['peso_v2']:<5.0%}"
              f"{x['delta_pp']:+6.1f} pp{forma}")

saida = "laudo_shapeaudit.html"
open(saida, "w", encoding="utf-8").write(r.to_html())
print(f"\n4. Laudo escrito em {saida} — abra no navegador.\n")
