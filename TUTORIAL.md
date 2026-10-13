# ShapeAudit — Tutorial

Todas as saídas deste documento foram **geradas executando o código**, não
escritas à mão. Dataset: Wine Quality (UCI), 1.599 amostras, 11 features.

---

## 1. Instalação

```bash
pip install shapeaudit
```

Dependências: `scikit-learn`, `pandas`, `numpy`, `scipy`, `shap`.
Opcional: `gudhi` (camada topológica).

---

## 2. O problema que o ShapeAudit resolve

Você retreinou um modelo. O R² caiu de **0.4199** para **0.4082** — uma queda
de 1,2%, dentro de qualquer margem que um time aceitaria.

```python
print(f"v1: R² = {v1.score(X_test, y_test):.4f}")   # 0.4199
print(f"v2: R² = {v2.score(X_test, y_test):.4f}")   # 0.4082
```

**Mas o v2 está quebrado.** A coluna `alcohol` virou constante no pipeline de
dados, e o modelo aprendeu a ignorá-la. Nenhuma métrica agregada mostra isso.

---

## 3. Auditoria em duas linhas

```python
from modelcheck import Guard

g = Guard()
r = g.audit(v1, v2, X_test, y_test)

print(r.verdict)   # BLOQUEADO
print(r.plain)
```

Saída real:

```
Resultado: BLOQUEADO

Uma ou mais variáveis deixaram de influenciar o modelo novo. Esse padrão é
típico de falha no pipeline de dados (coluna virou nula, join perdido, filtro
novo) e não de retreinamento normal.

Comece por: 'alcohol' — esta variável deixou de influenciar o modelo.
Também mudaram, possivelmente como consequência: 'volatile acidity',
'residual sugar', 'density'.

Em números: 4 variável(is) mudaram de padrão; 6 mudaram de intensidade;
a estrutura geral de decisão mudou.

Recomendação: Não promover. Uma ou mais variáveis deixaram de influenciar
o modelo — verifique o pipeline de dados antes de qualquer retreino.

Não verificado nesta auditoria:
  · train/serving consistency — não recebemos os dados de treino do modelo
    novo (parâmetro X_train)
  · class representation — a classe 3 tem apenas 1 exemplos na avaliação
  · fairness — não verificamos se o erro do modelo mudou para algum grupo
    de pessoas
  (o veredito acima não cobre estes pontos)
```

(Saída da v1.0.4, conferida rodando o script da §11.)

**Note a ordem.** `alcohol` — a causa raiz — vem primeiro, com "Comece por".
As outras três mudaram porque o modelo redistribuiu peso; são consequência.
Sem essa ordenação você investigaria quatro variáveis, começando pela errada.

### Campos disponíveis

```python
r.verdict               # APROVADO | ATENCAO | REVISAR | BLOQUEADO
r.problem_type          # IMPORTANCE_COLLAPSE
r.root_cause_features   # ['alcohol']            ← comece aqui
r.affected_features     # ['alcohol', 'residual sugar', 'chlorides', 'pH']
r.plain                 # o texto acima
r.plain_debug()         # idem, com o índice de saúde (é um método)
```

---

## 4. O contraste: um retreino saudável

Mesmo código, mas com dois modelos treinados nos mesmos dados (só a semente
muda):

```python
r = g.audit(v1, v1_outra_semente, X_test, y_test)
print(r.verdict)            # APROVADO
print(r.problem_type)       # NONE
print(r.affected_features)  # []
```

**Retreino saudável não é bloqueado.** Medido em 164 retreinos reais do
ONS e nos testes da biblioteca: nenhum BLOQUEADO falso. A camada de forma,
porém, **pode** pedir REVISAR num retreino saudável — no `demo_visual.py`,
mudando só a semente, `total sulfur dioxide` dá distância 0,334, logo acima
do limiar padrão de 0,33. Calibre o limiar no seu domínio (`calibrar_limiar`
e `Guard(frechet_threshold=...)`).

---

## 5. Retreino temporal: use `check_features`

Se o seu pipeline retreina em janela deslizante, **não use o índice de
saúde** — ele é dominado pela fração de dados compartilhada entre as versões
(ρ=+0.54, ver `LIMITATIONS.md §3`).

Declare as variáveis cuja integridade é crítica:

```python
c = g.check_features(v1, v2, X_test,
                     monitored=['alcohol', 'sulphates'])

print(c['alert'])      # True
print(c['collapsed'])  # ['alcohol']
print(c['plain'])
```

```
Falha de integridade: 'alcohol' deixou de influenciar o modelo novo.
Verifique o pipeline de dados dessas variáveis (coluna nula, join perdido,
filtro novo).
```

**Por que este é o modo mais forte:** F1=0.99 com especificidade 100%, medido
em 108 pares reais em quatro níveis de sobreposição de janela — inclusive
com janelas completamente disjuntas, onde o índice de saúde falha.

Monitore as features que vêm de fonte frágil: joins, APIs externas, ETL.
**Monitorar menos é mais seguro.**

---

## 6. O modelo está certo, os dados chegam errados

Existe uma classe de falha que nenhuma comparação entre modelos detecta: as
duas versões são internamente coerentes, e a diferença está entre o que o
modelo **espera** receber e o que **recebe**.

```python
from modelcheck import TrainServingValidator
from modelcheck.validation import plain_report

# alcohol chega 35% maior na inferência (unidade errada)
v = TrainServingValidator()
r = v.check(X_train, X_serving)

print(r['flagged'])   # ['alcohol']
print(r['alerts'])    # ['MEAN_SHIFT', 'OUT_OF_RANGE', 'PSI_ALERT', 'SCALE_CHANGE']
print(plain_report(r))
```

```
Inconsistência entre treino e inferência: 'alcohol' chega numa escala
diferente da que o modelo viu no treino.

Isso indica problema no pipeline de dados, não no modelo. Verifique como
essas variáveis são calculadas nos dois momentos (agregação, unidade,
imputação).
```

Custa milissegundos — só estatísticas descritivas e PSI, sem SHAP.

**Cuidado ao interpretar:** num split temporal, esta camada vai alertar se
houver deriva real nos dados. Testado com split aleatório do mesmo período:
zero alerta. Com split temporal de uma série com crise hídrica: alertou
corretamente (KS p=3×10⁻²¹¹). **Se alertar, verifique se a deriva é real
antes de considerar falso positivo.**

---

## 7. Integração com MLflow e Slack

```python
import mlflow
mlflow.log_metrics(r.to_mlflow())
mlflow.set_tags(r.to_tags())
```

```python
{'modelguard.health_score': 47.8,
 'modelguard.shape_divergences': 4.0,
 'modelguard.slope_alerts': 6.0,
 'modelguard.canary_alerts': 2.0}
```

```python
requests.post(WEBHOOK, json=r.to_slack('wine-quality'))
```

```
:no_entry: ShapeAudit — wine-quality: *BLOQUEADO*
Uma ou mais variáveis deixaram de influenciar o modelo novo...
```

---

## 8. Qual modo usar

| Sua situação | Use | Métrica medida |
|---|---|---|
| Trocou arquitetura/hiperparâmetro, **mesmo treino** | `audit(mode="retraining")` | F1=0.96 |
| Comparando arquiteturas de propósito | `audit(mode="exploratory")` | diferenças reportadas como esperadas |
| **Retreino temporal / janela deslizante** | `check_features(monitored=[...])` | **F1=0.99, espec. 100%** |
| Suspeita de problema no pipeline | `TrainServingValidator` | 6/6 skews, 0 FP |
| Quer prever se o modelo vai degradar | **nenhum** — refutado (n=1.128) | — |

---

## 9. Sequência de versões

```python
rel = g.audit_versions([v1, v2, v3, v4, v5], X_test, y_test,
                       labels=['jan', 'fev', 'mar', 'abr', 'mai'])
print(rel['summary'])
print(rel['trend'])       # estável | piorando | melhorando
```

**Mínimo de 5 versões (4 transições).** Com 3 transições a taxa de falso
positivo da tendência é de **34,2%** — medido em 3.000 séries aleatórias — e
por isso a API bloqueia. Poder de detecção: ~19% com 4 transições, ~86% com 8.

---

## 10. Antes de confiar no resultado

Três coisas que valem saber:

**O índice de saúde é o sinal mais fraco.** Ele aparece no rodapé do relatório
por decisão de projeto. Cinco medições independentes mostraram que é dominado
pela configuração do pipeline, satura perto de zero, e num caso medido
**inverteu a ordem** (retreino legítimo pontuou pior que pipeline quebrado).
Use `problem_type` e `root_cause_features`.

**Degradação parcial abaixo de ~40% não é detectada.** Dano real de
ΔR²=−0.08 passa sem alerta. Limite medido em três sistemas.

**Os limiares podem precisar de calibração no seu domínio.** O
`PSIShapDetector` emite aviso se usado sem calibrar; o
`ImportanceCollapseDetector` tem `calibrate()`. Medido: o p99 do PSI varia de
0.38 a 1.35 entre dois subsistemas do **mesmo** dataset.

`LIMITATIONS.md` tem as 14 seções completas, todas com número medido.

---

## 10-bis. Caso real: transição energética (dados do ONS)

> **Resultado de versão anterior.** Este caso foi medido quando o veredito
> ainda usava o índice de saúde, removido na v0.11. Combinações como
> "BLOQUEADO — MIXED" sem nenhuma variável colapsada não acontecem mais na
> 1.0.4: hoje BLOQUEADO exige colapso de variável ou modelo degenerado. Os
> dados deste caso não estão no repositório, então ele fica como narrativa;
> os números atuais estão em `LIMITATIONS.md` (seções 7-bis a 7-sexies).


Um exemplo em que **as três camadas dizem coisas diferentes, e as três estão
certas**. Dados: geração horária por tipo de usina no Sudeste, ONS, 2025 e
2026. Alvo: geração hidrelétrica.

### A matriz mudou de verdade

| tipo | 2025 | 2026 | variação |
|---|---|---|---|
| Fotovoltaica | 4.415 MW | 5.713 MW | **+29%** |
| Eólica | 4 MW | 59 MW | **+1.581%** |
| Térmica | 4.302 MW | 4.424 MW | +3% |
| Hidrelétrica | 26.294 MW | 28.422 MW | +8% |

R² praticamente igual: 0.558 (v1) e 0.546 (v2).

### O que cada camada diz

```python
r = Guard().audit(v1, v2, X_eval, y_eval)
# BLOQUEADO — MIXED
# 'TÉRMICA', 'NUCLEAR', 'mes' mudaram de padrão
```

```python
c = Guard().check_features(v1, v2, X_eval,
                           monitored=['FOTOVOLTAICA', 'TÉRMICA'])
# alerta=False — nenhuma feature perdeu importância
```

```python
TrainServingValidator().check(X_2025, X_2026)
# flagged: FOTOVOLTAICA, EOLIELÉTRICA, TÉRMICA, NUCLEAR, mes
# alerts: MEAN_SHIFT, OUT_OF_RANGE, PSI_ALERT, SCALE_CHANGE
```

### Como ler isso

**Não é contradição — é diagnóstico diferencial.**

O `check_features` diz que **nenhuma variável sumiu do modelo**: não é falha
de pipeline. A 4ª camada diz que **a distribuição dos dados mudou**: a eólica
saiu de 4 para 59 MW médios, a solar cresceu 29%. E o `audit` confirma que o
modelo **mudou como usa** as variáveis, em resposta a essa mudança real.

A conclusão prática é diferente das duas anteriores do tutorial: aqui o v2
**não** deve ser rejeitado — ele está certo, o mundo é que mudou. O que a
combinação diz é *"retreine, mas revalide os limiares operacionais, porque a
relação entre térmica e hidrelétrica não é mais a mesma"*.

**A regra que emerge:**

| check_features | 4ª camada | leitura |
|---|---|---|
| alerta | silencioso | falha de pipeline — conserte o ETL |
| silencioso | alerta | mudança real nos dados — retreine e recalibre |
| alerta | alerta | investigue os dois |
| silencioso | silencioso | retreino saudável |

---

## 10-ter. Caso real: crise hídrica de 2021 — quando o retreino é a resposta certa

> **Resultado de versão anterior.** Este caso foi medido quando o veredito
> ainda usava o índice de saúde, removido na v0.11. Combinações como
> "BLOQUEADO — MIXED" sem nenhuma variável colapsada não acontecem mais na
> 1.0.4: hoje BLOQUEADO exige colapso de variável ou modelo degenerado. Os
> dados deste caso não estão no repositório, então ele fica como narrativa;
> os números atuais estão em `LIMITATIONS.md` (seções 7-bis a 7-sexies).


O terceiro caso, e o único em que a ferramenta **aprova** o retreino. Dados:
CMO e EAR do Sudeste, ONS, 2012–2022. Alvo: custo marginal de operação.

### O evento

| ano | EAR médio | CMO médio |
|---|---|---|
| 2020 | 38,6% | R$ 176/MWh |
| **2021** | **25,7%** | **R$ 548/MWh** |
| 2022 | 55,8% | R$ 31/MWh |

A pior seca em 91 anos. O CMO triplicou.

### O experimento

`v1` treinado em 2012–2020 (**sem** a crise). `v2` em 2013–2022 (**com**).

```python
r = Guard().audit(v1, v2, X_eval, y_eval)
# APROVADO — pode seguir para produção

Guard().check_features(v1, v2, X_eval, monitored=['ear', 'cmo_lag1'])
# alerta=False

TrainServingValidator().check(X_2012_2020, X_2013_2022)
# flagged=[]  alerts=[]
```

**As três camadas silenciosas.** E o R² subiu de 0.699 para 0.878.

### Por que isso importa

O erro do modelo **durante a crise**:

```
v1 (sem a crise no treino): RMSE = R$ 645/MWh
v2 (com a crise no treino): RMSE = R$ 311/MWh
```

O v2 erra **metade**. Incorporar o evento extremo melhorou o modelo em tudo, e
a ferramenta reconhece isso: não há colapso de importância, não há mudança de
distribuição entre as janelas, e a estrutura de decisão continua coerente.

O único sinal foi `há divergência em cenários extremos` — que é literalmente
verdade e é o comportamento desejado: o v2 aprendeu a região que o v1 não
conhecia.

### O contraste com os outros dois casos

| caso | veredito | leitura |
|---|---|---|
| pipeline quebrado | BLOQUEADO | **conserte o ETL** |
| pandemia (estudo anterior, fora deste tutorial) | ATENÇÃO | mundo mudou — retreine e recalibre |
| transição energética | BLOQUEADO | mudança estrutural grande — revalide |
| **crise hídrica** | **APROVADO** | **o retreino foi a coisa certa** |

Uma ferramenta que só sabe reprovar não serve. **Reconhecer o retreino bom é
tão importante quanto pegar o ruim** — e é o que sustenta a confiança para
que o alerta seja levado a sério quando aparece.

---

## 11. Reproduzir este tutorial

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from modelcheck import Guard, TrainServingValidator

df = pd.read_csv('winequality-red.csv', sep=';')
X, y = df.drop('quality', axis=1), df['quality']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

mk = lambda Xd: RandomForestRegressor(
    n_estimators=100, max_depth=8, random_state=42).fit(Xd, y_train)

v1 = mk(X_train)

X_quebrado = X_train.copy()
X_quebrado['alcohol'] = X_quebrado['alcohol'].mean()   # pipeline quebrado
v2 = mk(X_quebrado)

r = Guard().audit(v1, v2, X_test, y_test)
print(r.verdict, '|', r.root_cause_features)
print(r.plain)
```

Dataset: <https://archive.ics.uci.edu/dataset/186/wine+quality>
