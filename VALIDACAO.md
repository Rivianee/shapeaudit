# Validação do modelcheck — histórico de correções e testes

## Limitações identificadas e correções (v0.2.0)

| Limitação | Correção | Como funciona |
|---|---|---|
| Correlação linear gerava falso alarme em relações não-monotônicas (forma de U) | Análise por bins | Divide feature em 5 quintis, checa se SHAP médio sobe/desce consistente (≥80% dos passos). Se não-monotônica, severidade vira `check` (revisar) em vez de `high` (alarme) |
| SHAP lento em dados grandes | Fallback automático | Acima de 2.000 linhas, usa permutation importance (mais grosseira, mas rápida). Background do SHAP amostrado em 100 linhas |
| Features categóricas quebravam/eram mal interpretadas | Tratamento explícito | Detector marca como `info` com SHAP médio por categoria; RuleLearner ignora (regra direcional não se aplica) |

## Resultados da validação (datasets reais)

| Dataset | Linhas | Features | Regras aprendidas | Divergências | Contradições | Tempo |
|---|---|---|---|---|---|---|
| breast_cancer (clf) | 569 | 10 | 9 | 5 | 2 (high) | 28s |
| diabetes (reg) | 442 | 10 | 7 | 2 | 2 (1 check, 1 high) | 29s |
| sintético 20k (reg) | 20.000 | 4 | 2 | 0 | 0 | **0,9s** (fallback) |

Destaques:
- **diabetes**: uma contradição veio como `check` (não-monotônica) — a correção do falso alarme funcionando na prática
- **20k linhas em 0,9s** vs ~30s do SHAP — fallback validado (`method = permutation_fallback` confirmado)
- Direção/importância corretas no fallback (teste automatizado confirma feature dominante)

## Suite de testes: 30 testes, 100% passando

Cobertura: explainer (clf/reg/fallback), comparador, divergências, regras YAML,
RuleLearner (direções, ruído, categóricas, roundtrip), contradições (linear,
não-monotônica, categórica, feature constante), fairness, drift, snapshots,
guard end-to-end, relatório, dashboard, MLflow, CI gate, plots.

## Limitações restantes (transparência)

- Permutation fallback é aproximação: menos preciso que SHAP
- Regras aprendidas herdam viés dos dados históricos — revisar antes de usar
- Severidade `check` requer análise humana (a ferramenta não decide sozinha)

## Rodada 2 de validação (v0.2.1)

### Mudança: `max_shap_rows` configurável
Usuário escolhe precisão vs velocidade: `ModelGuard(..., max_shap_rows=50_000)`
força SHAP mesmo em dados grandes (aceita esperar ~30-60s); default 2.000 ativa
o fallback rápido.

### Novos testes com dados reais/realistas

| Cenário | Linhas | Modelos | Resultado | Tempo |
|---|---|---|---|---|
| Wine (real, 13 feats) | 178 | 3 (RF, GB, LogReg) | 4 divergências, 5 contradições | 19s |
| Crédito simulado | 5.000 | 2 | 0 alertas (modelos alinhados às regras — comportamento correto: sem falso positivo) | 49s |
| Crédito, SHAP forçado | 5.000 | 2 | mesmo resultado, `max_shap_rows=50k` | 48s |
| Drift real simulado | 5.000 | retreino com lógica mudada | **2 alertas certeiros**: divida_atual e num_atrasos — exatamente as features cuja importância mudou no cenário | <1s |

Destaques:
- **Zero falso positivo** no crédito limpo: modelos coerentes com os dados não geram alerta
- **Drift detectou exatamente as 2 features certas** quando a lógica do negócio mudou
- 3 modelos simultâneos funcionam (wine)

## Rodada 3 — bateria adversarial (8 cenários hostis) + Telco real

| # | Cenário | Esperado | Observado | Veredito |
|---|---|---|---|---|
| 01 | Ruído puro (target aleatório) | 0 regras, 0 alertas falsos | 0 regras, 0 contradições, 0 fairness | ✅ sem falso positivo |
| 02 | Target leakage (feature = target) | pegar a feature vazada | fairness marcou `feature_leaked` com peso 88–98% | ✅ leak detectado (marcando como sensível) |
| 03 | Relação em U + regra forçada | não dar alarme HIGH falso | correlação ~0 → abaixo do min_strength → sem alarme | ✅ conservador (nota: caminho `check` validado antes no diabetes) |
| 04 | Colinearidade com direções opostas | regras coerentes | 2 regras corretas, 0 falsos alertas | ✅ |
| 05 | Viés de seleção (gênero) | fairness pegar gender | `gender` = 21,6% > limite 20% na LogReg | ✅ viés detectado |
| 06 | Outliers invertem Pearson | learner resistir (Spearman) | learner aprendeu direção certa; **detector pegou a LinearRegression corrompida pelos outliers (HIGH)** | ✅ caso de uso real: modelo quebrado por outlier |
| 07 | Categórica com 20 categorias | ignorar sem quebrar | ignorada, zero crash | ✅ |
| 08 | Overfit (200×50) | limitação esperada | 6 regras (3 reais + 3 espúrias do ruído); 1 contradição espúria | ⚠️ limitação documentada: poucos dados + muitas features geram regras espúrias — revisar regras aprendidas quando n < ~10×features |

### Telco Customer Churn (dataset real, 7.032 clientes)
- Regras aprendidas: `tenure ↓ churn`, `TotalCharges ↓ churn` (corretas: cliente antigo sai menos)
- **Detectou: LogReg usa TotalCharges na direção OPOSTA aos dados (HIGH)** — efeito clássico de multicolinearidade (TotalCharges ≈ tenure × MonthlyCharges). Achado real, ótimo pra demo.
- 2 divergências RF vs LogReg | fairness (SeniorCitizen) ok | 51s

### Conclusões da bateria
1. Zero falso positivo em ruído — crítico pra confiança
2. Pega leakage, viés e modelos corrompidos por outliers
3. Limitação honesta: regras aprendidas em amostras pequenas com muitas features precisam de revisão humana

## Rodada 4 (v0.3.0) — melhorias de robustez e produto

### Implementado
1. **Score de confiança por regra (0-100)**: estabilidade da direção em 4 subamostras (60%) + força da correlação (40%). Regras com confiança baixa não entram no YAML.
2. **Corte adaptativo**: com aviso de "poucos dados" (n < 10×features), exigência sobe de 50 pra 80.
3. **Reprodutibilidade**: seeds fixas; teste automatizado confirma mesmo relatório em execuções repetidas.
4. **Bordas**: coluna 100% NaN (ignora + avisa), 50% NaN (avisa), 30 linhas × 100 features (avisa). Nada quebra.
5. **Sugestões de ação por contradição**: vazamento? constraint monotônica? multicolinearidade (aponta a feature correlacionada >0.7)?
6. **Relatório narrado** (`narrate(report)`): frases pra stakeholder em vez de tabela.
7. **Modelos testados ampliados**: + XGBoost, MLP, Ridge, Lasso (total: 8 tipos).

### Resultado-chave: caso overfit RESOLVIDO
Antes: 3 regras espúrias entravam no YAML.
Depois: confiança separa (reais: 82-100% | espúrias: 75-76%) + corte adaptativo
→ **YAML final contém exatamente as 3 regras reais (feature_0, 1, 2). Zero espúrias.**

### XGBoost no Telco
Contradição em TotalCharges veio como `check` (não-monotônica) — coerente:
árvore captura a relação não-linear que a LogReg linearizava errado.

### Suite: 40 testes, 100% passando

## Rodada 5 — benchmark quantitativo vs baseline (SHAP puro)

Baseline = calcular SHAP e deixar a detecção pro olho humano (prática comum hoje).
ModelGuard = SHAP + regras + detecção automática + sugestões.

| Dataset | Linhas | SHAP puro | ModelGuard | Detecções automáticas (SHAP puro) | Detecções automáticas (ModelGuard) | Falsos positivos |
|---|---|---|---|---|---|---|
| breast_cancer | 569 | 23s | 31s | 0 | 2 (high: mean radius, mean area invertidos na LogReg) | 0 |
| diabetes | 442 | 28s | 39s | 0 | 2 (s1: check no RF, high na LinReg) | 0 |
| wine | 178 | 14s | 13s | 0 | 4 (LogReg com 3 features invertidas) | 0 |
| telco_7k (real) | 7.032 | 12s | 8s | 0 | 1 (TotalCharges invertido na LogReg) | 0 |
| ruído puro | 1.000 | 43s | 70s | 0 | 0 | **0** |

### Leitura
- **Custo**: ModelGuard adiciona 0–60% de tempo sobre o SHAP que você já calcularia — segundos, não minutos.
- **Ganho**: SHAP puro detecta 0 problemas sozinho (exige revisão manual de gráficos, feature a feature, modelo a modelo). ModelGuard detectou 9 contradições reais nos 4 datasets com problemas — automaticamente, com severidade e sugestão de ação.
- **Confiabilidade**: zero falso positivo, incluindo no dataset de ruído puro (o teste mais hostil pra isso).
- Em dados grandes (telco) o ModelGuard foi até mais rápido que SHAP puro (fallback + amostragem de background).

## Rodada 6 (v0.4.0) — leakage, ranking, forma não-linear, Insurance real

### Novos módulos
1. **LeakageDetector** (sem depender do nome da feature): alerta se feature domina >70% da importância OU tem correlação quase perfeita (>0.95) com o target. Testado: pega feature vazada com nome neutro; silencioso em dados limpos.
2. **rank_models(report)**: ranking de coerência — "dos N modelos, X é o mais alinhado com as regras dos dados" (pondera contradições por severidade).
3. **Sugestão de feature engineering**: contradição não-monotônica agora sugere "adicione X² ou use modelo não-linear".
4. **ShapeDivergenceDetector**: compara o FORMATO da relação entre modelos (via bins de SHAP). Detecta quando modelo linear sofre underfitting em relação não-linear que a árvore captura.

### Caso descoberto e resolvido no processo
Simulação tipo bike-sharing (temperatura em U invertido): LinearRegression R²=0,51 vs RF R²=0,99 — e a v0.3 **não avisava nada** (divergência olhava só magnitude, não formato; regra monotônica não se aplica ao U). A v0.4 detecta:
> "'temp': [rf] capturam relação não-monotônica que [lin] tratam como linear. Possível underfitting. → Adicione 'temp²' ou prefira modelo não-linear."

### Insurance (dataset real, 1.338 segurados)
Fumante pesa muito mas abaixo do limiar de dominância; modelos coerentes com regras; zero alerta falso. Comportamento correto.

### Limitação conhecida mantida (honestidade)
RuleLearner continua monotônico: em U perfeitamente simétrico não aprende regra
(Spearman ≈ 0). O ShapeDivergenceDetector cobre esse buraco pelo lado dos modelos.
Regras condicionais ("X sobe→y sobe, mas só se Z>100") ficam no roadmap.

### Suite: 46 testes, 100% passando

## Rodada 7 (v0.5.0) — vazamento temporal + regras condicionais

### TemporalLeakageDetector
Dado uma coluna de data, detecta:
(a) feature que é cópia FUTURA de outra (f[t] ≈ g[t+k], corr>0.95);
(b) feature que prevê o target futuro melhor que o presente.

Teste (cenário bike-sharing com `temp_amanha` plantada):
> "Vazamento temporal: 'temp_amanha' parece ser o valor FUTURO de 'temp' (1 passo à frente)." ✅
Série limpa: silencioso. ✅

### Regras condicionais (learn_conditional)
Detecta quando a direção de uma feature depende de outra
("preço sobe → venda cai, mas só se não for premium").

Teste sintético com interação plantada (y = +3x se premium, −3x se não):
capturou as duas direções e avisou "regra global seria enganosa". ✅
Sem interação real: não inventa regra. ✅
Insurance real: nenhuma inversão de direção por fumante — resultado honesto
(bmi pesa MAIS pra fumantes, mas a direção não inverte; sem alarme). ✅

### Suite final: 50 testes, 100% passando

## Rodada 8 — datasets reais adicionais (mirrors GitHub)

- **Adult Census (48.842 linhas)**: fairness detectou `sex_male` com 16-20% do
  peso das decisões (limite 10%) nos dois modelos — o viés de gênero documentado
  do dataset, pego automaticamente. Zero contradição falsa. Fallback rápido em 48k.
- **Bike Sharing real (728 dias)**: ShapeDivergenceDetector pegou a relação em U
  da TEMPERATURA — a não-linearidade clássica documentada na literatura de ML
  interpretável ("lin trata como linear o que rf captura como curva; possível
  underfitting") — também detectou em humidity. Achado de livro-texto, em dado real.
- Bloqueados pela rede do ambiente: Credit Default e Attrition (Kaggle/UCI);
  demais cobertos por mirrors.

## Rodada 9 — Validação com 6 modelos (estilo HuggingFace, demo local)

Script: `examples/hf_validation.py` — suporta HuggingFace (online) e fallback local.

| Modelo | Dataset | Contradições | Achado |
|---|---|---|---|
| iris_rf (RF) | Iris | 1 (check) | sepal width direção invertida |
| iris_gb (GradientBoosting) | Iris | 1 (check) | mesma feature — consistente entre modelos |
| iris_lr (LogReg) | Iris | 1 (HIGH) | petal width invertida — modelo linear errou |
| breast_rf (RF) | Breast Cancer | 1 (check) | compactness error |
| breast_xgb (XGBoost) | Breast Cancer | 2 (medium) | compactness error + outra |
| diabetes_lr (Linear) | Diabetes | 1 (HIGH) | s1 invertida |

Destaques:
- 3 tipos de modelo (RF, GradientBoosting, LogReg, XGBoost, Linear) — todos funcionam
- iris_lr e diabetes_lr: contradições HIGH em modelo linear (exatamente o silent underfitting do paper)
- zero leakage detectado (correto — datasets sklearn são limpos)
- XGBoost encontrou mais contradições que RF no mesmo dataset — ilustra o multi-model comparator
