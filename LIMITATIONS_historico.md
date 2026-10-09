# ModelGuard — Limitações Conhecidas

Documento de fronteiras. Tudo aqui foi **medido**, com os dados anexos ao
repositório. A intenção é que você saiba quando **não** confiar na
ferramenta.

---

## 1. O que a ferramenta detecta bem

| Falha | Recall | Especificidade | Onde foi medido |
|---|---|---|---|
| Feature perde toda a importância (pipeline quebrado) | 97% | 100% | 108 pares, 4 subsistemas ENA reais |
| Mudança de forma da relação feature→predição | 100% | 88% | 20 cenários, 4 subsistemas ENA |
| Envenenamento de labels (15%) | detectado | — | Wine Quality, 4 alertas |
| Colunas reordenadas (schema drift) | detectado | — | ENA/SE |
| Feature congelada (stale) | detectado | — | ENA/SE |
| Vazamento de target | detectado | — | ENA/SE |
| Sinal invertido em feature | detectado | — | ENA/SE |
| Inconsistência treino×inferência (4ª camada) | detectado | 0 FP em split aleatório | ENA/SE, 6 tipos de skew |

---

## 2. O que a ferramenta NÃO detecta

### 2.1 Degradação parcial de feature abaixo de ~40%

Medido em 3 subsistemas, corrompendo a feature mais importante por frações
crescentes:

| fração corrompida | ΔR² médio | detectado em |
|---|---|---|
| 20% | −0.042 | 0/3 |
| 30% | −0.081 | 0/3 |
| 40% | −0.157 | 1/3 |
| 60% | −0.406 | 2/3 |
| 80% | −1.033 | 3/3 |

**Limite de detecção: 40% a 80%, dependendo do dataset.** Dano real de
ΔR²=−0.08 passa sem alerta.

**Por que:** o `ImportanceCollapseDetector` procura *colapso* de importância
(razão cai). Em degradação parcial a razão **sobe** (1.15, 1.20, 1.28) —
o modelo compensa se apoiando mais na feature enfraquecida. O detector é
cego a isso por construção.

**Mitigação:** use a 4ª camada (`TrainServingValidator`), que detecta os
casos em que a degradação vem de inconsistência de pipeline.

### 2.2 Falhas que preservam ordem e importância

Nenhum detector modelo-vs-modelo vê falhas em que as duas versões são
internamente coerentes e a diferença está entre o que cada uma *espera*
receber e o que *recebe*. Foi para isso que a 4ª camada existe — mas ela
precisa do `X_train`, que não está disponível em todos os fluxos.

### 2.3 Antecipação de degradação futura

**Refutado com dados reais.** Em 1.128 retreinos com janela deslizante em 4
subsistemas, todas as correlações entre os índices e a degradação futura
ficaram abaixo de |ρ|=0.04, sem significância (p>0.17). Um resultado
positivo preliminar (ρ=−0.206, p=0.016, n=137) **não se replicou** com 8×
mais poder estatístico.

O ModelGuard é ferramenta de **diagnóstico estrutural**, não de previsão.

### 2.4 Distinção entre sazonalidade e deriva

**Refutado.** Retreino em anos hidrológicos consecutivos (legítimo) dá
HS=22.1; comparação entre estação úmida e seca (deriva sazonal máxima) dá
HS=20.6. **p=0.70** — indistinguíveis. As features divergentes são as mesmas
nos dois casos.

### 2.5 Health Score em retreino temporal

**Não use.** O score correlaciona com a fração de dados de treino
compartilhada entre versões (ρ=+0.54, p=0.002): 100% de sobreposição → score
100; 26% → score 14.7. Dentro de cada estrato de sobreposição ele **não
discrimina** corrupção de retreino saudável (p=0.31 a 0.53).

Em um caso medido ele **inverte** a ordem: retreino legítimo 43
(BLOQUEADO), pipeline quebrado 57 (REVISAR).

**Use `check_features(monitored=[...])`** para retreino temporal:
F1=0.99, especificidade 100% em todos os níveis de sobreposição.

---

## 3. Limitações por método de representação

| Representação | Recall | Precisão | Especificidade | F1 |
|---|---|---|---|---|
| SHAP (só árvores) | 1.00 | 0.86 | 0.67 | 0.92 |
| ALE vetorizado | 1.00 | 0.80 | 0.50 | 0.89 |
| ALE + bootstrap seletivo | 0.75 | 1.00 | 1.00 | 0.86 |

*(18 cenários, 3 datasets)*

- **A especificidade do ALE é sensível ao tamanho da amostra de avaliação.**
  Medida em 0.50 com n=250 e 0.17 com n=640. Qualquer número deve declarar o n.
- **SHAP e ALE não são intercambiáveis em escala.** RF vs GB dá 96.1 (SHAP)
  e 55.2 (ALE) no mesmo par. Declare sempre o método usado.
- **Modelos não-árvore têm recall MAIOR** (1.00 vs 0.75) com o ALE seletivo,
  contra a intuição. Em redes neurais, espere alerta quando apenas o seed
  muda — a rede é genuinamente diferente.

---

## 4. Limitações do detector topológico

Recall **0.33**, precisão 1.00, especificidade 1.00 (20 cenários, 4
subsistemas reais). É camada **complementar**, não primária.

Cego a colapso de magnitude sem reorganização estrutural — exatamente os
casos que o `ImportanceCollapseDetector` acerta. Não existe threshold
universal: o valor calibrado em um dataset sobrepõe controles e positivos
em outro. Exige n de avaliação ≥ ~200 (com 176 registros gerou falso
positivo).

---

## 5. Limitações da detecção de tendência entre versões

Taxa de falso positivo medida em 3.000 séries aleatórias por tamanho:

| nº de transições | FP (α=0.05) |
|---|---|
| **3** | **34.2%** — bloqueado na API |
| 4 | 8.0% |
| 6 | 5.7% |
| 8+ | ~5.5% |

**Mínimo de 5 versões (4 transições).** Poder de detecção: ~19% com 4
transições, ~86% com 8. Séries curtas não são conclusivas.

---

## 6. Limitação com muitas features

Com 200 features e o sinal concentrado em duas, o detector acusa colapso em
feature **íntegra** (2/3 no modo lazy, 3/3 no modo completo). Não é efeito
do modo lazy — é pior sem ele.

**Mitigação:** use `check_features(monitored=[...])` com a lista curta das
features realmente críticas. Monitorar menos é mais seguro.

**Em aberto:** não foi testado se `calibrate()` resolve este caso.

---

## 7. Health Score — o que não fazer com ele

Além da limitação em retreino temporal (§2.5):

- **Não compare scores entre cenários sem IC.** RF vs GB (38.0, IC
  [19.5–60.1]) e feature neutralizada (11.1, IC [10.0–25.0]) **não são
  estatisticamente distinguíveis** (p=0.05).
- **A escala satura perto de zero.** Seis divergências × 15 pontos zeram o
  score, e cenários mais graves não se distinguem.
- **Não é calibrado.** Tentativa de calibração por regressão logística
  contra degradação real observada (137 retreinos CCEE) deu **AUC
  cross-validada 0.533** — indistinguível de aleatório. Os pesos seguem
  heurísticos, ajustados por qualidade de validação de cada detector.

Use `problem_type` e `affected_features` como saída primária.

---

## 7-bis. Intervalos de confiança — calibração medida

Cobertura empírica contra o score da população (5.000 amostras), nominal
95%, 18 repetições, n_eval=150:

| par | cobertura | largura média |
|---|---|---|
| divergente | **100%** | 23.1 pts |
| controle (seed) | **100%** | 44.5 pts |

**O IC é conservador (sobre-cobertura), não anticonservador.** Não há risco
de falsa precisão. O risco é o oposto: intervalos largos demais, com pouco
poder de discriminação. Uma largura de 44.5 pontos numa escala de 0–100 é
quase metade da escala.

**Causa:** o Health Score é discreto (múltiplos de ~5 e ~15 pontos conforme
a contagem de alertas). O bootstrap percentil sobre estatística discreta
produz intervalos que saltam de degrau em degrau.

**Como usar:** o IC é conclusivo para afirmar que dois cenários **não** são
distinguíveis. Não use a não-sobreposição como evidência de que **são**
distinguíveis com 95% de confiança — prefira `compare_scores_ci`, que usa
teste bootstrap da diferença.

**Bug corrigido na v1.35:** o `boot_p` do `compare_scores_ci` devolvia
p=0.0 (máxima significância) quando todas as diferenças eram exatamente
zero — declarando "distintos" no caso de nenhuma diferença. Pego pelo teste
de dois controles idênticos. Corrigido com contagem de sinais opostos e
correção de continuidade.

---

## 7-ter. Análise causal por subgrupo — bug corrigido na v1.36

O `ShapeCausalExplainer` (leave-group-out com IC de Wilson) tinha **dois
defeitos**, encontrados ao validá-lo contra ground truth:

**(a) Causalidade espúria em features irrelevantes.** Sem filtro de
importância, uma feature de **ruído puro** (0.08% da importância total, não
presente no target) foi apontada como causa com confiança **0.62 a 0.92** em
controle com grupo aleatório e nenhum efeito causal. Motivo: o perfil SHAP
de uma feature irrelevante é ruído, então a classificação de forma vira ao
remover qualquer subgrupo.

Corrigido com `min_rel_importance=0.02` (o mesmo mecanismo que o
`ImportanceCollapseDetector` já tinha). Após a correção, o controle
aleatório não aponta nenhuma causa.

**(b) `ZeroDivisionError` no IC de Wilson com n=0.** Agora devolve [0, 1].

**Limitação que permanece:** quando o grupo é derivado de uma feature (ex:
quartis de `x3`), essa feature aparece como causa com confiança alta —
inclusive 1.00. É **artefato estrutural**, não bug: remover um quartil de
`x3` altera mecanicamente a distribuição de `x3`. **Não use a variável de
agrupamento como feature na análise.**

**Impacto nos resultados publicados — RECALCULADO na v1.39:**

Dados: `Forest_attribute_and_LiDAR_metric_at_plot_level_in_Guangxi__China.xlsx`
(1.003 parcelas, 13 métricas, 4 espécies, target VOL, R²=0.872).

Importância relativa medida:

| métrica | importância | passa o filtro (2%)? |
|---|---|---|
| dp50 | 0.3862 | ✅ |
| Hmean | 0.2159 | ✅ |
| **hp95** | **0.0945** | **✅** |
| CC | 0.0903 | ✅ |
| VFPmean | 0.0659 | ✅ |
| VFPcv | 0.0407 | ✅ |
| Hstd | 0.0212 | ✅ |
| Hcv | 0.0202 | ✅ |
| dp75 | 0.0180 | ❌ |
| VFPstdev | 0.0143 | ❌ |
| LADcv | 0.0122 | ❌ |
| LADstd | 0.0120 | ❌ |
| **LADmean** | **0.0086** | **❌** |

Resultado dos dois números publicados:

| claim publicado | sem filtro | com filtro | veredito |
|---|---|---|---|
| Eucalyptus causa hp95: 93%, IC [0.70, 0.99] | 0.87, IC [0.62, 0.96] | **0.93, IC [0.70, 0.99]** | ✅ **CONFIRMADO exatamente** |
| Chinese fir causa LAD: 67%, IC [0.42, 0.85] | 0.73, IC [0.48, 0.89] | **feature excluída** | ❌ **NÃO SOBREVIVE** |

**Volume de achados:** sem filtro, 28 pares feature×espécie tinham confiança
≥0.5 — praticamente todas as combinações, que é a assinatura de causalidade
espúria. Com filtro, 8.

**Conclusão:** o claim de `hp95`/Eucalyptus deve ser mantido; o claim de
LAD/Chinese fir deve ser **retirado**. As três variantes de LAD têm menos de
1.3% da importância — o perfil SHAP delas é ruído, exatamente o mecanismo
que gerou confiança 0.62–0.92 em controles sintéticos sem efeito nenhum.

**Segundo risco (artefato estrutural) — medido e presente:**

A espécie explica parte substancial da variância das métricas
(eta², Kruskal p < 1e-15 em todas):

| métrica | eta² |
|---|---|
| Hmean | 0.246 |
| **hp95** | **0.183** |
| CC | 0.181 |
| Hstd | 0.118 |

Remover uma espécie **desloca** a distribuição da métrica por construção.
Isso significa que mesmo o claim de `hp95` carrega componente de artefato: a
altura de percentil 95 difere entre espécies (Eucalyptus 14.8 m,
Broad-leaved 18.3 m, Masson pine 12.8 m), então o leave-species-out altera
`hp95` mecanicamente.

**TESTE DECISIVO (v1.40) — o efeito NÃO sobrevive a um agrupamento menos
confundido.**

O distrito (Nanning, West, East) é 5× menos correlacionado com as métricas
que a espécie:

| métrica | eta² espécie | eta² distrito |
|---|---|---|
| hp95 | 0.183 | **0.036** |
| Hmean | 0.246 | **0.040** |
| CC | 0.181 | **0.031** |
| Hstd | 0.118 | **0.005** |

Rodando a mesma análise agrupada por **distrito**:

| feature | grupo | confiança | IC 95% |
|---|---|---|---|
| hp95 | Nanning | **0.47** | [0.25, 0.70] |
| hp95 | West | **0.40** | [0.20, 0.64] |
| hp95 | East | **0.20** | [0.07, 0.45] |

**Nenhum distrito atinge 0.5.** Contra 0.93 no agrupamento por espécie.

E o volume total de achados cai de 8 (espécie) para 3 (distrito), nenhum
envolvendo `hp95`.

**Conclusão:** os 93% do agrupamento por espécie são majoritariamente
**artefato de composição amostral**, não relação funcional. Quando o
confundidor é reduzido 5×, o efeito desaparece.

**Recomendação final:** não publicar nenhum dos dois claims causais do
LiDAR como causalidade. O que os dados sustentam é: "a relação entre `hp95`
e volume difere entre espécies", que é uma afirmação descritiva e já
conhecida na literatura de sensoriamento remoto florestal — não um achado
do detector.

---

## 7-quater. Auditoria do padrão "feature irrelevante domina" (v1.37)

Depois de encontrar causalidade espúria no `causal_bootstrap`, auditei os
outros módulos com a mesma lente. Ground truth: uma feature de ruído puro
(0.04% da importância, ausente do target).

| módulo | apontava o ruído? | status |
|---|---|---|
| `importance.py` (ICD) | não | já tinha `min_rel_weight=0.02` |
| `causal_bootstrap.py` | **sim (conf. 0.62–0.92)** | corrigido na v1.36 |
| **`shape.py` / `audit()` (SDD)** | **sim, no controle** | **corrigido: `min_rel_importance=0.01`** |
| `slope.py` | não apontava, mas **quebrava** | **corrigido: `SLOPE_COLLAPSED`** |
| `paired.py` | não (usa importância da árvore, com corte em 0.01) | ok |

### SDD — divergência de forma espúria em ruído

Em controle de variância de seed, o SDD apontava `x3` (ruído) como
divergente, derrubando o score de 100 para **58**. O perfil SHAP de uma
feature irrelevante é ruído, então a classificação de forma vira sozinha.

Corrigido com `min_rel_importance=0.01` (desativável com 0.0). Após a
correção o controle volta a 100 e a causa real em `x0` continua detectada.

**Nota de comportamento correto:** no caso com corrupção, `x3` continua
sendo apontada — e isso é certo. Quando `x0` é neutralizada, o modelo
redistribui peso e `x3` sai de 0.0005 para 0.0431 de importância relativa,
passando o filtro legitimamente. Não é falso positivo; é consequência real
da corrupção.

### slope.py — ZeroDivisionError com feature neutralizada

```
alert = f"v2 é {1/ratio:.1f}x mais suave"   # ratio = 0
→ ZeroDivisionError
```

O `ratio` é zero sempre que a feature é **neutralizada** em v2 — o cenário
mais comum de falha de pipeline. O erro derrubava o `check()` inteiro,
impedindo a análise de todas as outras features. Agora emite
`SLOPE_COLLAPSED`.

---

## 7-quinquies. Família "limiar absoluto" — 4 ocorrências (v1.41–v1.42)

Depois do bug 9, procurei sistematicamente a assinatura: **um limiar
comparado com uma quantidade que depende da escala do problema.** Encontrei
quatro ocorrências.

| bug | módulo | limiar | consequência medida |
|---|---|---|---|
| 9 | `shape.py` (SDD) | nenhum filtro de importância | ruído (0.04%) apontado no controle; score 58 em vez de 100 |
| 11 | `importance.py` (ICD) | razão só relativa | FP em feature íntegra com P=200 (2/3 dos casos) |
| 12 | `shape_extensions.py` | `threshold_magnitude` absoluto | categórica com 0.17% da importância apontada no controle |
| 13 | `slope.py` | `min_effect` absoluto sobre o slope | **target em escala 0.001 → ZERO alertas** |
| 14 | `slope.py` | nenhum filtro de importância | ruído podia ser apontado |

### Bug 13 é o mais grave dos quatro

`min_effect` era um limiar absoluto sobre o slope, que está em unidades do
target. Medido com os mesmos dados e a mesma corrupção:

| escala do target | alertas do slope |
|---|---|
| **0.001** | **0** |
| 1 | 2 |
| 1.000 | 2 |
| 1.000.000 | 2 |

**Um usuário com target em probabilidade, fração ou percentual decimal
receberia silêncio total** — o detector reportaria "nenhuma divergência" num
modelo quebrado. Falha silenciosa, do tipo mais perigoso.

Corrigido: `min_effect` agora é relativo ao desvio-padrão dos próprios SHAP
values. Após a correção, a feature corrompida é detectada em todas as
escalas testadas.

### Nota de comportamento correto (não é bug)

No caso com corrupção, a feature de ruído `b` passa a ser apontada — e está
certo. Ao neutralizar `a`, a importância relativa de `b` vai de **0.2% para
100%**: o modelo passa a usar `b` de fato. Mesmo fenômeno documentado na
seção 7-quater para o SDD.

### Recomendação de projeto

Qualquer limiar novo deve ser **relativo** a uma quantidade da mesma escala,
e o teste de invariância (multiplicar o target por 10³ e 10⁻³, exigir o mesmo
diagnóstico) deveria entrar no checklist de qualquer detector novo.

---

## 7-sexies. Invariância de escala — varredura completa (v1.43)

Depois do bug 13 (slope dava ZERO alertas com target em escala 0.001), varri
os cinco detectores geométricos com o mesmo teste: mesmos dados, target
multiplicado por 10⁻³, 1, 10³ e 10⁶, exigindo o mesmo número de alertas.

### Resultado: todos invariantes

| detector | alertas no controle | alertas no corrompido | invariante |
|---|---|---|---|
| GeometricCanary | 0, 0, 0, 0 | 1, 1, 1, 1 | ✅ |
| GeometryDivergence | 0, 0, 0, 0 | **0, 0, 0, 0** | ✅ |
| TopologicalDriftGudhi | 0, 0, 0, 0 | 1, 1, 1, 1 | ✅ |
| PSIShapDetector | 0, 0, 0, 0 | 3, 3, 3, 3 | ✅ |
| UniversalCanary | 0, 0, 0, 0 | 2, 2, 2, 2 | ✅ |

**Nenhum bug novo da família.** Os detectores geométricos operam sobre
distâncias normalizadas internamente, o que os protege.

### Mas o teste revelou outra limitação

**`GeometryDivergenceDetector` tem recall zero neste cenário.** No caso
corrompido (feature neutralizada), ele mantém score 94–96 e **zero alertas**,
enquanto o Canary cai para 37 e o UniversalCanary para 0.

Perfil: alta especificidade, recall muito baixo — o mesmo do detector
topológico. **Não use `GeometryDivergence` como detector primário.** Ele mede
deriva geométrica global (rotação, compressão do espaço de importâncias), que
não é o que acontece quando uma única feature é neutralizada.

### Recomendação de projeto

O teste de invariância de escala (multiplicar o target por 10³ e 10⁻³,
exigir o mesmo diagnóstico) entrou na suite como
`test_invariancia_de_escala_geometricos` e deve ser aplicado a qualquer
detector novo. Custo baixo, e a família de bug já provou que existe.

---

## 7-septies. Testes de propriedade — 4 categorias novas (v1.44)

Categorias de invariante que não estavam cobertas. Três passaram; uma
revelou diagnóstico enganoso.

| categoria | resultado |
|---|---|
| 1. Invariância a transformação monotônica da feature (log, sqrt, ×100) | ✅ score 100 em todas |
| 2. Colinearidade (ρ=0.998): aponta a feature corrompida? | ✅ aponta `x1`, não a cópia |
| 3. **Modelo degenerado (constante)** | ⚠ **diagnóstico enganoso — corrigido** |
| 4. Alinhamento por nome vs índice de coluna | ✅ alinha por nome |

### Categoria 3 — o achado

Comparando um modelo normal contra um modelo que sempre prevê a média, o
diagnóstico era:

```
score 11 — divergência de forma em ['a', 'b']
```

**Enganoso.** O engenheiro investigaria as features `a` e `b` quando o
problema é que o modelo inteiro colapsou. Não é bug de cálculo — é bug de
diagnóstico, que custa tempo de investigação no lugar errado.

**Corrigido:** `audit()` agora detecta previsão de variância ~zero e emite
`MODEL_COLLAPSED` (ou `MODEL_COLLAPSED_V1` / `BOTH_MODELS_COLLAPSED`), com o
campo `model_collapsed` no retorno e uma linha no topo do relatório dizendo
que as divergências abaixo são consequência, não causa.

### Categoria 2 — vale registrar o resultado positivo

Com duas features a ρ=0.998, corrompendo apenas uma, o `ImportanceCollapse`
aponta a correta e **não** aponta a cópia. Isso era um risco real: a
literatura documenta que o SHAP distribui importância entre features
correlacionadas, e o medo era que o detector culpasse a feature errada,
mandando o engenheiro investigar o lugar errado. Não acontece.

### Categoria 4 — por que importa

Pipelines de feature engineering reordenam colunas com frequência. Se a
comparação fosse por posição, `col1` de v1 seria comparada com `col3` de v2 e
toda auditoria após uma reordenação daria divergência artificial. Verificado
em 3 permutações: score 100 em todas.

---

## 8. Módulos experimentais

Movidos para `modelcheck/experimental/` por validação insuficiente:

| Módulo | Motivo |
|---|---|
| `calibration.py` | nunca validado; devolvia o mesmo threshold padrão |
| `topology.py` | superado por `universal.TopologicalDriftGudhi` |
| `zest.py` | validado só contra família de arquitetura, sem GT de divergência |
| `rhale.py` | um teste sintético controlado |
| `gtsf.py` | agregador; depende dos acima |
| `llm.py` | 429 linhas, zero teste. Inclui `EUAIActChecker` — claim regulatório sem validação. **Movido na v1.43** |
| `llm_audit.py` | 487 linhas, zero teste (`HallucinationMeter`, `ConsistencyAuditor`). **Movido na v1.43** |
| `vision.py` | 330 linhas, zero teste (`VisionFairness`, `SaliencyDiff`). **Movido na v1.43** |

**Nota sobre a remoção dos três últimos:** 1.246 linhas com claims fortes
(detecção de alucinação, fairness, conformidade EU AI Act) e nenhuma
validação de ground truth. Um `EUAIActChecker` não validado num pacote
instalável é passivo, não ativo — se um usuário o empregar numa auditoria
regulatória, o risco recai sobre ele. Voltam ao pacote principal quando
tiverem benchmark medido. Há um teste
(`test_modulos_nao_validados_fora_do_namespace`) que impede que retornem ao
namespace sem isso.

---

## 9. O que nunca foi executado

- **LLMGuard** — o script existe, nunca rodou com API real. Não deve ser
  citado como validado.
- **Baselines oficiais** — a comparação com Evidently, Alibi Detect e
  NannyML foi feita com **reimplementações** dos mecanismos, não com as
  bibliotecas versionadas.
- **Teste com usuário externo** — nenhum. É o único tipo de erro que a
  suite automatizada não pega.

---

## 10. Escopo defensável, em uma frase

> ModelGuard detecta falhas de pipeline de dados e mudanças estruturais
> entre versões de modelo quando o conjunto de treino é controlado, e
> inconsistências entre treino e inferência via validação de distribuição.
> Não prevê degradação futura, não distingue sazonalidade de deriva, e não
> detecta degradação parcial abaixo de ~40%.

---

*Dados de todas as medições em `benchmarks/` e nos CSVs anexos.*
