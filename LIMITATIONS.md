# ModelGuard — Limitações

> **Atualização v0.12 — leia isto primeiro.** O veredito não usa mais o índice
> de saúde, e a camada de forma passou a detectar por distância de Fréchet
> (a classificação em cinco formas só nomeia a mudança). Os números medidos com
> a versão atual são estes:
>
> | limite | medida atual |
> |---|---|
> | degradação parcial | não bloqueia até 80% das linhas (perda de 4 pontos de R²) |
> | bases de avaliação pequenas (< ~400 linhas) | forma pede revisão em ~1/3 dos retreinos saudáveis |
> | fora de árvores (MLP, ridge) | **experimental**: bloqueia falha de pipeline, mas alarma mais e é ~10x mais lento |
> | boosting (XGBoost, LightGBM) | limiar 5 a 6 vezes o da floresta nos mesmos dados — calibre por família |
> | limiar da distância | 0,12 a 0,47 em seis domínios; calibre com `calibrar_limiar` |
> | prevalência em retreino real | 5% (afluência) a 41% (carga) em 161 retreinos do ONS |
> | sazonalidade | continua indistinguível de deriva |
> | usuário externo | **nenhum ainda** |
>
> As seções abaixo foram escritas para versões anteriores e descrevem o
> comportamento daquela época. Onde houver conflito, vale a tabela acima.


Organizado por **pergunta que você está tentando responder**, não por ordem
histórica. Todo número aqui foi medido; os dados estão em `benchmarks/`.

O histórico cronológico das descobertas está em `LIMITATIONS_historico.md`.

---

## Como escolher o modo — tabela de decisão

| Sua situação | Use | Métrica medida | Não use |
|---|---|---|---|
| Trocou de arquitetura ou hiperparâmetro, **mesmo treino** | `audit(mode="retraining")` | F1=0.96 | — |
| Comparando arquiteturas de propósito | `audit(mode="exploratory")` | diferenças reportadas como esperadas | — |
| **Retreino temporal / janela deslizante** | `check_features(monitored=[...])` | **F1=0.99, especificidade 100%** | ❌ Health Score |
| Suspeita de problema no pipeline de dados | `TrainServingValidator` | 6/6 skews, 0 FP | — |
| Auditoria de conformidade / vazamento | `TemporalLeakageDetector.check_availability()` | conclusivo por metadados | ❌ o modo estatístico (recall 3/5) |
| Quer prever se o modelo vai degradar | **nenhum** | refutado, n=1.128 | ❌ todos |

---

## 1. O que é medido e confiável

| Detector | Recall | Especificidade | Onde foi medido |
|---|---|---|---|
| `check_features` | 97% | **100%** | 108 pares, 4 subsistemas ENA reais |
| SDD (`audit`) | 100% | 88% | 20 cenários, 4 subsistemas |
| `ImportanceCollapse` | 8/8 | 0 FP | 4 datasets públicos + ENA |
| `TrainServingValidator` | 6/6 skews | 0 FP em split aleatório | ENA |
| `PairedDivergence` | 12/12 features | 0 FP em 5 controles | 16 casos reais |
| `PSIShapDetector` | pega o que o ICD não vê | 0 alerta em controle | 4 testes decisivos |

**Propriedades garantidas** (travadas em 166 testes): controle idêntico → score
exatamente 100 · feature de ruído → silêncio · invariância de escala do target
(10⁻³ a 10⁶) · invariância a log/sqrt/×100 nas features · alinhamento por nome
de coluna · determinismo em 3 execuções · até 400 features sem falso positivo ·
dados malformados → erro com instrução.

---

## 2. Onde a ferramenta é cega

### 2.1 Degradação parcial abaixo de ~40%

| fração corrompida | ΔR² | detectado em |
|---|---|---|
| 20% | −0.042 | 0/3 subsistemas |
| 30% | −0.081 | 0/3 |
| 40% | −0.157 | 1/3 |
| 60% | −0.406 | 2/3 |
| 80% | −1.033 | 3/3 |

Dano real de ΔR²=−0.08 passa sem alerta. O ICD procura *colapso* de
importância; em degradação parcial a razão **sobe** (o modelo compensa se
apoiando mais na feature enfraquecida).

**Mitigação:** `TrainServingValidator`, que pega os casos vindos de
inconsistência de pipeline.

### 2.2 Falhas que preservam ordem e importância relativa

Nenhum detector modelo-vs-modelo vê falhas em que as duas versões são
internamente coerentes e a diferença está entre o que cada uma *espera*
receber e o que *recebe*. É a razão de existir a 4ª camada — mas ela precisa
do `X_train`, que não está disponível em todo fluxo.

### 2.3 Distinção entre sazonalidade e deriva — refutado

Retreino em anos hidrológicos consecutivos (legítimo): HS=22.1. Estação úmida
vs seca (deriva sazonal máxima): HS=20.6. **p=0.70, indistinguíveis.** As
features divergentes são as mesmas nos dois casos.

### 2.4 Antecipação de degradação — refutado duas vezes

| teste | n | resultado |
|---|---|---|
| CCEE mensal | 137 | ρ=−0.206, p=0.016 (preliminar) |
| **ENA diário, 4 subsistemas** | **1.128** | **todos \|ρ\|<0.04, p>0.17** |

O resultado preliminar **não se replicou** com 8× mais poder. O único
subsistema significativo teve o **sinal invertido**. É ferramenta de
diagnóstico estrutural, não de previsão.

---

## 3. Health Score — o componente mais fraco

Cinco medições independentes mostram que é o sinal menos confiável:

- **Dominado pela sobreposição das janelas de treino** (ρ=+0.54, p=0.002):
  100% de sobreposição → score 100; 26% → score 14.7.
- **Não discrimina dentro de cada estrato** de sobreposição (p=0.31 a 0.53).
- **Inverte a ordem** em um caso medido: retreino legítimo 43 (BLOQUEADO),
  pipeline quebrado 57 (REVISAR).
- **Satura** perto de zero (6 divergências × 15 pontos zeram a escala).
- **Calibração falhou:** regressão logística contra degradação real observada
  (137 retreinos) deu **AUC 0.533** — indistinguível de aleatório.

**Use `problem_type` e `affected_features` como saída primária.**

**Aplicado na interface (v1.47):** o relatório passou a começar pelo tipo de
anomalia e pelas features afetadas. O índice foi para o rodapé, marcado como
"detalhe técnico", com a ressalva de que não deve ser usado isoladamente. Ele
segue disponível em `result['global_score']` para uso programático, e o
`AuditResult` tem `plain_debug` para quem quiser o número junto.

**Aposentado da interface na v1.47.** O score não aparece mais em `r.plain`,
em `repr()`, em `r.summary` nem no cabeçalho do Slack. Continua acessível em
`r.score` e `r.to_mlflow()` para compatibilidade, e em `r.plain_debug()` —
que o imprime **com a ressalva** de não comparar cenários.

---

## 3-bis. Estabilidade do veredito entre sementes (v1.58)

Testei se os vereditos são estáveis quando só a semente do modelo muda —
nada nos dados, nada na configuração.

| caso | 8–15 sementes | estável? |
|---|---|---|
| pipeline quebrado (Wine) | **BLOQUEADO 8/8** | ✅ |
| causa raiz identificada | `['alcohol']` 7/8, `['pH','alcohol']` 1/8 | ✅ |
| **crise hídrica (CMO)** | **APROVADO 11/15, ATENÇÃO 4/15** | ⚠ |

### Leitura

**O diagnóstico é estável; o veredito nem sempre.**

Nos casos de falha clara (pipeline quebrado) o veredito não oscila: 8 de 8
sementes dão BLOQUEADO, e a causa raiz é `alcohol` em 7 de 8 — na oitava
aparece `pH` junto, o que é redistribuição legítima, não erro.

No caso limítrofe (crise hídrica, onde o retreino é benéfico mas há
divergência real em cenários extremos), o veredito alterna entre APROVADO e
ATENÇÃO conforme a semente. **Ambos são leituras defensáveis do mesmo
diagnóstico** — o `problem_type` é o mesmo, muda só a faixa em que o score
cai.

Variando também `n_estimators` (50, 100, 300) e `max_depth` (4, 6, 10, None),
o padrão se repete: métricas estáveis, veredito oscilando na fronteira.

### Consequência prática

**Reporte o `problem_type` e as features afetadas, não só o veredito.** Em
casos de fronteira, o rótulo APROVADO/ATENÇÃO carrega menos informação que a
descrição do que mudou.

Isto reforça a decisão da v1.47 de tirar o índice de saúde da manchete do
relatório: o mesmo argumento vale para o veredito em casos limítrofes.

---

## 4. Intervalos de confiança — conservadores

Cobertura empírica com nominal 95%: **100%**, largura média 23–44 pontos numa
escala de 0–100. Causa: o score é discreto (múltiplos de ~5 e ~15).

**Conclusivo** para afirmar que dois cenários **não** são distinguíveis.
**Não** serve como evidência de que **são** — prefira `compare_scores_ci`.

Exemplo medido: `RF vs GB` (38.0, IC [19.5–60.1]) e `feature neutralizada`
(11.1, IC [10.0–25.0]) **não são estatisticamente distinguíveis** (p=0.23).

---

## 5. Limitações por método de representação

| Representação | Recall | Precisão | Especificidade | F1 |
|---|---|---|---|---|
| SHAP (só árvores) | 1.00 | 0.86 | 0.67 | 0.92 |
| ALE vetorizado | 1.00 | 0.80 | 0.50 | 0.89 |
| ALE + bootstrap seletivo | 0.75 | 1.00 | 1.00 | 0.86 |

- **A especificidade do ALE depende do n de avaliação** (0.50 com n=250; 0.17
  com n=640). Declare sempre o n.
- **SHAP e ALE não são intercambiáveis em escala:** RF vs GB dá 96.1 (SHAP) e
  55.2 (ALE) no mesmo par. Declare o método.
- **Modelos não-árvore têm recall maior** (1.00 vs 0.75) com o ALE seletivo,
  contra a intuição. Em redes neurais, espere alerta quando só o seed muda.

---

## 6. Detectores complementares — recall baixo por projeto

| detector | recall | precisão | não use como primário porque |
|---|---|---|---|
| `TopologicalDriftGudhi` | 0.33 | 1.00 | cego a colapso de magnitude sem reorganização estrutural |
| `GeometryDivergence` | **0.00** no cenário testado | 1.00 | mede deriva geométrica global, não perda de feature |

Ambos exigem n de avaliação ≥ ~200 e **não têm threshold universal** — o valor
calibrado num dataset sobrepõe controles e positivos em outro.

---

## 7. Limiares exigem calibração por domínio

**PSI-SHAP:** os padrões da indústria (0.10/0.25) são **40% do necessário** —
abaixo da média da variação natural entre modelos idênticos. Calibrado:
0.25/0.64. E **não transferem**: p99 vai de 0.38 (ENA/SE) a 1.35 (ENA/NE).
`compare()` emite `UserWarning` se chamado sem calibrar.

**Colapso de importância:** a variação natural difere por feature. Medido no
ENA em 8 retreinos saudáveis: `ENA_MLT` p5=0.70, mas **`EAR` p5=0.31** —
abaixo do padrão 0.40, gerando falso positivo. Use
`ImportanceCollapseDetector.calibrate()`.

**Análise causal:** requer `min_rel_importance ≥ 0.02`. Sem o filtro, uma
feature de ruído (0.08% da importância) foi apontada como causa com confiança
**0.62–0.92** em controle sem efeito nenhum.

---

## 8. Análise causal — duas limitações estruturais

**(a) Não use a variável de agrupamento como feature.** Se os grupos são
quartis de `x3`, então `x3` aparece como causa com confiança 1.00 — remover um
quartil altera mecanicamente a distribuição. É artefato, não bug.

**(b) Grupos correlacionados com as features invalidam o claim causal.**
Caso real (LiDAR, 1.003 parcelas): agrupando por **espécie**, `hp95` dava
confiança 0.93. A espécie explica 18.3% da variância de `hp95` (eta²,
p<10⁻⁴⁰). Reagrupando por **distrito** (eta²=0.036, 5× menos confundido):
confiança cai para **0.47, 0.40 e 0.20** — nenhum grupo atinge 0.5.

**Os dois claims causais do LiDAR foram retirados.** O detector mede
associação entre remoção de grupo e mudança de forma; chamar isso de causa
exige que o grupo seja independente da feature.

**Sempre reporte o eta² do agrupamento junto com a confiança.**

---

## 9. Detecção de tendência entre versões

Falso positivo medido em 3.000 séries **aleatórias** por tamanho:

| transições | FP (α=0.05) |
|---|---|
| **3** | **34.2%** — bloqueado na API |
| 4 | 8.0% |
| 6 | 5.7% |
| 8+ | ~5.5% |

**Mínimo de 5 versões (4 transições).** Poder: ~19% com 4 transições, ~86%
com 8. Séries curtas não são conclusivas.

---

## 10. Vazamento temporal — só metadados é conclusivo

O modo **estatístico** tem recall 3/5 e zero FP. Pega vazamento explícito
(feature do futuro, target copiado) mas **perde** média móvel centrada e
agregado mensal.

Causa medida: em problemas **autorregressivos**, a feature legítima coincide
com o target em algum lag (ρ(k=1)=1.000), o que confunde qualquer assinatura
de correlação cruzada. Uma regra por assimetria testada deu **especificidade
0%**.

**Use `check_availability()` com os lags declarados.** É o único caminho
conclusivo, e ele lista as features **não declaradas** — onde o vazamento
silencioso mora.

---

## 10-bis. Bugs 15 e 16 — a API pública do SDD (v1.46)

Ao cobrir `shape.py` (34% → 91%), dois bugs na classe
`ShapeDivergenceDetector.check()`:

**Bug 15 — quebrava com a saída padrão do SHAP.** O método usava
`sv.values` direto, assumindo DataFrame. Com `numpy.ndarray` — que é
exatamente o que `shap.TreeExplainer.shap_values()` devolve — levantava
`AttributeError`. Quem chamasse a classe diretamente com a saída crua do
SHAP recebia exceção. Corrigido com normalização de entrada (aceita
DataFrame, array 2D e array 3D de classificação).

**Bug 16 — `KeyError: 'monotonic'` com n pequeno.** Com conjunto de
avaliação pequeno o perfil vem de `_shape_profile_small_n`, que não tem a
chave `monotonic`. O `check()` acessava direto e derrubava a análise
inteira. **Qualquer usuário com n pequeno recebia exceção.** Corrigido
derivando a monotonicidade da forma classificada.

Ambos estavam no caminho da API pública e nenhum dos ~40 experimentos os
expôs, porque sempre chamamos o SDD através de `audit()`, que já
normalizava a entrada.

---

## 10-ter. Bug 17 — resultado plausível para entrada inválida (v1.47)

Sondando a API pública com 24 entradas inválidas, **20 devolviam resultado
normal** em vez de erro:

| entrada inválida | comportamento anterior |
|---|---|
| `feature_names=['nao_existe']` | resultado completo, como se estivesse tudo bem |
| `feature_names=[]` | score 100 |
| **`X` vazio (0 linhas)** | **`frechet_distance: nan`, score 100** |
| `X` com 1 linha | score 100 |
| `X` como numpy | `AttributeError` cru |
| `model_v2=None` | `AttributeError` cru |
| `monitored='a'` (string) | virava `['a', '']` |

O caso de `X` vazio é o extremo: **"está tudo bem" sobre nenhum dado.**

**Corrigido** com `_validate.py`, validação central aplicada em `canary`,
`ale_selective`, `importance`, `psi_shap`, `universal` e `audit`. Resultado:
**22 de 24 sondas agora levantam erro com instrução acionável**. As 2
restantes são comportamento correto — numpy 2D é convertido, e string única
em `monitored` é aceita.

**Como foi encontrado:** os ~40 experimentos do projeto sempre chamavam os
detectores por dentro, via `audit()`, que já normalizava a entrada. Ninguém
nunca instanciou as classes direto, do jeito que a documentação do SHAP
ensina. Os bugs 15, 16 e 17 viviam todos nessa fronteira.

**Lição de checklist:** não basta testar mais cenários — é preciso testar o
mesmo cenário **pelo caminho que o usuário usa**. Ver `tests/test_public_api.py`.

---

## 10-quater. Bug 18 — calibração que falha em silêncio (v1.48)

`ImportanceCollapseDetector.calibrate()` devolvia os thresholds **padrão**
quando nenhuma feature reunia pares suficientes, **sem sinalizar**:

```python
{'thresholds': {'sig': 0.4, 'sig2': 0.4, 'ruido': 0.4},
 'stats': {'sig': {'n': 0}, ...},          # ← só aqui aparecia o problema
 'default': 0.4}
```

O usuário recebia um dict com aparência de calibração válida, seguia com o
limiar fixo de 0.40 e achava que tinha calibrado por domínio. É exatamente o
cenário que a calibração existe para evitar — o `EAR` do ENA tem p5=0.31 e
gera falso positivo com o limiar padrão.

**Corrigido:** o retorno agora tem `ok`, `reason` e `n_calibradas`, e
`per_feature_threshold` fica vazio quando a calibração falha (em vez de
guardar os padrões como se fossem calibrados).

---

## 10-quinquies. Bugs 20, 21 e 22 — a lente "nada para medir" nos módulos LLM (v1.52)

Apliquei a lente que rendeu os bugs 17, 18 e 19 sistematicamente: **18 sondas
de entrada vazia** em todos os componentes de `llm.py` e `llm_audit.py`.

**Bug 20 — `label='BAIXO'` sem nada verificado.** O `HallucinationMeter`
devolvia `score=0.0` e `label='BAIXO'` (risco baixo de alucinação) quando
nenhum claim era extraído. Uma resposta `"Sim."` recebia o **mesmo veredito**
de uma resposta plenamente fundamentada no contexto. Havia dois pontos de
retorno com esse defeito.

Corrigido: `label='NAO_AVALIAVEL'`, `score=None`, e um alerta explicando que
respostas sem entidades, números ou superlativos não são verificáveis por
este método.

**Bug 21 — `mean_inconsistency: nan`.** O `ConsistencyAuditor` com lista de
prompts vazia fazia `np.mean([])`. Um `nan` **atravessa comparações em
silêncio** (`nan > 0.25` é `False`), então o auditor reportava "nenhum risco"
sem ter auditado nada. Corrigido com `None` + alerta CRÍTICO.

**Bug 22 — limiar de similaridade não invariante ao tamanho do texto.** Este
é da família do bug 13. O `_claim_in_context` usa limiar **absoluto** de 0.45
sobre similaridade TF-IDF. Medido: o claim **correto** `'Itaipu tem 14 GW'`
contra o contexto completo dava **0.258** — abaixo do limiar — e era marcado
como não verificado. **Falso positivo de alucinação num fato literal do
contexto.**

Causa: similaridade TF-IDF cai quando o claim é curto e a sentença longa.
Corrigido com fallback de cobertura lexical: um claim cujos termos
significativos estão todos presentes numa sentença do contexto está
verificado, independente do tamanho relativo.

Resultado após as correções (contexto de 3 sentenças):

| caso | score | label |
|---|---|---|
| fato literal | 0.0 | BAIXO |
| fato parafraseado | 0.0 | BAIXO |
| alucinação numérica | 0.5 | MODERADO |
| local errado | 0.5 | MODERADO |
| fato ausente | 1.0 | ALTO |

**28 testes** em `tests/test_llm_experimental.py`.

---

## 10-sexies. Bugs 23 e 24 — `llm.py` validado (v1.53)

Ground truth construído para os quatro componentes restantes, com LLM falso
determinístico (sem API).

| componente | caso saudável | caso problemático | resultado |
|---|---|---|---|
| `LLMDiff` | modelos idênticos | respostas de tema oposto | ✅ 0.0 vs 1.0 |
| `HallucinationDetector` | LLM estável | LLM que se contradiz | ✅ 0 vs risco |
| `LLMFairness` | respostas equivalentes | respostas enviesadas por gênero | ✅ 0.0 vs 1.0 |
| `LLMDrift` | mesmo tema parafraseado | temas opostos | ⚠ **bug 23** |

### Bug 23 — o limiar do MMD não separa em modo TF-IDF

Calibração medida em 3 temas e 30 pares:

```
mesmo tema, parafraseado : média 0.123   max 0.310
temas diferentes         : média 0.296   min 0.000
```

**As distribuições se sobrepõem.** Com o limiar padrão de 0.05, **67% dos
pares saudáveis viram falso positivo**; subindo para 0.31 o recall cai para
58%. Nenhum limiar separa.

**Causa raiz:** sem `sentence-transformers` instalado, o `ResponseEmbedder`
cai em **TF-IDF ajustado no próprio lote** (23 dimensões para 8 frases). O
MMD passa a medir sobreposição **lexical**, não semântica, e o resultado
depende do vocabulário do lote.

Não é corrigível por calibração — é limitação da representação. O retorno
agora declara `embedding_mode` e `threshold_reliable`, com alerta explícito
quando roda em modo lexical.

**Recomendação:** instale `sentence-transformers` para usar o `LLMDrift`. Sem
ele, trate `drift_detected` como indicativo, nunca como veredito.

### Bug 24 — alertas sem chave `type`

Cinco dos seis pontos de alerta em `llm.py` emitiam apenas `severity`,
`metric` e `message`, sem `type`. Quem filtrasse por tipo de alerta
(`[a for a in alerts if a['type'] == ...]`) recebia `KeyError` — ou pior,
com `.get()`, silenciosamente perdia alertas. Contrato de saída agora
uniforme.

**37 testes** em `tests/test_llm_experimental.py`.

---

## 10-septies. Bugs 25, 26 e 27 — `vision.py` validado (v1.54)

Último módulo com zero teste. Validado **sem torch/tensorflow**, com um
`VisionWrapper` falso cuja regra de decisão eu controlo.

**Bug 25 — cosseno entre médias contaminava o veredito.** O `EmbeddingDrift`
decidia por `mmd > t1 OR cosine > t2`. Medido em amostras da **mesma
distribuição**: `mmd=0.004` (correto) mas `cosine=0.824` contra limiar 0.15 —
**falso positivo garantido em qualquer par saudável**.

Causa: em alta dimensão vetores aleatórios são quase ortogonais e a média
tende a zero, tornando o ângulo entre médias instável. O MMD sozinho separava
corretamente (0.004 saudável vs 0.617 com deslocamento).

Corrigido: só o MMD decide; o cosseno vira diagnóstico com severidade INFO.
Resultado: 4/4 casos corretos.

**Bug 26 — números sem significado em entrada degenerada.** Arrays vazios
levantavam `InvalidParameterError` do sklearn (gamma inválido), sem
explicação. Com **1 amostra** devolvia `mmd=0.787` — número sem significado
estatístico apresentado como medida válida. Dimensões incompatíveis davam
`ValueError` de broadcast. Agora os três levantam erro com instrução, e o
mínimo é 10 amostras por lado.

**Bug 27 — contrato de saída sem veredito.** O `VisionFairness` devolvia
`groups`, `accuracy_gap` e `alerts`, **sem booleano de veredito** — enquanto
`EmbeddingDrift` tem `drift_detected` e os detectores de `llm.py` têm
`bias_detected` e `significant_change`. Quem consumisse a saída de forma
uniforme recebia `KeyError`. Adicionado `bias_detected`.

**Não validado:** `SaliencyDiff` e `VisionShapExplainer` exigem torch ou
tensorflow, indisponíveis neste ambiente.

**14 testes** em `tests/test_vision_experimental.py`.

---

## 11. Uma classe de bug que vale conhecer

Cinco dos 14 bugs corrigidos eram da mesma família: **um limiar absoluto
comparado com uma quantidade que depende da escala do problema.**

| bug | módulo | consequência |
|---|---|---|
| 9 | SDD | ruído apontado no controle (score 58 em vez de 100) |
| 11 | ICD | FP em feature íntegra com P=200 |
| 12 | `shape_extensions` | categórica com 0.17% da importância apontada |
| **13** | **slope** | **target em escala 0.001 → ZERO alertas** |
| 14 | slope | sem filtro de importância |

O bug 13 é o mais instrutivo: **qualquer usuário com target em probabilidade,
fração ou percentual decimal recebia silêncio total** — "nenhuma divergência"
num modelo quebrado. Ficou invisível porque nenhum dos ~40 experimentos usou
target sub-unitário.

**Se você estender o ModelGuard:** todo limiar novo deve ser relativo a uma
quantidade da mesma escala, e o teste de invariância (multiplicar o target por
10³ e 10⁻³, exigir o mesmo diagnóstico) está em
`tests/test_properties.py` e deve ser aplicado.

---

## 12. Módulos experimentais — fora do namespace

| módulo | motivo |
|---|---|
| `calibration` | nunca validado; devolvia o threshold padrão |
| `topology` | superado por `universal.TopologicalDriftGudhi` |
| `zest` | validado só contra família de arquitetura |
| `rhale` | um teste sintético |
| `gtsf` | agregador dos acima |
| `llm` | **validado (v1.53).** O `EUAIActChecker` foi **removido do pacote na v1.55** — ver §12-bis |
| `llm_audit` | **validado (v1.51–v1.52):** `HallucinationMeter` recall 100% / especificidade 100% em 8 casos de ground truth; `ConsistencyAuditor` e `PromptAuditor` validados. **Quatro bugs corrigidos: 19, 20, 21 e 22.** Segue em experimental porque o pacote `llm.py` inclui componentes sem validação. Detalhe antigo: `HallucinationMeter` recall 100% e especificidade 100% em 8 casos de ground truth construído; `PromptAuditor` detecta contexto irrelevante e prompt injection. Bug 19 corrigido no caminho. Segue em experimental porque o pacote inclui componentes não validados |
| `vision` | **validado (v1.54):** `EmbeddingDrift` 4/4 com ground truth de separação conhecida; `VisionFairness` distingue modelo justo de enviesado (gap 0.0 vs 0.52). **Três bugs: 25, 26, 27.** `SaliencyDiff` exige torch/tf — não validável neste ambiente |

Há um teste que **falha se voltarem** ao namespace sem validação.

---

## 12-bis. Validação de entrada — bug 17 (v1.47)

Sondei a API pública com 24 entradas inválidas, como um usuário erraria.
**20 devolviam resultado NORMAL em vez de erro:**

| entrada inválida | comportamento antigo |
|---|---|
| `features=['nao_existe']` | resultado completo, como se estivesse tudo bem |
| **`X` vazio (0 linhas)** | `frechet_distance: nan`, score 100 |
| `X` com 1 linha | score 100 |
| `features=[]` | resultado normal |
| `X` como numpy | `AttributeError` cru |
| `model_v2=None` | `AttributeError` cru |

O caso de `X` vazio devolvendo score 100 é o extremo: "está tudo bem" sobre
**nada**. Um resultado plausível para entrada inválida é a pior falha
possível, porque o usuário confia num número que não significa nada.

**Corrigido** com `modelcheck/_validate.py` — validação central aplicada em
`audit`, `canary`, `ale_selective`, `importance`, `psi_shap` e `universal`.
Resultado: **22 de 24 sondas passaram a dar erro com instrução acionável**
(os 2 restantes são comportamento correto: numpy 2D é convertido e
`monitored='a'` é aceito como string única).

**Mínimo de 10 linhas** em `X` passou a ser exigido, com mensagem explícita.

### A lacuna de método que isso expôs

Os bugs 15, 16 e 17 viviam todos na **fronteira entre a biblioteca e o
usuário**. Os ~40 experimentos do projeto sempre chamaram os detectores por
dentro, via `audit()`, que já normalizava tudo.

Testamos exaustivamente comportamento, propriedades e ground truth — e nunca
testamos a porta de entrada. `tests/test_public_api.py` (63 testes) cobre
isso agora: cada classe pública chamada como um usuário externo faria, com
SHAP cru, mais os `report()` e as entradas inválidas.

---

## 12-bis. EUAIActChecker — removido do pacote (v1.55)

O `EUAIActChecker` produzia um *score de conformidade com o EU AI Act* a
partir de um checklist de auto-declaração, com citações de artigos
específicos (Art. 9, 10, 11, 12, 52, 61, Annex III).

**Removido do pacote instalável.** Arquivado em `rascunhos-juridicos/`, fora
do `modelcheck`, com README de aviso.

### Por quê

Todo detector deste projeto foi validado contra ground truth construído: um
problema conhecido é injetado e verifica-se se o detector o encontra. Foram
27 bugs encontrados assim.

**Este componente não é validável dessa forma.** Não existe "corrupção
injetada" que teste se a citação do Art. 52 está correta — isso é revisão
jurídica, não teste estatístico.

As numerações de artigo foram escritas de memória e nunca conferidas contra o
texto oficial, que mudou entre a proposta e a versão adotada. E o resultado é
auto-declaração: quem responde é quem quer o resultado positivo.

Publicar num pacote instalável um número chamado "score de conformidade" sem
essa revisão **transfere risco para quem instala**, sem que a pessoa tenha
como saber que o número não foi verificado.

### Efeito colateral que a remoção expôs

O `LLMGuard` continuava chamando a classe removida, o que daria `NameError`
em runtime para quem passasse `eu_ai_act_responses`. Três referências órfãs
(import, chamada e narração) foram limpas.

### Guardas na suite

Dois testes impedem o retorno: um específico para o `EUAIActChecker` e outro
mais amplo, que falha se **qualquer** componente de conformidade regulatória
(`GDPRChecker`, `ComplianceScorer`, `LegalAuditor`…) entrar no pacote sem
revisão jurídica.

---

## 13. O que nunca foi executado

- **Teste com usuário externo.** Zero pessoas além dos autores usaram a
  ferramenta. É o único erro que 166 testes não pegam: o relatório fazer
  sentido técnico e não fazer sentido para quem decide.
- **Baselines oficiais — parcialmente fechado (v1.50).** A comparação com
  **Evidently 0.7.21 oficial** foi executada: no caso exemplar (RF vs GB,
  ΔR²=0.007) o `DataDriftPreset` reporta **0 colunas com drift** enquanto o
  SDD detecta **3 divergências de forma**. Confirma a prova de novidade que
  havia sido feita com reimplementação.

  **Alibi Detect e NannyML não foram executados** por conflito de
  dependências irreconciliável no ambiente: `alibi-detect 0.13` exige
  `numba`, que exige `numpy<=1.26`, enquanto `shap>=0.52` exige `numpy>=2`.
  Rodar os dois exige ambientes separados. As reimplementações dos mecanismos
  (MMD, performance monitoring) seguem em `baselines.py`.
- **LLMGuard.** Script existe, nunca rodou com API real.
- **Dados não tabulares.** Nada foi testado em texto, imagem ou séries
  multivariadas com dependência longa.

---

## 14. Escopo defensável

> ModelGuard detecta falhas de pipeline de dados e mudanças estruturais entre
> versões de modelo **tabular** quando o conjunto de treino é controlado, e
> inconsistências entre treino e inferência via validação de distribuição.
>
> Não prevê degradação futura. Não distingue sazonalidade de deriva. Não
> detecta degradação parcial abaixo de ~40%. Não foi validado com usuários
> nem fora de dados tabulares.

**Cinco claims foram refutados pelos próprios testes do projeto:** antecipação
de degradação (2×), distinção sazonalidade/deriva, Health Score em retreino
temporal, e os dois claims causais do LiDAR.

---

*18 bugs corrigidos · 270+ testes · métricas em `benchmarks/`*
