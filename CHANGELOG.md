# Changelog

## 1.0.4

Revisão completa da biblioteca por quatro revisores independentes, cada um
com uma parte do código. 53 defeitos confirmados por reprodução e corrigidos,
cada um com teste que falha na 1.0.3 e passa nesta.

### Veredito errado — corrigido

- **Nomes fora de ordem trocavam a culpa** (BUG 69). Com `feature_names` em
  ordem diferente das colunas, o colapso de `a` era atribuído a `d`.
- **numpy virava APROVADO** (BUG 74). Com arrays em vez de DataFrame, SHAP e
  integridade falhavam por dentro e o veredito saía limpo. Sem nomes,
  quebrava com `AttributeError`.
- **Falha da camada de integridade virava APROVADO** (BUG 71). Agora é
  INDETERMINADO, com lacuna declarada. Mesma regra para a camada de forma.
- **Sentinela -999 e join desalinhado viravam "reaprendizado"** (BUG 73), com
  a ação "não é falha de pipeline". Agora uma coluna acusada pela checagem
  treino × uso nunca é tratada como reaprendizado.
  A primeira versão dessa correção foi ampla demais e virou BUG 83: negava
  reaprendizado a qualquer coluna acusada, e em série temporal toda coluna
  deriva um pouco — três retreinos saudáveis do ONS (ENA e carga) viraram
  BLOQUEADO. Ficou assim: só nega quando o alerta é de corrupção (escala,
  nulos, infinitos, constante, níveis novos, taxa); deriva de forma, média
  ou faixa — o que série com tendência produz sozinha — não nega.
- **Modelo que prevê um valor só saía REVISAR** (BUG 72). O ramo de colapso
  do modelo era código morto pelo `Guard`.
- **Multiclasse via só a classe 1** (BUG 68). Variável que separa as classes
  0 e 2 passava invisível. Permutação em multiclasse media distância entre
  rótulos (BUG 70).
- **`combinar` rebaixava o veredito do Guard** (BUG 76); `exit_code`
  devolvia 0 para INDETERMINADO e desligava a falha com `fail_on='HIGH'`
  (BUG 77).
- **O limiar calibrado não chegava à camada de forma** (BUG 80). Novo
  parâmetro `Guard(frechet_threshold=...)`. O padrão (0,33) não mudou.

- **BLOQUEADO falso em alta dimensão** (BUGs 81, 82) — achado depois da
  revisão, na bateria de entradas patológicas, e já presente na 1.0.3. Com
  200 variáveis, retreinos saudáveis saíam BLOQUEADO em 2 a 4 de cada 4
  casos (floresta) e 3 de 3 (boosting estocástico). Agora 0. Floresta:
  teste de permutação entre árvores; boosting: critério de massa de ruído.
  `LIMITATIONS.md` 7-sexies tem os números e o que ficou.

### Sinal que some entre o detector e a saída — corrigido

Classe de bug que apareceu várias vezes (o bug 76 foi o primeiro). Agora
há um teste (`test_caminhos.py`) que exige que API, dicionário, laudo,
texto, CLI, códigos de saída, `combinar` e snapshot digam a mesma coisa em
cada tipo de veredito.

- **CLI saía 0 num BLOQUEADO** (BUG 85): o padrão de `--fail-on` era
  `none`. Agora `high`. INDETERMINADO sai sempre 2 (antes podia sair 0).
- **`compare --monitor` não fazia nada** (BUG 86). Agora roda a checagem
  dirigida e bloqueia se a variável monitorada colapsou.
- **CLI sem dados de treino** (BUG 91): a camada treino × uso nunca rodava
  pela CLI. Novo `--train`; novo `--mode`.
- **Snapshot usava cópia antiga da lógica** (BUG 93): comparava forma pela
  troca de classe (critério que detectava 0% no Benchmark B) e chamava
  mudança de forma de ATENÇÃO. Agora usa a mesma distância, o mesmo limiar
  e o mesmo veredito do Guard. Snapshots antigos continuam lendo, com
  lacuna declarada.
- **`combinar` ignorava o modo exploratório** (BUG 92).

### Documentação que prometia o que o código não fazia — corrigido

- `mode='exploratory'` não mudava nada (BUG 87); erro de digitação no modo
  era aceito.
- Multiclasse fora de árvore dava APROVADO em vez da lacuna documentada
  (BUG 84).
- `check_features` com nome inexistente dizia que a variável "continua
  influenciando o modelo" (BUG 89). Agora é erro.
- `calibrar_limiar` devolvia `None` com poucos pares e o `Guard` usava o
  padrão sem avisar; só media a classe 0 em multiclasse (BUG 90). Agora é
  erro, com o motivo, e mede todas as classes.
- `frechet_threshold` em modelo que não é árvore não tem efeito: agora
  avisa.
- `label_v1`/`label_v2` não apareciam em lugar nenhum (BUG 88).
- `pytest.ini` dizia que os testes lentos só rodam com `-m slow`, mas nada
  os excluía.
- README, COMO_USAR, TUTORIAL: INDETERMINADO nas tabelas, `--fail-on`,
  arquivos que estão no repositório e não no `pip`, nome do produto,
  estudos de caso de versão antiga marcados como tal.

### Correção que ampliou gatilho — corrigido

- **Multiclasse** (BUG 94): a correção do bug 68 media o peso de cada
  variável dentro da própria classe; no vinho, retreinos saudáveis foram de
  1 em 6 para 6 em 6 com REVISAR. Agora o peso é medido contra todas as
  classes: 0 em 6, e a mudança real continua detectada.

### Teste permanente

`tests/test_regressao_saudaveis.py`: retreinos reais do ONS e retreinos
com variáveis binárias, categóricas e multiclasse não podem ser bloqueados.
Amostra por padrão (~4 min); os 157 completos com `pytest -m slow` — rode
antes de toda publicação.

### Checagem treino × uso — corrigido

- Coluna toda `inf` passava como APROVADO (BUG 43).
- Binária rara que zera não era acusada (BUG 44); binárias comuns (5–50%)
  nunca tinham a taxa testada (BUG 45).
- PSI juntava o nível mais alto de variáveis discretas ao vizinho (BUG 46).
- Categóricas só comparavam o conjunto de níveis, não as frequências
  (BUG 47).
- Falso alarme de escala em cauda pesada: 45% → 0% (BUG 48); piso de ruído
  ignorava o tamanho da amostra de uso (BUG 49).

### Laudo — corrigido

- Reaprendizado aparecia como "✓ nenhuma variável deixou de influenciar"
  (BUG 50); "✓" para camada que não rodou (BUG 53); texto afirmava coluna
  constante sem ter verificado (BUG 51); explicação contradizia o veredito
  (BUG 54); "a decisão mudou? Sim" com zero casos trocados (BUG 60).
- CLI montava um laudo incompleto que dizia "✓" onde a API dizia ATENCAO
  (BUG 55). Agora existe `AuditResult.to_dict()` e os dois usam ele.
- XSS no `InteractiveDivergenceExplorer` (BUG 56).
- Segundo laudo num notebook recoloria o primeiro — BLOQUEADO ficava verde
  (BUG 58); o escopo de CSS da 1.0.3 tirava as margens do arquivo solto
  (BUG 59).
- Windows: exemplo do README e saída da CLI quebravam em cp1252 (BUG 57).
- Laudo em inglês com trechos em português; `lang='pt-BR'` saía em inglês;
  NaN aparecia como "+nan" e quebrava o desenho (BUGs 61, 62).

### Outros

- `FutureWarning` do pandas 2.x em `slope.py` e `shape.py` (os avisos
  vermelhos no notebook).
- Sementes fixas onde faltavam (BUGs 78, 79).
- `import modelguard.frechet` funciona; Homepage do PyPI corrigida.

### Documentado

- `LIMITATIONS.md` 7-quater: em domínio com efeito por interação, o limiar
  padrão dá REVISAR em 6 de 6 retreinos saudáveis; calibrado, 2 de 6.
- `LIMITATIONS.md` 7-quinquies: multiclasse fora de árvores e o limite do
  `PSIShapDetector`.

---

## 1.0.3

Quatro correções, todas encontradas usando a ferramenta na própria demo.
Duas delas afetavam o caminho mais percorrido do projeto.

### Corrigido

- **Bug 41 — o laudo afirmava um check que não havia rodado.** O resumo
  mostrava "✓ dados de treino e de uso consistentes" tanto quando a
  comparação treino × uso passava quanto quando ela nem acontecia, porque
  `skew_features` vazio significava as duas coisas. Num instrumento de
  decisão esse é o pior tipo de erro: a pessoa promove o modelo confiando
  num visto que não existe. Agora existe `AuditResult.skew_status`
  (`ok` / `erro` / `ausente`), e o laudo diz **"não verificado"** quando é o
  caso, com o motivo. Dicionários gerados pela 1.0.2 continuam funcionando.

- **Bug 40 — coluna constante derrubava a checagem treino × uso.** O teste de
  proporção da binária rara era aplicado a qualquer coluna de dois níveis,
  mesmo quando os valores não eram uma codificação 0/1. Uma coluna `{0,1}`
  que passava a chegar constante em `3.0` fazia a variância ficar negativa, a
  raiz devolver um número **complexo**, e a checagem morrer com
  `TypeError: '>' not supported between 'complex' and 'float'`. Coluna
  constante é justamente o sintoma que essa checagem existe para pegar. Agora
  o teste de proporção só roda quando os dois lados são mesmo 0/1; valor fora
  da escala vira `NEW_LEVEL`, que é o alerta correto.

- **Bug 42 — o estilo do laudo escapava para a página hospedeira.** As regras
  usavam `:root`, `*` e `body`, então dentro de um notebook iam para o
  documento do editor: o texto do laudo herdava a cor do tema escuro e ficava
  branco no branco. Tudo foi preso a `.folha`.

- **Detector de f-string cego na versão que ele protege.** O teste do bug 38
  usava `tokenize.FSTRING_START`, que só existe no Python 3.12+ — ou seja,
  levantava `AttributeError` em 3.11, exatamente onde o bug acontece. Agora
  tem dois caminhos: em Python < 3.12 compila cada arquivo (ali a sintaxe é
  erro de fato, e isso é a prova); em 3.12+ tokeniza.

### Documentado

- `LIMITATIONS.md` seções **7-bis** e **7-ter**: o veredito depende da janela
  de avaliação quando duas variáveis ficam perto do limiar de clone de 0,95,
  e o protocolo de avaliação correto em série temporal (com avaliação
  estritamente futura a camada treino/serving dispara em 94% dos retreinos
  saudáveis).

### Testes

- Casos novos para os bugs 40 e 41 em `tests/test_classes_de_bug.py`, cada um
  com a regressão do bug anterior incluída.
---


## 1.74.0

Primeira versão publicada. O histórico abaixo resume o que mudou desde que
a biblioteca passou a ser testada sistematicamente.

### Adicionado

- **Snapshots** (`modelcheck snapshot save/list`, `modelcheck diff`) —
  compara o modelo atual contra um perfil gravado, sem precisar do modelo
  anterior carregado. É o que separa comparação pontual de monitoramento.
- **CLI com códigos de saída** — `0` limpo, `1` problema encontrado, `2` erro
  de execução. O `2` é distinto de propósito: um pipeline não pode tratar "a
  auditoria não rodou" como "o modelo passou".
- **Formatos de saída** — `text`, `json` e `html`. O HTML é um laudo
  autocontido, bilíngue (pt/en), que abre de um anexo de e-mail sem rede.
- **`not_evaluated`** — toda auditoria declara o que **não** verificou, com a
  consequência de cada lacuna. "Não avaliamos X" e "X está ok" são afirmações
  diferentes.
- **Guardas de comparabilidade** — recusa comparações inválidas (regressor
  contra classificador, colunas fora de ordem) e sinaliza as frágeis (amostra
  pequena, classe rara, sinal de imputação sem `X_train`).
- **Compatibilidade de esquema** — snapshots de versões anteriores abrem sem
  migração manual; formatos futuros são recusados com instrução.

### Corrigido

Trinta e dois defeitos, organizados em cinco classes. As três primeiras têm
guarda automática que falha na CI sem executar modelo:

| classe | ocorrências |
|---|---|
| resposta plausível onde não havia o que medir | 18 |
| limiar absoluto sobre quantidade de escala livre | 6 |
| chave assumida em retorno de forma variável | 2 |
| implementações paralelas divergindo | 1 |
| exceção capturada sem registro | 1 |

Destaques:

- **#32** — colunas em ordem diferente da do treino produziam veredito
  apontando a **variável errada**. Agora recusa com instrução.
- **#30** — um `NameError` engolido fazia a camada de forma parar de medir
  sem que nada aparecesse. Encontrado pelo próprio `not_evaluated`.
- **#28** — a mesma pergunta, feita por três caminhos da biblioteca, dava três
  respostas.
- Veredito guiado pela camada mais confiável (F1 0,99) em vez do índice de
  saúde (AUC 0,533).
- RHALE desligado por padrão: consumia 93% do tempo da auditoria.
- Warnings do sklearn em toda auditoria, por treinar com array e prever com
  DataFrame.

### Limites conhecidos

`LIMITATIONS.md` é parte do pacote e vale ler antes de confiar em qualquer
resultado. Em resumo: degradação parcial abaixo de ~40% não é detectada, o
método não prevê degradação futura, sazonalidade não se distingue de deriva,
e **nenhum usuário externo validou a ferramenta ainda**.

### Idiomas

O laudo HTML e a CLI existem em português e inglês, seguindo
`MODELCHECK_LANG`. O relatório de texto (`.plain`) existe só em português e
declara isso quando o ambiente pede inglês.
