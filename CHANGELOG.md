# Changelog

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
