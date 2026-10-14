# ShapeAudit

**Seu modelo novo tem a mesma acurácia do antigo. Mas será que ele aprendeu a
mesma coisa?**

Toda vez que um modelo é retreinado, alguém precisa decidir se promove a versão
nova. Na prática, a decisão costuma ser uma só: *a acurácia caiu?* Se não caiu,
sobe.

O problema é que essa pergunta não cobre tudo. Duas versões podem acertar na
mesma proporção e **usar as variáveis de um jeito diferente** — inclusive uma
delas ter perdido uma coluna no caminho, sem que a métrica acuse.

Um exemplo real, com dados públicos de churn de telecom:

```
versão em produção:  0.815
versão candidata:    0.807
diferença:           0.008     ← praticamente nada

ShapeAudit:  NÃO PROMOVA — a variável `Contract` deixou de
             influenciar a previsão e chega constante no treino
```

Uma coluna virou constante depois de uma mudança no pipeline. **Um gate por
métrica aprovaria essa versão.**

## Instalar

```bash
pip install shapeaudit
```

## Usar

```python
from shapeaudit import Guard

r = Guard().audit(modelo_antigo, modelo_novo, X_avaliacao, y_avaliacao,
                  X_train=dados_de_treino_do_modelo_novo)

print(r.verdict)               # APROVADO | REVISAR | ATENCAO | BLOQUEADO | INDETERMINADO
print(r.root_cause_features)   # ['Contract']
open('laudo.html', 'w', encoding='utf-8').write(r.to_html())  # laudo para abrir no navegador
```

`X_train` é o treino do **modelo novo**. É esse argumento que separa "a coluna
sumiu do pipeline" de "o modelo deixou de usar a coluna". Sem ele a auditoria
roda, mas avisa que não verificou isso.

## O que você recebe

| veredito | o que aconteceu | o que fazer |
|---|---|---|
| **BLOQUEADO** | uma variável sumiu do modelo e está constante no treino | não promova: verifique o pipeline |
| **ATENÇÃO** | os dados chegam diferentes do treino, ou uma variável quase idêntica assumiu o lugar de outra | promova com monitoramento, recalibre |
| **REVISAR** | com os mesmos dados, o modelo novo responde de outro jeito | olhe as curvas antes de promover |
| **APROVADO** | nenhuma verificação encontrou mudança | pode promover |
| **INDETERMINADO** | uma verificação principal não rodou (ex.: base de avaliação pequena demais, modelo sem suporte) | não promova com base nesta auditoria: veja a lacuna declarada no laudo |

E um laudo HTML que responde, nesta ordem: o que aconteceu, o que mudou de
peso e de comportamento, onde exatamente mudou, se isso muda decisões, por onde
começar a investigar, e **o que não foi verificado**.

## Experimentar em dois minutos

Os arquivos abaixo estão no [repositório](https://github.com/Rivianee/shapeaudit),
não no pacote do `pip` — baixe o repositório (ou só o arquivo) antes.

- **No Colab**, sem instalar nada: abra `ShapeAudit_Colab.ipynb` e rode tudo.
  Termina com o laudo e com um caso de pipeline quebrado.
- **Com uma planilha sua**: `python demo_excel.py meus_dados.xlsx nome_do_alvo`
  (sem argumentos usa `exemplo_churn.xlsx`). Audita um retreino da sua
  planilha e grava `laudo_shapeaudit.html`.

## Três perguntas, três camadas

1. **Alguma variável sumiu do modelo?** Compara o peso de cada variável nas
   duas versões e distingue "a coluna sumiu" de "o modelo deixou de usar".
2. **Os dados chegam como no treino?** Compara o treino do modelo novo com os
   dados que ele vai pontuar.
3. **O modelo passou a reagir de outro jeito?** Compara a curva de efeito de
   cada variável entre as duas versões. Uma distância detecta a mudança e uma
   classificação em cinco formas a nomeia.

O veredito usa só essas três. Extremos, subgrupos e impacto em grupos de
pessoas aparecem como verificações adicionais.

## Calibrar no seu caso

O limiar padrão veio de uma bancada sintética. Calibre com retreinos que você
**teria promovido** — não precisa de rótulo nem de falha conhecida:

```python
from shapeaudit import Guard
from shapeaudit.frechet import calibrar_limiar

cal = calibrar_limiar([(modelo_jan, modelo_fev), ...], X_avaliacao)
r = Guard(frechet_threshold=cal['limiar']).audit(producao, candidata, X_avaliacao, y)
```

Em domínio com efeito por interação forte, calibrar não é opcional: com o
limiar padrão, retreinos saudáveis pedem revisão quase sempre
(`LIMITATIONS.md`, seção 7-quater).

Medido: de 0,12 a 0,42 em seis conjuntos de dados, e o limiar de modelos de
boosting é 2,5 a 6 vezes o de florestas nos mesmos dados (LightGBM 2,5–3,5×,
XGBoost 5,6–6,3×). Calibre por
domínio **e** por família de modelo.

## O que ele não faz

- **Não bloqueia degradação parcial:** com até 80% das linhas de uma variável
  corrompidas, o modelo perde 4 pontos de R² e nada dispara.
- **Em bases pequenas** (menos de ~400 linhas de avaliação) pede revisão em
  cerca de um terço dos retreinos saudáveis.
- **Fora de árvores** (MLP, ridge) é experimental: bloqueia falha de pipeline,
  mas alarma mais e é cerca de dez vezes mais lento.
- **Não distingue sazonalidade de mudança real.**
- **Não diz qual das duas versões está certa** — só que elas discordam.

`LIMITATIONS.md` traz cada limite com o número medido.

## Confiança no veredito

Em 240 controles saudáveis de benchmark, 157 retreinos reais de séries públicas
do setor elétrico brasileiro e 12 conjuntos públicos, **nenhum retreino saudável
foi bloqueado**. Quando o pipeline quebrou de verdade, bloqueou 16 de 18 casos
nomeando a variável.

A recíproca não vale: não bloquear não garante que está tudo bem.

## Nomes

A distribuição é `shapeaudit`. Três nomes de import funcionam, e apontam para
a mesma implementação:

```python
import shapeaudit      # nome atual
import modelguard      # nome anterior (pacote `modelguard-ml`), mantido
import modelcheck      # nome original, mantido por compatibilidade
```

Quem já usava `modelguard-ml` não precisa mudar código: troque a instalação
para `shapeaudit` e os imports antigos continuam funcionando.

## Licença

MIT. README em português; a documentação de limites também.
