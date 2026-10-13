# ShapeAudit — passo a passo

## 1. Instalar

```bash
pip install shapeaudit
```

Para ver a demo completa, instale também o scikit-learn (se ainda não tiver):

```bash
pip install scikit-learn
```

## 2. O exemplo mínimo (30 segundos)

```python
from shapeaudit import Guard

r = Guard().audit(modelo_antigo, modelo_novo, X_aval, y_aval,
                  X_train=dados_de_treino_do_modelo_novo)

print(r.verdict)               # APROVADO | REVISAR | ATENCAO | BLOQUEADO
print(r.verdict_driver)        # qual camada decidiu
print(r.root_cause_features)   # ['alcohol']
print(r.plain)                 # diagnóstico em português
print(r.not_evaluated)         # o que NÃO foi verificado
```

**`X_train` é o treino do modelo NOVO.** É esse argumento que permite
distinguir "a coluna sumiu do pipeline" de "o modelo deixou de usar a coluna".
Sem ele, a ferramenta funciona, mas declara a lacuna.

## 3. Os cinco vereditos

| veredito | o que aconteceu | o que fazer |
|---|---|---|
| **BLOQUEADO** | uma variável deixou de influenciar o modelo e está constante no treino | não promova: conserte o pipeline |
| **ATENÇÃO** | os dados chegam diferentes do treino, ou um clone assumiu o peso de outra variável | promova com monitoramento, recalibre |
| **REVISAR** | com os mesmos dados, o modelo novo responde diferente | olhe as curvas antes de promover |
| **APROVADO** | nenhuma camada encontrou mudança | pode promover |
| **INDETERMINADO** | uma verificação principal não rodou (ex.: base de avaliação pequena demais, modelo sem suporte) | não promova com base nesta auditoria: veja a lacuna declarada no laudo |

## 4. Começar pela sua planilha

`demo_excel.py`, `demo_visual.py` e `exemplo_churn.xlsx` estão no
[repositório](https://github.com/Rivianee/shapeaudit), não no pacote do `pip`.

```bash
python demo_excel.py                        # usa exemplo_churn.xlsx
python demo_excel.py meus_dados.xlsx alvo   # usa a sua
```

A planilha precisa de: uma linha por caso, uma coluna por variável e **uma
coluna com o que se quer prever**. Texto vira código automaticamente.

O script treina duas versões, audita e escreve `laudo_shapeaudit.html`. Saída:

```
   VEREDITO: REVISAR   (shape)

   o que mudou:
     Contract              24% → 32%    +7.2 pp
     tenure                16% → 13%    -2.8 pp  ← forma mudou
     MonthlyCharges        10% → 15%    +5.8 pp  ← forma mudou
```

Quando for usar de verdade, troque as duas linhas marcadas no script pelos
**seus** dois modelos já treinados.

## 5. O que o laudo responde

Na ordem em que ele abre:

| seção | pergunta |
|---|---|
| veredito | o que aconteceu? |
| resumo | o que mudou e o que ficou estável? |
| por que este veredito | qual camada decidiu, e com que evidência? |
| **o que mudou de peso e de comportamento** | o modelo usa os mesmos sinais? e do mesmo jeito? |
| variáveis que mudaram de padrão | onde exatamente mudou, e quanto? |
| isso muda decisões? | quantos casos trocam na sua regra de decisão |
| próxima ação | o que fazer agora, em ordem |
| cobertura | o que **não** foi verificado |

### A seção que mais importa

```
GANHOU INFLUÊNCIA
  Contract          20% → 30%   +9.9 pp

MESMO PESO, COMPORTAMENTO DIFERENTE
  tenure            13% → 11%   -1.9 pp   mesma forma, curva diferente
```

A segunda categoria é a que só esta ferramenta mostra: **olhando importância
de variáveis, ninguém veria que `tenure` mudou.** O peso é praticamente o
mesmo; o jeito de usar, não.

## 6. A demo pronta (sem planilha)

```bash
python demo_visual.py
```

Ela treina três pares de modelos no Wine Quality (dados embutidos, sem
internet), audita cada um e escreve um laudo HTML por cenário:

```
  ▍ Pipeline quebrado — 'alcohol' chega constante
  ▍ BLOQUEADO   feature_integrity+train_serving
  ▍ comece por: alcohol
  ▍   alcohol               crescente → irregular  d=1.921
  ▍ não verificado: class representation, fairness
```

Abra `laudo_quebrado.html` no navegador. Para cada variável que mudou, o laudo
mostra um cartão com:

- **o gráfico das duas curvas** (cinza = versão antiga, vermelho = nova), com o
  valor da variável no eixo horizontal e o efeito dela na predição no vertical;
- **a faixa sombreada** onde as duas versões mais discordam;
- **uma frase com onde, quanto e em que direção**, por exemplo:

> *"As duas versões continuam decrescentes, mas a curva muda em parte da faixa.
> O maior desacordo está entre 0,305 e 0,36 de `volatile acidity`: nessa faixa
> o efeito da versão nova é mais alto (+0,211 → +0,321). Isso corresponde ao
> que você sabe do domínio? Se sim, a mudança é aceitável; se não, vale
> investigar os dados dessa faixa."*

A ferramenta não decide se a mudança é boa — ela entrega o endereço exato para
o analista investigar.

## 7. Calibrar o limiar no seu domínio

O limiar padrão (0,33) veio de uma bancada sintética. Calibre com retreinos que
você **teria promovido** — não precisa de rótulo nem de falha conhecida:

```python
from shapeaudit.frechet import calibrar_limiar

pares = [(modelo_jan, modelo_fev), (modelo_fev, modelo_mar), ...]
print(calibrar_limiar(pares, X_aval)['limiar'])
```

Medido: de 0,12 a 0,47 em seis domínios, e o limiar do XGBoost é **cinco a seis
vezes** o da floresta nos mesmos dados. Calibre por domínio **e** por família.

## 8. No CI

```bash
shapeaudit --help
```

```bash
shapeaudit compare producao.pkl candidata.pkl avaliacao.csv --target y \
    --train treino_candidata.csv --format html -o laudo.html
```

Códigos de saída:

| código | quando |
|---|---|
| 0 | nada no nível de `--fail-on` |
| 1 | encontrou problema no nível de `--fail-on` |
| 2 | erro de execução, **ou veredito INDETERMINADO** (a auditoria não conseguiu decidir) |

`--fail-on` é `high` por padrão: **BLOQUEADO falha o job**. Use `medium`
para falhar também em REVISAR, `low` para incluir ATENÇÃO, e `none` só se
quiser o laudo sem nunca falhar (modo observação). Até a 1.0.3 o padrão era
`none`, e um modelo bloqueado passava no CI.

`--train` passa os dados de treino da candidata; sem ele a camada treino ×
uso não roda (o laudo diz "não verificado"). `--monitor a,b` roda a
checagem dirigida nessas variáveis e bloqueia se alguma colapsou.

## 9. Antes de confiar

Leia `LIMITATIONS.md`. Em resumo:

- degradação parcial não bloqueia até 80% das linhas de uma variável;
- em bases de avaliação pequenas (menos de ~400 linhas), a camada de forma
  pede revisão em cerca de um terço dos retreinos saudáveis;
- fora de árvores (MLP, ridge) é **experimental**;
- a ferramenta diz que as duas versões discordam, **não** qual delas está
  certa.
