# Testes — por que cada um existe

Cada arquivo protege contra uma **classe** de defeito que já aconteceu. Antes
de apagar ou afrouxar um teste, leia a classe que ele guarda: quase todos
nasceram de um bug real, e vários desses bugs passaram despercebidos por
muito tempo.

## Como rodar

```bash
pytest tests -q                                   # suíte normal (~15 min)
pytest tests/test_regressao_saudaveis.py -m slow  # 157 retreinos reais (~40 min)
```

Rode a segunda linha **antes de toda publicação**. Os testes marcados como
`slow` só rodam quando pedidos com `-m slow`.

## As classes de defeito que mais se repetiram

### 1. O sinal existe e não chega à saída

O detector mediu, mas o veredito, o laudo ou o código de saída não
refletiram. É o pior modo de falha: produz um **APROVADO falso** sem nenhum
erro aparente.

Casos reais:
- `combinar` jogava fora o ATENÇÃO do Guard (bug 76);
- a CLI saía com código 0 num BLOQUEADO (85) e tratava INDETERMINADO como
  sucesso;
- `compare --monitor` não tinha efeito (86);
- o snapshot usava uma cópia antiga da lógica (93);
- o laudo afirmava "✓" para uma checagem que não rodou (41, 50, 53).

| arquivo | o que garante |
|---|---|
| `test_caminhos.py` | para cada tipo de veredito, API, dicionário, laudo, texto, CLI, códigos de saída, `combinar` e snapshot dizem a mesma coisa |
| `test_consistency.py` | módulos que reimplementam o mesmo cálculo dão o mesmo número (bug 28) |
| `test_revisao_laudo.py` | o laudo não afirma o que não verificou |

### 2. A correção amplia o gatilho

Consertar um falso negativo cria falsos positivos em outro lugar. Aconteceu
quatro vezes:
- **30:** a camada de forma falhava em silêncio;
- **59:** o escopo do CSS tirou as margens do laudo;
- **83:** a correção da sentinela bloqueou 3 retreinos saudáveis do ONS;
- **94:** a correção de multiclasse pôs o vinho em REVISAR em 6 de 6
  retreinos saudáveis.

| arquivo | o que garante |
|---|---|
| `test_regressao_saudaveis.py` | retreinos reais e saudáveis (ONS, churn, vinho) não são bloqueados nem disparam alarme sem motivo |
| `test_golden.py` | o veredito de casos fixos não muda sem alguém mudar o teste |
| `test_regression.py` | cenários das medições documentadas continuam dando o mesmo resultado |

### 3. A documentação promete, o código ignora

O caso que abriu esta classe: o README mandava calibrar o limiar e
passá-lo ao Guard, mas o Guard descartava o valor (bug 80). Outros casos:
- `mode='exploratory'` não fazia nada (87);
- a calibração devolvia `None` em silêncio (90);
- um nome monitorado inexistente recebia visto de "continua
  influenciando" (89).

Regra: teste o **efeito**, não a atribuição. "Mudar o limiar muda o
veredito", não "o atributo guarda o valor".

| arquivo | o que garante |
|---|---|
| `test_revisao_docs.py` | cada parâmetro documentado tem efeito observável |
| `test_tutorial.py` | o TUTORIAL roda como está escrito |
| `test_public_api.py` | a API se comporta como um usuário de fora a chamaria |

### 4. Entrada que o experimento nunca usou

Os experimentos sempre passavam DataFrame, nomes na ordem e floresta de
árvores. Usuários reais passam outras coisas:
- numpy (74);
- nomes fora de ordem, que trocavam a culpa de variável (69);
- multiclasse (68);
- 200 variáveis, com BLOQUEADO falso (81, 82);
- coluna toda infinita (43).

| arquivo | o que garante |
|---|---|
| `test_revisao_nucleo.py` | cada defeito do núcleo do veredito achado na revisão de 2026-10 |
| `test_revisao_validacao.py` | a checagem treino × uso: infinito, binárias, categóricas, cauda pesada |
| `test_chaos.py` | dado malformado gera erro claro, nunca exceção crua |
| `test_edge_cases.py` | n mínimo, variável constante, tudo irrelevante |
| `test_comparabilidade.py` | condições que invalidam a comparação são declaradas |
| `test_classes_de_bug.py` | guardas por classe (bugs 31–42), incluindo sintaxe que só existe no Python 3.12 |

### 5. Estado escondido

| arquivo | o que garante |
|---|---|
| `test_idempotencia.py` | a mesma auditoria dá o mesmo resultado byte a byte, no mesmo processo e entre processos |
| `test_properties.py` | invariantes que todo detector respeita |

## Demais arquivos

| arquivo | assunto |
|---|---|
| `test_frechet.py`, `test_shape_core.py`, `test_importance_core.py` | cobertura dos três componentes centrais |
| `test_laudo.py` | renderização do laudo |
| `test_snapshot_cli.py`, `test_robustez_snapshot.py`, `test_schema_compat.py` | snapshot, CLI e compatibilidade de esquema entre versões |
| `test_pipeline.py` | composição com ferramentas de deriva (Evidently etc.) |
| `test_fairness.py` | módulo de fairness |
| `test_performance.py` | limites de tempo (`slow`) |
| `test_llm_experimental.py`, `test_vision_experimental.py` | módulos experimentais, fora do veredito |
| `checa_fstring.py` | detector usado por `test_classes_de_bug.py` (bug 38) |

## Dados

`dados/` tem a ENA e a carga diárias do subsistema SE (ONS, públicos) e o
`winequality-red.csv` (UCI). São usados pela regressão permanente; os
modelos são retreinados a cada execução, então o teste pega mudança tanto no
código quanto no comportamento das dependências.
