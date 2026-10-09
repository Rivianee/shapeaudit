# Demonstração do ShapeAudit

Uma página só, estática, que conta uma semana na vida de um modelo de
cancelamento de clientes e mostra o laudo que a biblioteca gera em cada caso.

## Abrir

Clique duas vezes em `index.html`. Não precisa de servidor, internet, nem
instalar nada — os três laudos foram gerados pela biblioteca e estão aqui como
arquivos.

## Na apresentação

A página tem um momento planejado. Ela abre mostrando **só a AUC** dos três
casos: 0,862, 0,859 e 0,859. Você diz *"um portão por métrica aprovaria os
três"* e então clica em **Revelar o que a auditoria viu** — aí aparecem os
vereditos e quantos clientes trocam na fila de retenção.

Atalhos de teclado, para não precisar do mouse:

| tecla | o que faz |
|---|---|
| `espaço` | revela os vereditos |
| `→` `←` | passa entre os três casos |

Na impressão (Ctrl+P) os três casos saem abertos, sem os botões.

## O que tem aqui

| arquivo | o que é |
|---|---|
| `index.html` | a página da demonstração |
| `laudo1.html` | retreino de rotina → **Pode promover** |
| `laudo2.html` | outro algoritmo → **Revise antes de promover** |
| `laudo3.html` | pipeline quebrado → **Não promova**, causa `Contract` |

Os laudos saíram de `estudo_de_caso.py`, com uma diferença: a distância
numérica foi removida dos cartões, porque a plateia é mista. O laudo que a
biblioteca gera por padrão mostra o número.

## Reproduzir

```bash
pip install shapeaudit openpyxl
python estudo_de_caso.py
```

Dados: 7.043 clientes do conjunto público de churn de telecom.
