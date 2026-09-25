# ShapeAudit

**Duas versões do mesmo modelo. Mesma acurácia. Decisões opostas.**

O ShapeAudit compara duas versões de um modelo de ML e detecta quando a métrica ficou verde mas a lógica aprendida mudou. Em 12 anos de retreinos reais, ficou quieto em 81% deles — e falou exatamente nos dois momentos em que o mundo virou.

[![PyPI](https://badge.fury.io/py/shapeaudit.svg)](https://pypi.org/project/shapeaudit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

---

## O que pode estar acontecendo com o seu modelo agora

Você retreinou. A acurácia continua 0,86. O dashboard está verde. Você fez o deploy.

Semanas depois, alguém descobre que o modelo está negando crédito pelos motivos errados. Ou que parou de usar a variável mais importante. Ou que duas versões do mesmo modelo — uma em produção, outra em staging — discordam em 12% da fila de campanha, com métricas idênticas.

A métrica não mostrou porque a métrica é uma média. E a média esconde.

## O que o ShapeAudit faz

Compara a **forma** das curvas de atribuição (SHAP/ALE) entre duas versões do modelo. Se a variável que antes empurrava a decisão para um lado agora empurra para o outro, ele avisa. Com nome, severidade e ação.

```bash
pip install shapeaudit
