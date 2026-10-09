# modelcheck

Valida modelos de ML contra regras de negócio, detecta drift de explicação e fairness — tudo via SHAP.

## Por quê?
Seu modelo pode ter boa acurácia e ainda assim decidir pelo motivo errado.

## Instalação
```bash
pip install -e .
```

## Uso
```python
from modelcheck.guard import ModelGuard
from modelcheck.report import summary

guard = ModelGuard(models, X, rules_path="rules.yaml",
                   sensitive_features=["idade"])
report = guard.run(X)
print(summary(report))

# monitorar drift depois:
report2 = guard.run(X_novo, baseline_snapshots=report["snapshots"])
```

## Features
- **Contradições**: modelo vs regras de negócio (raro no mercado)
- **Comparação multi-modelo**: onde XGBoost e LogReg discordam
- **Drift de explicação**: alerta se o "porquê" das previsões mudou
- **Fairness**: features sensíveis pesando demais
- **RuleLearner**: aprende regras dos dados automaticamente

```python
from modelcheck.rules.rule_learner import RuleLearner
learner = RuleLearner()
learner.to_yaml(learner.learn(X, y), "rules.yaml")
```

## Regras (rules.yaml)
```yaml
rules:
  - feature: price
    expected_direction: negative
    description: "Preço alto reduz venda"
```

## Testes
```bash
pytest tests/
```

## Licença
MIT
