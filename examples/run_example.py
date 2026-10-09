"""Exemplo de uso do ModelGuard."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from modelcheck.guard import ModelGuard


def load_sample_data():
    data = {
        "price": [10, 50, 30, 80, 20, 60, 15, 90],
        "age": [25, 40, 30, 55, 22, 48, 27, 60],
        "target": [1, 0, 1, 0, 1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def train_models(X, y):
    rf_model = RandomForestClassifier(n_estimators=50, random_state=0)
    rf_model.fit(X, y)

    lr_model = LogisticRegression()
    lr_model.fit(X, y)

    return {"random_forest": rf_model, "logistic_regression": lr_model}


def main():
    df = load_sample_data()
    X = df[["price", "age"]]
    y = df["target"]

    models = train_models(X, y)

    guard = ModelGuard(
        models=models,
        X_background=X,
        rules_path="examples/rules.yaml",
    )

    report = guard.run(X)

    from modelcheck.report import summary
    print(summary(report))


if __name__ == "__main__":
    main()

# uso do relatório legível
from modelcheck.report import summary
