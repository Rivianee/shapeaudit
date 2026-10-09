"""
Validação do modelcheck com modelos do HuggingFace.

Uso:
    pip install huggingface_hub joblib xgboost lightgbm
    python hf_validation.py

Gera relatório de validação para N modelos públicos.
"""
import sys, warnings, json, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_iris, load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from modelcheck.guard import ModelGuard
from modelcheck.rules.rule_learner import RuleLearner
from modelcheck.report import summary
from modelcheck.narrate import narrate

# ── Modelos curados do HuggingFace ──────────────────────────────────────────
# Cada entrada: (repo_id, filename, task, dataset_loader, target_col ou None)
HF_MODELS = [
    # sklearn
    ("vblagoje/sklearn-iris",               "sklearn_model.joblib", "clf", "iris",     None),
    ("julien-c/wine-quality",               "sklearn_model.joblib", "reg", "iris",     None),
    ("skops-community/breast-cancer-clsf",  "sklearn-model.joblib", "clf", "breast",   None),
    # xgboost
    ("xgboost-community/xgboost-titanic",   "model.json",           "clf", "breast",   None),
]

FALLBACK_MODELS = []  # será preenchido com modelos locais se HF falhar


def load_dataset(name):
    if name == "iris":
        d = load_iris()
        X = pd.DataFrame(d.data, columns=d.feature_names)
        y = pd.Series(d.target)
    elif name == "breast":
        d = load_breast_cancer()
        X = pd.DataFrame(d.data, columns=d.feature_names)
        y = pd.Series(d.target)
    elif name == "diabetes":
        d = load_diabetes()
        X = pd.DataFrame(d.data, columns=d.feature_names)
        y = pd.Series(d.target)
    else:
        raise ValueError(f"Dataset desconhecido: {name}")
    return X, y


def load_hf_model(repo_id, filename):
    """Baixa e carrega modelo do HuggingFace."""
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    if filename.endswith(".joblib") or filename.endswith(".pkl"):
        return joblib.load(path)
    elif filename.endswith(".json"):
        import xgboost as xgb
        m = xgb.XGBClassifier()
        m.load_model(path)
        return m
    else:
        raise ValueError(f"Formato não suportado: {filename}")


def run_one(model, model_name, X, y, task):
    """Roda modelcheck em um modelo e retorna resultado."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

    learner = RuleLearner()
    learned = learner.learn(X_tr, y_tr)
    rules_path = f"/tmp/hf_{model_name}.yaml"
    if learned:
        learner.to_yaml(learned, rules_path)
    else:
        rules_path = None

    t0 = time.time()
    guard = ModelGuard(
        {model_name: model},
        X_tr,
        rules_path=rules_path,
    )
    report = guard.run(X_te)
    elapsed = time.time() - t0

    n_contra = sum(len(v) for v in report.get("contradictions", {}).values())
    n_div    = len(report.get("divergences", []))
    leakage  = report.get("leakage", {})
    n_leak   = sum(len(v) for v in leakage.values())

    return {
        "model":         model_name,
        "n_rows":        len(X),
        "n_features":    X.shape[1],
        "rules_learned": len(learned),
        "contradictions":n_contra,
        "divergences":   n_div,
        "leakage_alerts":n_leak,
        "elapsed_s":     round(elapsed, 1),
        "narration":     narrate(report),
    }


def main():
    print("=" * 60)
    print("modelcheck — Validação com modelos do HuggingFace")
    print("=" * 60)

    results = []

    for repo_id, filename, task, dataset, _ in HF_MODELS:
        model_name = repo_id.split("/")[-1]
        print(f"\n[{model_name}] Baixando do HuggingFace...")
        try:
            model = load_hf_model(repo_id, filename)
            X, y  = load_dataset(dataset)
            print(f"  Modelo carregado. Rodando modelcheck...")
            res = run_one(model, model_name, X, y, task)
            results.append({**res, "source": "huggingface", "status": "ok"})
            print(f"  OK | contradições={res['contradictions']} "
                  f"divergências={res['divergences']} "
                  f"tempo={res['elapsed_s']}s")
            print(f"  {res['narration'].split(chr(10))[0]}")
        except Exception as e:
            print(f"  FALHOU: {e}")
            results.append({"model": model_name, "status": "failed", "error": str(e)})

    print("\n" + "=" * 60)
    print("RESUMO")
    print("=" * 60)
    ok = [r for r in results if r.get("status") == "ok"]
    fail = [r for r in results if r.get("status") == "failed"]
    print(f"Modelos testados: {len(results)} | OK: {len(ok)} | Falhou: {len(fail)}")

    if ok:
        print(f"\nResultados:")
        for r in ok:
            print(f"  {r['model']:30s} contra={r['contradictions']} "
                  f"div={r['divergences']} {r['elapsed_s']}s")

    if fail:
        print(f"\nFalhas (verificar repo_id/filename):")
        for r in fail:
            print(f"  {r['model']}: {r.get('error','')[:60]}")

    out = "/tmp/hf_validation_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nResultados completos: {out}")


# ── Uso alternativo: testar com modelos locais sklearn ──────────────────────
def run_local_demo():
    """Demo completo sem precisar do HuggingFace."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, LinearRegression
    import xgboost as xgb

    MODELS_LOCAL = {
        "iris_rf":  (RandomForestClassifier(n_estimators=50, random_state=0), "iris",   "clf"),
        "iris_gb":  (GradientBoostingClassifier(random_state=0),              "iris",   "clf"),
        "iris_lr":  (LogisticRegression(max_iter=1000),                        "iris",   "clf"),
        "breast_rf":(RandomForestClassifier(n_estimators=50, random_state=0), "breast", "clf"),
        "breast_xgb":(xgb.XGBClassifier(n_estimators=50, random_state=0,
                       eval_metric="logloss", verbosity=0),                   "breast", "clf"),
        "diabetes_lr":(LinearRegression(),                                     "diabetes","reg"),
    }

    print("\n" + "=" * 60)
    print("Demo local (substituto para HuggingFace quando offline)")
    print("=" * 60)

    results = []
    for name, (model_cls, dataset, task) in MODELS_LOCAL.items():
        X, y = load_dataset(dataset)
        X_tr, X_te, y_tr, _ = train_test_split(X, y, random_state=0)
        model_cls.fit(X_tr, y_tr)
        res = run_one(model_cls, name, X, y, task)
        results.append(res)
        print(f"  {name:20s} contra={res['contradictions']} "
              f"leak={res['leakage_alerts']} {res['elapsed_s']}s")
        print(f"    {res['narration'].split(chr(10))[0]}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true",
                        help="Usar modelos locais (não precisa do HuggingFace)")
    args = parser.parse_args()

    if args.local:
        run_local_demo()
    else:
        main()
