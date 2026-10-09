import pandas as pd
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from modelcheck.guard import ModelGuard
from modelcheck.explainers.shap_explainer import ShapExplainer
from modelcheck.rules.business_rules import RuleSet
from modelcheck.contradict import ContradictionDetector
from modelcheck.compare.model_comparator import ModelComparator
from modelcheck.report import summary


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "price": rng.uniform(10, 100, 100),
        "age": rng.uniform(20, 60, 100),
    })
    y_clf = (X["price"] < 50).astype(int)
    y_reg = 100 - X["price"] + 0.5 * X["age"]
    return X, y_clf, y_reg


@pytest.fixture
def rules(tmp_path):
    p = tmp_path / "rules.yaml"
    p.write_text("""rules:
  - feature: price
    expected_direction: negative
    description: "Preço alto reduz venda"
  - feature: age
    expected_direction: positive
    description: "Idade aumenta score"
""")
    return str(p)


def test_explainer_classifier(data):
    X, y, _ = data
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    imp = ShapExplainer(m, X).feature_importance(X)
    assert set(imp["feature"]) == {"price", "age"}
    assert (imp["mean_abs_shap"] >= 0).all()


def test_explainer_regressor(data):
    X, _, y = data
    m = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    imp = ShapExplainer(m, X).feature_importance(X)
    assert len(imp) == 2
    assert imp.iloc[0]["feature"] == "price"


def test_comparator_table(data):
    X, y, _ = data
    models = {
        "rf": RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y),
        "lr": LogisticRegression().fit(X, y),
    }
    c = ModelComparator(models, X)
    c.compute_all(X)
    table = c.build_comparison_table()
    assert list(table.columns) == ["rf", "lr"]
    assert table.shape == (2, 2)


def test_divergences_threshold(data):
    X, y, _ = data
    models = {
        "rf": RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y),
        "lr": LogisticRegression().fit(X, y),
    }
    c = ModelComparator(models, X)
    c.compute_all(X)
    assert len(c.find_divergences(0.0)) >= len(c.find_divergences(1.1))
    assert c.find_divergences(1.1) == []


def test_ruleset_yaml(rules):
    rs = RuleSet.from_yaml(rules)
    assert len(rs.rules) == 2
    assert rs.get_rule_for_feature("price").expected_direction == "negative"
    assert rs.get_rule_for_feature("inexistente") is None


def test_contradiction_detected(data, rules):
    X, _, y = data
    m = LinearRegression().fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    det = ContradictionDetector(RuleSet.from_yaml(rules))
    assert det.check(sv, X) == []

    m2 = LinearRegression().fit(X, -y)
    sv2 = ShapExplainer(m2, X).explain(X)
    result = det.check(sv2, X)
    features = [c["feature"] for c in result]
    assert "price" in features


def test_constant_feature_no_crash(rules):
    X = pd.DataFrame({"price": [50.0] * 20, "age": np.arange(20, dtype=float)})
    y = pd.Series(range(20), dtype=float)
    m = LinearRegression().fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    det = ContradictionDetector(RuleSet.from_yaml(rules))
    result = det.check(sv, X)
    assert all(c["feature"] != "price" for c in result)


def test_guard_classification(data, rules):
    X, y, _ = data
    models = {
        "rf": RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y),
        "lr": LogisticRegression().fit(X, y),
    }
    report = ModelGuard(models, X, rules_path=rules).run(X)
    assert "comparison_table" in report
    assert "divergences" in report
    assert "contradictions" in report


def test_guard_regression(data, rules):
    X, _, y = data
    models = {
        "rf": RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y),
        "lin": LinearRegression().fit(X, y),
    }
    report = ModelGuard(models, X, rules_path=rules).run(X)
    assert set(report["contradictions"]) == {"rf", "lin"}


def test_guard_without_rules(data):
    X, y, _ = data
    models = {"lr": LogisticRegression().fit(X, y)}
    report = ModelGuard(models, X).run(X)
    assert "contradictions" not in report


def test_summary_output(data, rules):
    X, y, _ = data
    models = {"lr": LogisticRegression().fit(X, y)}
    report = ModelGuard(models, X, rules_path=rules).run(X)
    assert "MODELCHECK REPORT" in summary(report)


# --- fairness ---
def test_fairness_violation(data):
    X, y, _ = data
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    guard = ModelGuard({"rf": m}, X, sensitive_features=["price"],
                       max_sensitive_weight=0.1)
    report = guard.run(X)
    # price domina o y_clf -> deve estourar limite de 10%
    # v1.66: o FairnessChecker passou a devolver dict com `status`,
    # `weights` e `violations` — antes devolvia só a lista de violações.
    # A mudança veio do bug 29: lista vazia era indistinguível de
    # "variável sensível não existe em X".
    assert report["fairness"]["rf"]["violations"][0]["feature"] == "price"


def test_fairness_ok(data):
    X, y, _ = data
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    guard = ModelGuard({"rf": m}, X, sensitive_features=["age"],
                       max_sensitive_weight=0.9)
    report = guard.run(X)
    assert report["fairness"]["rf"]["status"] == "OK"
    assert report["fairness"]["rf"]["violations"] == []


# --- drift ---
def test_drift_detects_change(data):
    X, y_clf, y_reg = data
    m1 = LinearRegression().fit(X, y_reg)                 # price domina
    m2 = LinearRegression().fit(X, X["age"] * 3.0)        # age domina
    g1 = ModelGuard({"m": m1}, X)
    baseline = g1.run(X)["snapshots"]
    g2 = ModelGuard({"m": m2}, X)
    report = g2.run(X, baseline_snapshots=baseline)
    assert len(report["drift"]["m"]) > 0


def test_drift_stable(data):
    X, _, y = data
    m = LinearRegression().fit(X, y)
    g = ModelGuard({"m": m}, X)
    baseline = g.run(X)["snapshots"]
    report = g.run(X, baseline_snapshots=baseline)
    assert report["drift"]["m"] == []


def test_snapshot_save_load(tmp_path, data):
    from modelcheck.drift import ExplanationSnapshot
    X, _, y = data
    m = LinearRegression().fit(X, y)
    snap = ModelGuard({"m": m}, X).run(X)["snapshots"]["m"]
    path = tmp_path / "snap.json"
    ExplanationSnapshot.save(snap, path)
    loaded = ExplanationSnapshot.load(path)
    assert loaded == snap


def test_summary_with_all_sections(data, rules):
    X, y, _ = data
    m = LogisticRegression().fit(X, y)
    g = ModelGuard({"lr": m}, X, rules_path=rules,
                   sensitive_features=["age"])
    baseline = g.run(X)["snapshots"]
    report = g.run(X, baseline_snapshots=baseline)
    text = summary(report)
    assert "Fairness" in text and "Drift" in text


# --- rule learner ---
def test_rule_learner_learns_directions(data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data  # y = 100 - price + 0.5*age
    rules = RuleLearner().learn(X, y)
    directions = {r["feature"]: r["expected_direction"] for r in rules}
    assert directions["price"] == "negative"
    assert directions["age"] == "positive"


def test_rule_learner_ignores_noise(data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    X = X.copy()
    rng = np.random.RandomState(1)
    X["ruido"] = rng.normal(size=len(X))
    rules = RuleLearner().learn(X, y)
    assert "ruido" not in [r["feature"] for r in rules]


def test_rule_learner_yaml_roundtrip(tmp_path, data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    learner = RuleLearner()
    learned = learner.learn(X, y)
    path = learner.to_yaml(learned, tmp_path / "learned.yaml")
    rs = RuleSet.from_yaml(path)
    assert rs.get_rule_for_feature("price").expected_direction == "negative"


def test_learned_rules_work_with_guard(tmp_path, data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    learner = RuleLearner()
    path = learner.to_yaml(learner.learn(X, y), tmp_path / "learned.yaml")
    # modelo treinado no target invertido -> deve contradizer regras aprendidas
    m = LinearRegression().fit(X, -y)
    report = ModelGuard({"m": m}, X, rules_path=str(path)).run(X)
    assert len(report["contradictions"]["m"]) > 0


# --- plots ---
def test_plot_generates_file(tmp_path, data, rules):
    from modelcheck.plots import plot_comparison
    X, y, _ = data
    models = {"lr": LogisticRegression().fit(X, y)}
    report = ModelGuard(models, X, rules_path=rules).run(X)
    path = plot_comparison(report, str(tmp_path / "p.png"))
    import os
    assert os.path.exists(path) and os.path.getsize(path) > 0


# --- ci gate ---
def test_ci_gate_blocks_contradiction(data, rules):
    from modelcheck.ci import ci_gate
    X, _, y = data
    m = LinearRegression().fit(X, -y)  # contradiz regras
    report = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    passed, reasons = ci_gate(report)
    assert not passed and len(reasons) > 0


def test_ci_gate_passes_clean_model(data, rules):
    from modelcheck.ci import ci_gate
    X, _, y = data
    m = LinearRegression().fit(X, y)  # alinhado com regras
    report = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    passed, reasons = ci_gate(report)
    assert passed and reasons == []


# --- dashboard ---
def test_dashboard_generates_html(tmp_path, data, rules):
    from modelcheck.dashboard import generate_dashboard
    X, y, _ = data
    m = LogisticRegression().fit(X, y)
    g = ModelGuard({"lr": m}, X, rules_path=rules, sensitive_features=["age"])
    baseline = g.run(X)["snapshots"]
    report = g.run(X, baseline_snapshots=baseline)
    path = generate_dashboard(report, str(tmp_path / "d.html"))
    html = open(path).read()
    assert "modelcheck" in html and "Comparação" in html


# --- mlflow ---
def test_mlflow_logging(tmp_path, data, rules):
    pytest.importorskip("mlflow", reason="dependencia opcional")
    import os
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    import mlflow
    from modelcheck.integrations.mlflow_integration import log_to_mlflow
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    X, y, _ = data
    m = LogisticRegression().fit(X, y)
    report = ModelGuard({"lr": m}, X, rules_path=rules).run(X)
    with mlflow.start_run():
        counts = log_to_mlflow(report)
    assert "contradictions" in counts


# --- correções: não-linear, categóricas, dados grandes ---
def test_nonmonotonic_flagged_as_check(tmp_path):
    """Relação em U: não deve dar HIGH, deve marcar 'check' com nota."""
    from modelcheck.rules.business_rules import RuleSet
    from modelcheck.contradict import ContradictionDetector
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, 300)
    X = pd.DataFrame({"feat": x, "other": rng.normal(size=300)})
    y = x ** 2  # forma de U
    m = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    p = tmp_path / "r.yaml"
    p.write_text("""rules:
  - feature: feat
    expected_direction: negative
    description: "regra de teste"
""")
    det = ContradictionDetector(RuleSet.from_yaml(str(p)))
    result = det.check(sv, X)
    for c in result:
        if c["feature"] == "feat":
            assert c["severity"] == "check"
            assert "não-monotônica" in c.get("note", "")


def test_categorical_feature_handled(tmp_path, data):
    from modelcheck.rules.business_rules import RuleSet
    from modelcheck.contradict import ContradictionDetector
    X, _, _ = data
    X = X.copy()
    X["categoria"] = pd.Categorical(["A", "B"] * (len(X) // 2))

    class FakeShap:
        values = np.random.RandomState(0).normal(size=(len(X), 3))

    p = tmp_path / "r.yaml"
    p.write_text("""rules:
  - feature: categoria
    expected_direction: positive
    description: "regra em categórica"
""")
    det = ContradictionDetector(RuleSet.from_yaml(str(p)))
    result = det.check(FakeShap(), X[["price", "age", "categoria"]])
    assert result[0]["severity"] == "info"
    assert "categórica" in result[0]["note"]


def test_rule_learner_skips_categorical(data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    X = X.copy()
    X["cat"] = ["A", "B"] * (len(X) // 2)
    rules = RuleLearner().learn(X, y)
    assert "cat" not in [r["feature"] for r in rules]


def test_large_data_uses_permutation_fallback():
    rng = np.random.RandomState(0)
    n = 3000  # > max_shap_rows (2000)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    y = 2 * X["a"] + 0.1 * rng.normal(size=n)
    m = LinearRegression().fit(X, y)
    exp = ShapExplainer(m, X)
    imp = exp.feature_importance(X)
    assert exp.method == "permutation_fallback"
    assert imp.iloc[0]["feature"] == "a"  # direção/importância corretas


# --- v0.3: confiança, reprodutibilidade, bordas, ações, narrativa ---
def test_confidence_score_present(data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    rules = RuleLearner().learn(X, y)
    assert all(0 <= r["confidence"] <= 100 for r in rules)
    assert all(r["confidence"] >= 70 for r in rules)  # sinal forte no fixture


def test_low_confidence_filtered_in_yaml(tmp_path):
    from modelcheck.rules.rule_learner import RuleLearner
    rng = np.random.RandomState(1)
    n = 60  # pouco dado, sinal fraco -> confiança baixa em espúrias
    X = pd.DataFrame(rng.normal(size=(n, 8)),
                     columns=[f"f{i}" for i in range(8)])
    y = X["f0"] * 2 + rng.normal(size=n)
    learner = RuleLearner(min_correlation=0.15)
    rules = learner.learn(X, y)
    path = learner.to_yaml(rules, tmp_path / "r.yaml")
    kept = RuleSet.from_yaml(str(path)).rules
    # tudo que ficou no yaml tem confiança >= 50
    kept_feats = {r.feature for r in kept}
    for r in rules:
        if r.get("low_confidence"):
            assert r["feature"] not in kept_feats


def test_reproducibility_same_seed_same_report(data, rules):
    X, y, _ = data
    m = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    r1 = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    r2 = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    assert r1["snapshots"] == r2["snapshots"]
    assert r1["contradictions"] == r2["contradictions"]


def test_all_nan_column_warns_not_crashes(capsys):
    from modelcheck.rules.rule_learner import RuleLearner
    rng = np.random.RandomState(0)
    X = pd.DataFrame({"ok": rng.normal(size=100),
                      "vazia": [np.nan] * 100})
    y = pd.Series(X["ok"] * 2 + rng.normal(size=100))
    rules = RuleLearner().learn(X, y)
    out = capsys.readouterr().out
    assert "100% NaN" in out
    assert "vazia" not in [r["feature"] for r in rules]


def test_half_nan_feature_warns(capsys):
    from modelcheck.rules.rule_learner import RuleLearner
    rng = np.random.RandomState(0)
    col = rng.normal(size=100)
    col[:50] = np.nan
    X = pd.DataFrame({"meio_nan": col, "ok": rng.normal(size=100)})
    y = pd.Series(np.nan_to_num(col) + rng.normal(size=100))
    RuleLearner().learn(X, y)
    assert "NaN" in capsys.readouterr().out


def test_tiny_dataset_many_features_warns(capsys):
    from modelcheck.rules.rule_learner import RuleLearner
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=(30, 100)),
                     columns=[f"f{i}" for i in range(100)])
    y = pd.Series(rng.normal(size=30))
    RuleLearner().learn(X, y)
    assert "POUCOS DADOS" in capsys.readouterr().out


def test_contradiction_has_suggestions(data, rules):
    X, _, y = data
    m = LinearRegression().fit(X, -y)
    report = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    for c in report["contradictions"]["m"]:
        assert len(c["suggestions"]) > 0


def test_narrate_generates_text(data, rules):
    from modelcheck.narrate import narrate
    X, _, y = data
    m = LinearRegression().fit(X, -y)
    report = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    text = narrate(report)
    assert "dados históricos" in text and "→" in text


def test_narrate_clean_report(data, rules):
    from modelcheck.narrate import narrate
    X, _, y = data
    m = LinearRegression().fit(X, y)
    report = ModelGuard({"m": m}, X, rules_path=rules).run(X)
    assert "Nenhum problema" in narrate(report)


def test_works_with_mlp_and_ridge(data, rules):
    from sklearn.neural_network import MLPRegressor
    from sklearn.linear_model import Ridge, Lasso
    X, _, y = data
    models = {
        "mlp": MLPRegressor(hidden_layer_sizes=(16,), max_iter=800,
                            random_state=0).fit(X, y),
        "ridge": Ridge().fit(X, y),
        "lasso": Lasso().fit(X, y),
    }
    report = ModelGuard(models, X, rules_path=rules).run(X)
    assert set(report["contradictions"]) == {"mlp", "ridge", "lasso"}


# --- v0.4: leakage, ranking, FE suggestion ---
def test_leakage_detected_without_name_hint():
    from modelcheck.leakage import LeakageDetector
    rng = np.random.RandomState(0)
    n = 500
    t = rng.randint(0, 2, n)
    X = pd.DataFrame({
        "col_a": t + 0.01 * rng.normal(size=n),  # vazada (nome neutro)
        "col_b": rng.normal(size=n),
        "col_c": rng.normal(size=n),
    })
    y = pd.Series(t)
    m = LogisticRegression().fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    alerts = LeakageDetector().check(sv, X, y)
    assert alerts and alerts[0]["feature"] == "col_a"


def test_leakage_silent_on_clean_data():
    from modelcheck.leakage import LeakageDetector
    rng = np.random.RandomState(0)
    n = 400
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.normal(size=n)})
    y = pd.Series(X["a"] + X["b"] + 0.5 * rng.normal(size=n))
    m = LinearRegression().fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    alerts = LeakageDetector().check(sv, X, y)
    assert alerts == []  # duas features dividem o peso: sem dominância nem espelho


def test_ranking_orders_by_coherence(data, rules):
    from modelcheck.ranking import rank_models
    X, _, y = data
    models = {
        "coerente": LinearRegression().fit(X, y),
        "incoerente": LinearRegression().fit(X, -y),
    }
    report = ModelGuard(models, X, rules_path=rules).run(X)
    ranked = rank_models(report)
    assert ranked[0]["model"] == "coerente"
    assert "mais" in ranked[0]["summary"]


def test_fe_suggestion_on_nonlinear(tmp_path):
    from modelcheck.rules.business_rules import RuleSet
    from modelcheck.contradict import ContradictionDetector
    rng = np.random.RandomState(0)
    x = rng.uniform(-3, 3, 300)
    X = pd.DataFrame({"feat": x, "other": rng.normal(size=300)})
    y = x ** 2
    m = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    sv = ShapExplainer(m, X).explain(X)
    p = tmp_path / "r.yaml"
    p.write_text("rules:\n  - feature: feat\n    expected_direction: negative\n"
                 "    description: teste\n")
    result = ContradictionDetector(RuleSet.from_yaml(str(p))).check(sv, X)
    for c in result:
        if c["feature"] == "feat" and c["severity"] == "check":
            assert any("²" in s for s in c["suggestions"])


# --- shape divergence ---
def test_shape_divergence_detects_underfitting():
    from modelcheck.shape import ShapeDivergenceDetector
    rng = np.random.RandomState(7)
    n = 1500
    temp = rng.uniform(0, 40, n)
    X = pd.DataFrame({"temp": temp, "o": rng.normal(size=n)})
    y = pd.Series(-(temp - 25) ** 2 + rng.normal(0, 10, n))
    X_tr = X.iloc[:1000]; y_tr = y.iloc[:1000]; X_te = X.iloc[1000:]
    models = {"lin": LinearRegression().fit(X_tr, y_tr),
              "rf": RandomForestRegressor(n_estimators=30, random_state=0).fit(X_tr, y_tr)}
    shap_all = {n_: ShapExplainer(m, X_tr).explain(X_te)
                for n_, m in models.items()}
    # v1.66: o filtro tau (espalhamento de importância >= 0.30) exige que a
    # IMPORTÂNCIA mude junto com a forma. Aqui linear e RF atribuem peso
    # semelhante a `temp` — é Instabilidade de Forma pura, o caso que o
    # Benchmark B do artigo mede e onde o critério tau detecta 6%.
    # Testamos com tau desativado, que é a Definição 3.
    alerts = ShapeDivergenceDetector(tau=None).check(shap_all, X_te)
    feats = [a["feature"] for a in alerts]
    assert "temp" in feats


def test_shape_divergence_silent_when_both_linear(data):
    from modelcheck.shape import ShapeDivergenceDetector
    X, _, y = data
    models = {"a": LinearRegression().fit(X, y),
              "b": LinearRegression().fit(X, y * 2)}
    shap_all = {n_: ShapExplainer(m, X).explain(X) for n_, m in models.items()}
    alerts = ShapeDivergenceDetector().check(shap_all, X)
    assert alerts == []


# --- v0.5: leakage temporal + regras condicionais ---
def _bike_like(n=400, leak=True):
    rng = np.random.RandomState(3)
    dates = pd.date_range("2024-01-01", periods=n)
    temp = 20 + 10*np.sin(np.arange(n)/30) + rng.normal(0, 2, n)
    y = pd.Series(50 + 3*temp + rng.normal(0, 5, n))
    X = pd.DataFrame({"data": dates, "temp": temp,
                      "humid": rng.uniform(20, 90, n)})
    if leak:
        X["temp_amanha"] = np.roll(temp, -1)  # futuro!
        X = X.iloc[:-1]; y = y.iloc[:-1]
    return X, y


def test_temporal_leakage_detected():
    from modelcheck.temporal import TemporalLeakageDetector
    X, y = _bike_like(leak=True)
    alerts = TemporalLeakageDetector().check(X, y, date_col="data")
    feats = {a["feature"] for a in alerts}
    assert "temp_amanha" in feats


def test_temporal_leakage_silent_clean():
    from modelcheck.temporal import TemporalLeakageDetector
    X, y = _bike_like(leak=False)
    alerts = TemporalLeakageDetector().check(X, y, date_col="data")
    assert [a for a in alerts if a["feature"] == "temp"] == [] or \
           all(a["type"] != "future_mirror" for a in alerts)


def test_conditional_rules_capture_interaction():
    from modelcheck.rules.rule_learner import RuleLearner
    rng = np.random.RandomState(0)
    n = 600
    z = rng.randint(0, 2, n)          # ex: premium sim/não
    x = rng.uniform(0, 10, n)         # ex: preço
    y = pd.Series(np.where(z == 1, 3*x, -3*x) + rng.normal(0, 1, n))
    X = pd.DataFrame({"preco": x, "premium": z})
    cond = RuleLearner().learn_conditional(X, y)
    assert cond, "deveria achar regra condicional"
    rule = [c for c in cond if c["feature"] == "preco"][0]
    dirs = {d for d, _ in rule["by_group"].values()}
    assert dirs == {"positive", "negative"}
    assert "enganosa" in rule["description"]


def test_conditional_none_when_no_interaction(data):
    from modelcheck.rules.rule_learner import RuleLearner
    X, _, y = data
    X = X.copy()
    X["grupo"] = [0, 1] * (len(X) // 2)
    cond = RuleLearner().learn_conditional(X, y,
                                           condition_features=["grupo"])
    assert cond == []  # direção de price/age não muda por grupo


# --- shape v2: classificação e formas opostas ---
def test_shape_classification_correct():
    from modelcheck.shape import _shape_profile
    rng = np.random.RandomState(0)
    x = rng.uniform(0, 10, 1000)
    assert _shape_profile(x, (x - 5) ** 2)["shape"] == "u"
    assert _shape_profile(x, -((x - 5) ** 2))["shape"] == "u_invertido"
    assert _shape_profile(x, 3 * x)["shape"] == "crescente"
    assert _shape_profile(x, -3 * x)["shape"] == "decrescente"


def test_opposite_nonmonotonic_shapes_flagged():
    from modelcheck.shape import ShapeDivergenceDetector
    rng = np.random.RandomState(1)
    n = 800
    x = rng.uniform(0, 10, n)
    X = pd.DataFrame({"x": x, "o": rng.normal(size=n)})
    y_u = pd.Series((x - 5) ** 2 + rng.normal(0, 1, n))
    y_inv = pd.Series(-((x - 5) ** 2) + rng.normal(0, 1, n))
    m1 = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y_u)
    m2 = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y_inv)
    sv = {"a": ShapExplainer(m1, X).explain(X.iloc[:300]),
          "b": ShapExplainer(m2, X).explain(X.iloc[:300])}
    # v1.66: mesma razão do teste acima — U vs U-invertido têm magnitude
    # idêntica de atribuição, só o sinal muda. O filtro tau bloqueia por
    # construção; ver Benchmark B.
    alerts = ShapeDivergenceDetector(tau=None).check(sv, X.iloc[:300])
    hit = [a for a in alerts if a["feature"] == "x"]
    assert hit and "DIFERENTES" in hit[0]["message"]


# --- ALE nativo ---
def test_ale_explain_shape(data):
    from modelcheck.ale import ale_explain
    from modelcheck.shape import ShapeDivergenceDetector
    X, _, y = data
    m = LinearRegression().fit(X, y)
    ale = ale_explain(m, X)
    assert ale.values.shape == (len(X), X.shape[1])
    assert not np.all(ale.values == 0)


def test_ale_vs_shap_bike():
    import os
    if not os.path.exists("data/bike.csv"):
        pytest.skip("CSV externo ausente neste ambiente")
    """Bike temp: ALE e SHAP devem ambos detectar divergência."""
    import sys; sys.path.insert(0,'/home/claude/modelcheck')
    from modelcheck.ale import ale_explain
    from modelcheck.shape import ShapeDivergenceDetector
    from modelcheck.explainers.shap_explainer import ShapExplainer
    bike = pd.read_csv('/home/claude/adv/bike.csv')
    X = bike[["temp","hum","windspeed"]]; y = bike["cnt"]
    Xtr, Xte = X.iloc[:500], X.iloc[500:]
    lin = LinearRegression().fit(Xtr, y.iloc[:500])
    rf = RandomForestRegressor(n_estimators=30, random_state=0).fit(Xtr, y.iloc[:500])
    # SHAP
    sv_shap = {"lin": ShapExplainer(lin,Xtr).explain(Xte),
               "rf":  ShapExplainer(rf, Xtr).explain(Xte)}
    shap_alerts = ShapeDivergenceDetector().check(sv_shap, Xte)
    assert any(a["feature"]=="temp" for a in shap_alerts)
    # ALE
    sv_ale = {"lin": ale_explain(lin, Xte),
              "rf":  ale_explain(rf,  Xte)}
    ale_alerts = ShapeDivergenceDetector().check(sv_ale, Xte)
    assert any(a["feature"]=="temp" for a in ale_alerts)


# --- per-bin significance test ---
def test_per_bin_test_reduces_noise():
    """Com per_bin_test=True, detector deve ser mais conservador em dados ruidosos."""
    from modelcheck.shape import ShapeDivergenceDetector
    rng = np.random.RandomState(42)
    n = 80  # amostra pequena — ruído alto por bin
    x = rng.uniform(0,10,n); z = rng.normal(size=n)
    y = pd.Series(3*x + 0.3*rng.normal(size=n))  # linear
    X = pd.DataFrame({"x":x,"z":z})
    lin = LinearRegression().fit(X, y)
    rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    sv = {"lin": ShapExplainer(lin,X).explain(X),
          "rf":  ShapExplainer(rf, X).explain(X)}
    # sem per-bin test
    a1 = ShapeDivergenceDetector().check(sv, X)
    # com per-bin test
    a2 = ShapeDivergenceDetector(per_bin_test=True).check(sv, X)
    # ambos devem produzir <= alertas com per_bin_test (mais conservador)
    assert len(a2) <= len(a1)


# --- temporal shape monitoring ---
def test_temporal_shape_detects_version_change():
    from modelcheck.temporal_shape import ShapeSnapshot
    rng = np.random.RandomState(0)
    n = 800
    x = rng.uniform(0,10,n); z = rng.normal(size=n)
    X = pd.DataFrame({"x":x,"z":z})
    # v1: RF treinado em U
    y1 = pd.Series((x-5)**2/25 + 0.3*rng.normal(size=n))
    # v2: RF retreinado em linear
    y2 = pd.Series(3*x/10 + 0.3*rng.normal(size=n))
    Xtr, Xte = X.iloc[:560], X.iloc[560:]
    m1 = RandomForestRegressor(n_estimators=30, random_state=0).fit(Xtr, y1.iloc[:560])
    m2 = RandomForestRegressor(n_estimators=30, random_state=0).fit(Xtr, y2.iloc[:560])
    snap1 = ShapeSnapshot.create(m1, Xte)
    snap2 = ShapeSnapshot.create(m2, Xte)
    changes = ShapeSnapshot.diff(snap1, snap2)
    assert any(c["feature"]=="x" for c in changes)
    change_x = [c for c in changes if c["feature"]=="x"][0]
    assert change_x["severity"] == "high"  # monotonia mudou


def test_temporal_shape_stable_retrain():
    """Mesmo modelo, seed diferente: não deve detectar mudança de forma."""
    from modelcheck.temporal_shape import ShapeSnapshot
    rng = np.random.RandomState(1)
    n = 800; x = rng.uniform(0,10,n); z = rng.normal(size=n)
    X = pd.DataFrame({"x":x,"z":z})
    y = pd.Series(3*x/10 + 0.3*rng.normal(size=n))
    Xtr, Xte = X.iloc[:560], X.iloc[560:]
    m1 = RandomForestRegressor(n_estimators=30, random_state=0).fit(Xtr, y.iloc[:560])
    m2 = RandomForestRegressor(n_estimators=30, random_state=1).fit(Xtr, y.iloc[:560])
    snap1 = ShapeSnapshot.create(m1, Xte)
    snap2 = ShapeSnapshot.create(m2, Xte)
    changes = ShapeSnapshot.diff(snap1, snap2)
    # retreino estável não deve mudar forma
    assert all(c["severity"] != "high" for c in changes)
