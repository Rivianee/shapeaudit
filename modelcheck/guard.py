"""ModelGuard: ponto de entrada único para checagem de modelos."""
from modelcheck.compare.model_comparator import ModelComparator
from modelcheck.contradict import ContradictionDetector
from modelcheck.drift import ExplanationDriftDetector, ExplanationSnapshot
from modelcheck.explainers.shap_explainer import ShapExplainer
from modelcheck.fairness import FairnessChecker
from modelcheck.leakage import LeakageDetector
from modelcheck.shape import ShapeDivergenceDetector
from modelcheck.rules.business_rules import RuleSet


class _ModelGuardLegacy:
    def __init__(self, models: dict, X_background, rules_path=None,
                 sensitive_features=None, max_sensitive_weight=0.2,
                 max_shap_rows=2000):
        # max_shap_rows: acima disso usa fallback rápido.
        # Prefere SHAP mesmo se demorar? Aumente (ex: 50_000).
        self.models = models
        self.X_background = X_background
        self.rule_set = RuleSet.from_yaml(rules_path) if rules_path else None
        self.sensitive_features = sensitive_features
        self.max_sensitive_weight = max_sensitive_weight
        self.max_shap_rows = max_shap_rows
        self._shap_cache = {}

    def _get_shap(self, model_name, X):
        if model_name not in self._shap_cache:
            explainer = ShapExplainer(self.models[model_name], self.X_background,
                                      max_shap_rows=self.max_shap_rows)
            self._shap_cache[model_name] = explainer.explain(X)
        return self._shap_cache[model_name]

    def _validate_input(self, X):
        if X is None or len(X) == 0:
            raise ValueError(
                "ModelCheck: dataset vazio. Forneça um X com pelo menos 1 linha.")
        if hasattr(X, "isna") and X.isna().any().any():
            cols = list(X.columns[X.isna().any()])
            raise ValueError(
                f"ModelCheck: valores ausentes (NaN) nas colunas {cols}. "
                "Trate os NaN (imputação ou remoção) antes de rodar a auditoria.")
        if hasattr(X, "columns") and hasattr(self.X_background, "columns"):
            esperadas = list(self.X_background.columns)
            recebidas = list(X.columns)
            faltando = [c for c in esperadas if c not in recebidas]
            extras = [c for c in recebidas if c not in esperadas]
            if faltando or extras:
                msg = "ModelCheck: colunas de X não batem com o background."
                if faltando:
                    msg += f" Faltando: {faltando}."
                if extras:
                    msg += f" Extras: {extras}."
                raise ValueError(msg)

    def run(self, X, y=None, divergence_threshold=0.3, baseline_snapshots=None,
            drift_threshold=0.15):
        self._validate_input(X)
        self._shap_cache = {}
        report = {}

        comparator = ModelComparator(self.models, self.X_background,
                                    max_shap_rows=self.max_shap_rows)
        comparator.compute_all(X)
        report["comparison_table"] = comparator.build_comparison_table()
        report["divergences"] = comparator.find_divergences(divergence_threshold)

        if self.rule_set:
            report["contradictions"] = self._check_contradictions(X)

        if self.sensitive_features:
            report["fairness"] = self._check_fairness(X)

        report["leakage"] = self._check_leakage(X, y)

        if len(self.models) > 1:
            shap_all = {n: self._get_shap(n, X) for n in self.models}
            report["shape_divergences"] = ShapeDivergenceDetector().check(
                shap_all, X)

        report["snapshots"] = self._build_snapshots(X)

        if baseline_snapshots:
            report["drift"] = self._check_drift(
                baseline_snapshots, report["snapshots"], drift_threshold
            )

        return report

    def _check_contradictions(self, X):
        detector = ContradictionDetector(self.rule_set)
        return {
            name: detector.check(self._get_shap(name, X), X)
            for name in self.models
        }

    def _check_leakage(self, X, y):
        detector = LeakageDetector()
        return {
            name: detector.check(self._get_shap(name, X), X, y)
            for name in self.models
        }

    def _check_fairness(self, X):
        checker = FairnessChecker(
            self.sensitive_features, self.max_sensitive_weight
        )
        return {
            name: checker.check(self._get_shap(name, X), X)
            for name in self.models
        }

    def _build_snapshots(self, X):
        return {
            name: ExplanationSnapshot.create(self._get_shap(name, X), X)
            for name in self.models
        }

    def _check_drift(self, baselines, currents, threshold):
        detector = ExplanationDriftDetector(threshold)
        drift = {}
        for name, current in currents.items():
            baseline = baselines.get(name)
            if baseline:
                drift[name] = detector.compare(baseline, current)
        return drift


class ModelGuard(_ModelGuardLegacy):
    """
    Compatibilidade: mantém a API original `ModelGuard(models_dict, X_bg).run(X)`
    e, quando instanciada sem dicionário de modelos, delega para a API nova
    `Guard` (audit / check_features / audit_versions).
    """

    def __new__(cls, *args, **kwargs):
        legacy = (args and isinstance(args[0], dict)) or ('models' in kwargs)
        if legacy:
            return super().__new__(cls)
        from modelcheck.api import Guard
        return Guard(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        if (args and isinstance(args[0], dict)) or ('models' in kwargs):
            super().__init__(*args, **kwargs)
