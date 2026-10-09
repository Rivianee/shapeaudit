"""Integração MLflow: loga resultados do modelcheck no run ativo."""
import json
import tempfile
import os


def log_to_mlflow(report, dashboard=True):
    """Loga métricas + artefatos no run MLflow ativo."""
    import mlflow

    n_contra = sum(len(v) for v in report.get("contradictions", {}).values())
    n_drift = sum(len(v) for v in report.get("drift", {}).values())
    n_fair = sum(len(v) for v in report.get("fairness", {}).values())
    n_div = len(report.get("divergences", []))

    mlflow.log_metrics({
        "modelcheck_contradictions": n_contra,
        "modelcheck_drift_alerts": n_drift,
        "modelcheck_fairness_violations": n_fair,
        "modelcheck_divergences": n_div,
    })

    with tempfile.TemporaryDirectory() as tmp:
        json_path = os.path.join(tmp, "modelcheck_report.json")
        serializable = {
            k: v for k, v in report.items() if k != "comparison_table"
        }
        serializable["comparison_table"] = report["comparison_table"].to_dict()
        with open(json_path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        mlflow.log_artifact(json_path)

        if dashboard:
            from modelcheck.dashboard import generate_dashboard
            dash_path = os.path.join(tmp, "modelcheck_dashboard.html")
            generate_dashboard(report, dash_path)
            mlflow.log_artifact(dash_path)

    return {"contradictions": n_contra, "drift": n_drift,
            "fairness": n_fair, "divergences": n_div}
