"""Gate de CI/CD: reprova deploy se modelo falhar nas checagens."""
import sys


def ci_gate(report, fail_on_severity="high", fail_on_drift=True,
            fail_on_fairness=True):
    """Retorna (passou, motivos). Usar em pipeline: exit(1) se reprovar."""
    reasons = []
    levels = {"low": 0, "medium": 1, "high": 2}
    min_level = levels[fail_on_severity]

    for model, items in report.get("contradictions", {}).items():
        for c in items:
            if levels[c["severity"]] >= min_level:
                reasons.append(
                    f"contradição [{c['severity']}] {model}/{c['feature']}"
                )

    if fail_on_drift:
        for model, items in report.get("drift", {}).items():
            for d in items:
                reasons.append(f"drift {model}/{d['feature']}")

    if fail_on_fairness:
        for model, items in report.get("fairness", {}).items():
            for v in items:
                reasons.append(f"fairness {model}/{v['feature']}")

    return len(reasons) == 0, reasons


def run_gate(report, **kwargs):
    """Imprime resultado e encerra com exit code pra CI."""
    passed, reasons = ci_gate(report, **kwargs)
    if passed:
        print("✅ modelcheck: aprovado para deploy")
        sys.exit(0)
    print("❌ modelcheck: deploy bloqueado")
    for r in reasons:
        print(f"  - {r}")
    sys.exit(1)
