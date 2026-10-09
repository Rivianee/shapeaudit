"""Relatório legível do ModelGuard."""


def summary(report):
    lines = ["=" * 50, "MODELCHECK REPORT", "=" * 50]

    lines.append("\n[Divergências entre modelos]")
    if not report["divergences"]:
        lines.append("  Nenhuma. Modelos concordam nas features.")
    for d in report["divergences"]:
        lines.append(f"  ⚠ {d['feature']}: spread={d['spread']}")
        for m, v in d["values"].items():
            lines.append(f"      {m}: {v:.3f}")

    contradictions = report.get("contradictions", {})
    lines.append("\n[Contradições com regras de negócio]")
    total = sum(len(c) for c in contradictions.values())
    if total == 0:
        lines.append("  Nenhuma.")
    for model, items in contradictions.items():
        for c in items:
            lines.append(
                f"  ✗ [{c['severity'].upper()}] {model} / {c['feature']}: "
                f"esperado {c['expected']}, encontrado {c['found']} "
                f"(força {c['strength']})"
            )
            lines.append(f"      Regra: {c['rule_description']}")

    _fairness_section(report, lines)
    _drift_section(report, lines)

    lines.append("=" * 50)
    return "\n".join(lines)


def _fairness_section(report, lines):
    fairness = report.get("fairness")
    if fairness is None:
        return
    lines.append("\n[Fairness — features sensíveis]")
    # v1.66: o FairnessChecker passou a devolver dict (com status, weights,
    # violations) em vez de lista. Aceita os dois formatos — código de
    # terceiros pode estar passando o antigo.
    def _viol(v):
        return v.get("violations", []) if isinstance(v, dict) else (v or [])

    total = sum(len(_viol(v)) for v in fairness.values())
    if total == 0:
        lines.append("  OK. Nenhuma feature sensível acima do limite.")
    for model, _bruto in fairness.items():
        items = _viol(_bruto)
        for v in items:
            lines.append(
                f"  ✗ {model}: '{v['feature']}' pesa {v['weight']:.0%} "
                f"(limite {v['max_allowed']:.0%})"
            )


def _drift_section(report, lines):
    drift = report.get("drift")
    if drift is None:
        return
    lines.append("\n[Drift de explicação]")
    total = sum(len(v) for v in drift.values())
    if total == 0:
        lines.append("  OK. Explicações estáveis vs baseline.")
    for model, items in drift.items():
        for d in items:
            lines.append(
                f"  ⚠ {model}: '{d['feature']}' mudou de "
                f"{d['baseline']:.0%} para {d['current']:.0%}"
            )
