"""Model Health Score: nota única de 0 a 100 combinando os detectores."""

_DATA_PEN = {"CRITICO": 10, "CRÍTICO": 10, "ALERTA": 5, "INFO": 1}


def _cap(v, cap):
    return min(v, cap)


def health_score(guard_report=None, data_scan=None, diff_result=None):
    """Combina resultados em um score 0-100 com breakdown.

    guard_report: saída de ModelGuard.run()
    data_scan:    saída de dataspec.core.scan()
    diff_result:  saída de explaindiff.core.diff()
    """
    breakdown = {}
    total_pen = 0

    if data_scan:
        pen = sum(_DATA_PEN.get(a.get("severity", "INFO").upper(), 1)
                  for a in data_scan.get("alerts", []))
        pen = _cap(pen, 40)
        breakdown["dados"] = -pen
        total_pen += pen

    if guard_report:
        pen = 0
        pen += 8 * len(guard_report.get("contradictions", []) or [])
        pen += 3 * len(guard_report.get("divergences", []) or [])
        pen += 2 * len(guard_report.get("shape_divergences", []) or [])
        leak = guard_report.get("leakage", {}) or {}
        for r in leak.values():
            alerts = r.get("alerts", r) if isinstance(r, dict) else r
            if alerts:
                pen += 10
        fair = guard_report.get("fairness", {}) or {}
        for r in fair.values():
            viol = r.get("violations", r) if isinstance(r, dict) else r
            if viol:
                pen += 10
        pen = _cap(pen, 50)
        breakdown["modelo"] = -pen
        total_pen += pen

    if diff_result:
        pen = 0
        pen += 5 * sum(1 for c in diff_result.get("weight_changes", [])
                       if c.get("significant"))
        pen += 8 * len(diff_result.get("direction_changes", []) or [])
        pen += 4 * len(diff_result.get("shape_changes", []) or [])
        pen = _cap(pen, 30)
        breakdown["mudancas"] = -pen
        total_pen += pen

    score = max(0, 100 - total_pen)
    if score >= 80:
        status, cor = "SAUDAVEL", "verde"
    elif score >= 60:
        status, cor = "ATENCAO", "amarelo"
    else:
        status, cor = "CRITICO", "vermelho"
    return {"score": score, "status": status, "cor": cor,
            "breakdown": breakdown}
