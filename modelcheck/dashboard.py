"""Dashboard HTML self-contained do relatório modelcheck."""
import json


def _severity_color(sev):
    return {"high": "#d62728", "medium": "#ff9900", "low": "#999"}.get(sev, "#999")


def generate_dashboard(report, save_path="modelcheck_dashboard.html"):
    table = report["comparison_table"]
    normalized = table.div(table.max(axis=0), axis=1).fillna(0)

    contradicted = {}
    for model, items in report.get("contradictions", {}).items():
        for c in items:
            contradicted[(model, c["feature"])] = c

    # barras de comparação
    bars_html = ""
    for feature in normalized.index:
        bars_html += f"<div class='feature'><b>{feature}</b>"
        for model in normalized.columns:
            v = normalized.loc[feature, model]
            key = (model, feature)
            color = "#d62728" if key in contradicted else "#1f77b4"
            tag = " ✗ contradição" if key in contradicted else ""
            bars_html += (
                f"<div class='row'><span class='mname'>{model}</span>"
                f"<div class='bar' style='width:{v*100:.0f}%;background:{color}'></div>"
                f"<span class='tag'>{v:.2f}{tag}</span></div>"
            )
        bars_html += "</div>"

    # cards
    def card(title, items, formatter, empty="Nenhum(a). ✅"):
        inner = "".join(formatter(i) for i in items) if items else f"<p>{empty}</p>"
        return f"<div class='card'><h2>{title}</h2>{inner}</div>"

    contra_items = [
        {"model": m, **c}
        for m, cs in report.get("contradictions", {}).items() for c in cs
    ]
    # v1.66: o FairnessChecker devolve dict; itera sobre `violations`.
    # Aceita o formato antigo (lista) para não quebrar quem passa o dado
    # direto.
    def _viol(b):
        return b.get("violations", []) if isinstance(b, dict) else (b or [])

    fair_items = [
        {"model": m, **v}
        for m, bruto in report.get("fairness", {}).items()
        for v in _viol(bruto)
    ]
    drift_items = [
        {"model": m, **d}
        for m, ds in report.get("drift", {}).items() for d in ds
    ]

    contra_html = card(
        "Contradições vs regras de negócio", contra_items,
        lambda c: (
            f"<p style='color:{_severity_color(c['severity'])}'>"
            f"✗ [{c['severity'].upper()}] {c['model']} / {c['feature']}: "
            f"esperado {c['expected']}, encontrado {c['found']} "
            f"(força {c['strength']})<br><small>{c['rule_description']}</small></p>"
        ),
    )
    fair_html = card(
        "Fairness", fair_items,
        lambda v: (
            f"<p style='color:#d62728'>✗ {v['model']}: '{v['feature']}' "
            f"pesa {v['weight']:.0%} (limite {v['max_allowed']:.0%})</p>"
        ),
    ) if "fairness" in report else ""
    drift_html = card(
        "Drift de explicação", drift_items,
        lambda d: (
            f"<p style='color:#ff9900'>⚠ {d['model']}: '{d['feature']}' "
            f"{d['baseline']:.0%} → {d['current']:.0%}</p>"
        ),
    ) if "drift" in report else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>modelcheck</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:30px auto;color:#222;background:#fafafa}}
h1{{border-bottom:3px solid #1f77b4;padding-bottom:8px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:16px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
.feature{{margin:12px 0}}
.row{{display:flex;align-items:center;gap:8px;margin:2px 0}}
.mname{{width:140px;font-size:13px;color:#555}}
.bar{{height:14px;border-radius:3px;min-width:2px}}
.tag{{font-size:12px;color:#d62728;font-weight:bold}}
</style></head><body>
<h1>modelcheck — relatório</h1>
<div class='card'><h2>Comparação de modelos (SHAP normalizado)</h2>{bars_html}</div>
{contra_html}{fair_html}{drift_html}
</body></html>"""

    with open(save_path, "w") as f:
        f.write(html)
    return save_path
