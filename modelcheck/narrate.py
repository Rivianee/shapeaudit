"""Relatório narrado: frases pra stakeholder, não tabela."""


def narrate(report):
    lines = []
    for model, items in report.get("contradictions", {}).items():
        for c in items:
            if c["severity"] == "info":
                continue
            sev = {"high": "ALTA", "medium": "MÉDIA", "low": "baixa",
                   "check": "a revisar"}[c["severity"]]
            lines.append(
                f"O modelo '{model}' trata '{c['feature']}' na direção "
                f"{'positiva' if c['found']=='positive' else 'negativa'}, mas os "
                f"dados históricos indicam o contrário. Severidade {sev}."
            )
            for s in c.get("suggestions", [])[:2]:
                lines.append(f"  → {s}")
    for model, items in report.get("fairness", {}).items():
        for v in items:
            lines.append(
                f"A feature sensível '{v['feature']}' responde por "
                f"{v['weight']:.0%} das decisões do modelo '{model}' "
                f"(limite: {v['max_allowed']:.0%}). Avalie impacto discriminatório."
            )
    for model, items in report.get("drift", {}).items():
        for d in items:
            lines.append(
                f"O peso de '{d['feature']}' no modelo '{model}' mudou de "
                f"{d['baseline']:.0%} para {d['current']:.0%} desde o baseline. "
                f"Investigue mudança nos dados ou no comportamento."
            )
    if not lines:
        lines.append("Nenhum problema relevante detectado. Modelos coerentes "
                     "com os dados e regras conhecidas.")
    return "\n".join(lines)
