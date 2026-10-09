"""Ranking de modelos por coerência com as regras dos dados."""


def rank_models(report):
    """Menos contradições (ponderadas por severidade) = mais coerente."""
    weights = {"high": 3, "medium": 2, "low": 1, "check": 1, "info": 0}
    scores = {}
    for model, items in report.get("contradictions", {}).items():
        scores[model] = sum(weights.get(c["severity"], 1) for c in items)
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    result = [{"model": m, "incoherence_score": s,
               "rank": i + 1} for i, (m, s) in enumerate(ranked)]
    if result:
        best = result[0]["model"]
        result[0]["summary"] = (
            f"Dos {len(result)} modelos testados, '{best}' é o mais "
            "alinhado com as regras dos dados."
        )
    return result
