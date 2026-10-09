"""
ale_selective.py — SDD via ALE com bootstrap seletivo

Problema medido (v1.23): o ALE vetorizado tem recall 1.00 mas
especificidade 0.17 — dispara divergência até em `v1 vs v1`.
Causa: classificar forma sobre ~5 pontos de perfil é instável.

Estratégia (bootstrap seletivo):
  Passo 1 — triagem rápida: ALE vetorizado sem bootstrap em TODAS as
            features. Barato.
  Passo 2 — confirmação: bootstrap APENAS nas features que a triagem
            marcou. Caro, mas em poucas features.
  Passo 3 — confirma a divergência somente se a forma se mantiver
            estável em >= min_agreement das réplicas em AMBOS os modelos.

Custo: proporcional ao nº de suspeitas, não ao nº de features.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from collections import Counter


def _shape_of(model, X, feature):
    from modelcheck.shape import _shape_profile_small_n
    from modelcheck.ale_fast import ale_curve_fast
    out = ale_curve_fast(model, X, feature)
    if out is None:
        return None
    p = _shape_profile_small_n(out[0], out[1], min_n=3)
    return p['shape'] if p else None


def _amplitude(model, X, feature):
    from modelcheck.ale_fast import ale_curve_fast
    out = ale_curve_fast(model, X, feature)
    return 0.0 if out is None else float(np.ptp(out[1]))


class SelectiveALEDetector:
    """
    SDD via ALE com confirmação por bootstrap apenas nas suspeitas.

        det = SelectiveALEDetector()
        res = det.compare(m1, m2, X_eval, feature_names)
        print(det.report(res))
    """

    def __init__(self, n_boot: int = 12,
                 frac: float = 0.75,
                 min_agreement: float = 0.70,
                 min_amplitude_ratio: float = 0.05,
                 seed: int = 0):
        self.n_boot = n_boot
        self.frac = frac
        self.min_agreement = min_agreement
        self.min_amplitude_ratio = min_amplitude_ratio
        self.seed = seed

    def _stable_shape(self, model, X, feature):
        """Forma modal e concordância entre réplicas."""
        rng = np.random.RandomState(self.seed)
        n = max(int(len(X) * self.frac), 40)
        shapes = []
        for _ in range(self.n_boot):
            idx = rng.choice(len(X), min(n, len(X)), replace=False)
            s = _shape_of(model, X.iloc[idx].reset_index(drop=True), feature)
            if s:
                shapes.append(s)
        if not shapes:
            return None, 0.0
        sh, c = Counter(shapes).most_common(1)[0]
        return sh, c / len(shapes)

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        from modelcheck._validate import validate_inputs
        X_eval, feature_names = validate_inputs(
            model_v1, model_v2, X_eval, feature_names,
            caller='SelectiveALEDetector')
        if feature_names is None:
            feature_names = list(X_eval.columns)

        # ── Passo 1: triagem rápida ──────────────────────────────────
        screened = []
        amps = {}
        for f in feature_names:
            s1 = _shape_of(model_v1, X_eval, f)
            s2 = _shape_of(model_v2, X_eval, f)
            a1 = _amplitude(model_v1, X_eval, f)
            a2 = _amplitude(model_v2, X_eval, f)
            amps[f] = (a1, a2)
            if s1 and s2 and s1 != s2:
                screened.append((f, s1, s2))

        # amplitude de referência: maior efeito observado
        max_amp = max([max(a) for a in amps.values()] or [1e-9])

        # ── Passo 2/3: confirmação só nas suspeitas ──────────────────
        confirmed, rejected = [], []
        for f, s1_fast, s2_fast in screened:
            a1, a2 = amps[f]
            # descarta features com efeito irrelevante (ruído numérico)
            if max(a1, a2) / max_amp < self.min_amplitude_ratio:
                rejected.append({'feature': f, 'reason': 'LOW_AMPLITUDE',
                                 'shape_v1': s1_fast, 'shape_v2': s2_fast,
                                 'amp_ratio': round(max(a1, a2) / max_amp, 4)})
                continue
            sh1, c1 = self._stable_shape(model_v1, X_eval, f)
            sh2, c2 = self._stable_shape(model_v2, X_eval, f)
            if sh1 is None or sh2 is None:
                rejected.append({'feature': f, 'reason': 'UNSTABLE_PROFILE',
                                 'shape_v1': s1_fast, 'shape_v2': s2_fast})
                continue
            if c1 < self.min_agreement or c2 < self.min_agreement:
                rejected.append({'feature': f, 'reason': 'LOW_AGREEMENT',
                                 'shape_v1': sh1, 'shape_v2': sh2,
                                 'conf_v1': round(c1, 2), 'conf_v2': round(c2, 2)})
                continue
            if sh1 == sh2:
                rejected.append({'feature': f, 'reason': 'NOT_CONFIRMED',
                                 'shape_v1': sh1, 'shape_v2': sh2,
                                 'conf_v1': round(c1, 2), 'conf_v2': round(c2, 2)})
                continue
            confirmed.append({'feature': f, 'shape_v1': sh1, 'shape_v2': sh2,
                              'conf_v1': round(c1, 2), 'conf_v2': round(c2, 2),
                              'amp_v1': round(a1, 5), 'amp_v2': round(a2, 5)})

        score = max(0, 100 - len(confirmed) * 15)
        return {
            'divergences':   confirmed,
            'screened':      [f for f, _, _ in screened],
            'rejected':      rejected,
            'n_screened':    len(screened),
            'n_confirmed':   len(confirmed),
            'n_features':    len(feature_names),
            'health_score':  score,
            'method':        'ALE + bootstrap seletivo',
            'bootstrap_calls': len(screened) * self.n_boot * 2,
        }

    def report(self, r: dict) -> str:
        L = [f"SelectiveALEDetector ({r['method']}):",
             f"  Triagem: {r['n_screened']}/{r['n_features']} suspeitas  →  "
             f"confirmadas: {r['n_confirmed']}",
             f"  Chamadas de bootstrap: {r['bootstrap_calls']} "
             f"(vs {r['n_features'] * 24} do bootstrap total)",
             f"  Health Score: {r['health_score']}/100"]
        for d in r['divergences']:
            L.append(f"  ⚠ '{d['feature']}': {d['shape_v1']} → {d['shape_v2']} "
                     f"(conf {d['conf_v1']:.0%}/{d['conf_v2']:.0%})")
        for x in r['rejected']:
            L.append(f"  ✓ '{x['feature']}' descartada: {x['reason']}")
        if not r['divergences'] and not r['rejected']:
            L.append("  ✅ Nenhuma suspeita na triagem.")
        return "\n".join(L)
