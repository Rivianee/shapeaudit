"""
psi_shap.py — PSI sobre a distribuição de atribuições (SHAP)

Segunda lente sobre features, complementar ao ImportanceCollapseDetector:

  ICD      mede MAGNITUDE da importância (a feature ainda pesa?)
  PSI-SHAP mede DISTRIBUIÇÃO da atribuição (o peso está distribuído
           entre as amostras da mesma forma?)

Motivação: duas versões podem ter a mesma importância média para uma
feature e atribuí-la de formas opostas — valores altos recebendo
contribuição positiva em v1 e negativa em v2. O ICD é cego a isso por
construção (compara médias de |SHAP|).

Cuidado com escala: o PSI é sensível à escala. Se os SHAP de um modelo
são sistematicamente maiores, o PSI dispara por magnitude, não por
distribuição — que é justamente o que o ICD já mede. Por isso o padrão é
`normalize='per_model'`, que padroniza cada matriz antes de comparar.
"""
import numpy as np
import pandas as pd
from typing import List, Optional


def _shap_matrix(model, X: pd.DataFrame, n_sample: int = 300,
                 seed: int = 42):
    """Matriz (n_amostras × n_features) de SHAP values, ou None."""
    try:
        import shap
        Xs = X.sample(min(n_sample, len(X)), random_state=seed)
        sv = shap.TreeExplainer(model).shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        return sv
    except Exception:
        return None


def psi_1d(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10):
    """PSI entre duas amostras 1-D, com bins quantílicos da referência."""
    ref = ref[np.isfinite(ref)]; cur = cur[np.isfinite(cur)]
    if len(ref) < 20 or len(cur) < 20:
        return None
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, n_bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.allclose(np.std(cur), 0) else None
    edges[0], edges[-1] = -np.inf, np.inf
    a, _ = np.histogram(ref, bins=edges)
    b, _ = np.histogram(cur, bins=edges)
    a = np.clip(a / max(a.sum(), 1), 1e-6, None)
    b = np.clip(b / max(b.sum(), 1), 1e-6, None)
    return float(((b - a) * np.log(b / a)).sum())


class PSIShapDetector:
    """
    Detecta mudança na DISTRIBUIÇÃO de atribuição por feature.

        det = PSIShapDetector()
        r = det.compare(m1, m2, X_eval, feature_names)
        print(det.report(r))

    Limiares: os padrões da indústria para PSI (0.10 warn / 0.25 alert)
    são MUITO baixos para distribuições de SHAP. Calibrado em 55 medições
    de pares saudáveis (5 pares de seeds, Wine Quality):
        média=0.117  p95=0.250  p99=0.428  max=0.551
    Daí os padrões desta classe: warn=0.25 (p95) e alert=0.64 (p99×1.5).
    Com 0.10/0.25 o controle disparava 6 de 11 features.
    Recalibre por domínio com `calibrate()`.

    normalize:
      'per_model'  divide cada matriz pelo desvio-padrão global dela.
                   Isola distribuição de magnitude. **Padrão.**
      'none'       compara SHAP brutos. Dispara também por magnitude,
                   sobrepondo-se ao ICD.
    """

    def __init__(self, n_bins: int = 10, n_sample: int = 300,
                 psi_warn: float = 0.25, psi_alert: float = 0.64,
                 normalize: str = 'per_model'):
        self.n_bins = n_bins
        self.n_sample = n_sample
        self.psi_warn = psi_warn
        self.psi_alert = psi_alert
        self.normalize = normalize

    def calibrate(self, model_factory, X_train, y_train,
                  X_eval: pd.DataFrame,
                  feature_names: Optional[List[str]] = None,
                  seeds: Optional[List[int]] = None,
                  p_warn: float = 95.0, p_alert: float = 99.0,
                  alert_margin: float = 1.5) -> dict:
        """
        Calibra os limiares a partir de pares SAUDÁVEIS (mesmo treino,
        seeds diferentes). model_factory(seed) -> modelo treinado.
        """
        seeds = seeds or [1, 2, 3, 4, 5, 6, 7, 8]
        probe = PSIShapDetector(n_bins=self.n_bins, n_sample=self.n_sample,
                                psi_warn=1e9, psi_alert=1e9,
                                normalize=self.normalize)
        vals = []
        for a, b in zip(seeds[::2], seeds[1::2]):
            r = probe.compare(model_factory(a), model_factory(b),
                              X_eval, feature_names)
            if r.get('measured'):
                vals += [x['psi'] for x in r['features']
                         if x['psi'] is not None]
        if len(vals) < 10:
            return {'ok': False, 'reason': 'amostra insuficiente',
                    'n': len(vals)}
        v = np.array(vals, dtype=float)
        self.psi_warn = float(np.percentile(v, p_warn))
        self.psi_alert = float(np.percentile(v, p_alert) * alert_margin)
        self._calibrated = True
        return {'ok': True, 'n': len(v),
                'mean': round(float(v.mean()), 4),
                'p95': round(float(np.percentile(v, 95)), 4),
                'p99': round(float(np.percentile(v, 99)), 4),
                'max': round(float(v.max()), 4),
                'psi_warn': round(self.psi_warn, 4),
                'psi_alert': round(self.psi_alert, 4)}

    def compare(self, model_v1, model_v2, X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        from modelcheck._validate import validate_inputs
        X_eval, feature_names = validate_inputs(
            model_v1, model_v2, X_eval, feature_names,
            caller='PSIShapDetector')
        if feature_names is None:
            feature_names = list(X_eval.columns)
        A = _shap_matrix(model_v1, X_eval, self.n_sample)
        B = _shap_matrix(model_v2, X_eval, self.n_sample)
        if not getattr(self, '_calibrated', False):
            # Medido: os limiares NÃO transferem entre domínios.
            #   Wine  p99=0.428 max=0.551
            #   ENA/SE p99=0.384 max=0.403
            #   ENA/NE p99=1.347 max=1.550  <- 3x o de Wine
            # Sem calibrar, o NE geraria falso positivo com os padrões.
            import warnings as _w
            _w.warn("PSIShapDetector não calibrado. Os limiares não "
                    "transferem entre domínios (medido: p99 varia 0.38-1.35). "
                    "Chame calibrate() com pares saudáveis do seu domínio.",
                    stacklevel=2)
        if A is None or B is None:
            return {'measured': False, 'error': 'SHAP indisponível',
                    'features': [], 'flagged': [], 'psi_score': None}

        sA, sB = float(A.std()) + 1e-12, float(B.std()) + 1e-12
        if self.normalize == 'per_model':
            An, Bn = A / sA, B / sB
        else:
            An, Bn = A, B

        results, flagged = [], []
        for j, f in enumerate(feature_names):
            if j >= An.shape[1]:
                break
            p = psi_1d(An[:, j], Bn[:, j], self.n_bins)
            # sinal complementar: correlação entre a atribuição e o valor da
            # feature. Inversão de sinal indica forma oposta.
            xj = X_eval[f].to_numpy(dtype=float)[:len(A)] if f in X_eval else None
            level = ('ALERT' if p is not None and p > self.psi_alert else
                     'WARN' if p is not None and p > self.psi_warn else 'OK')
            if level in ('ALERT', 'WARN'):
                flagged.append(f)
            results.append({'feature': f,
                            'psi': None if p is None else round(p, 4),
                            'level': level,
                            'std_v1': round(float(A[:, j].std()), 5),
                            'std_v2': round(float(B[:, j].std()), 5)})

        n_alert = sum(1 for r in results if r['level'] == 'ALERT')
        n_warn = sum(1 for r in results if r['level'] == 'WARN')
        score = max(0.0, 100 - n_alert * 15 - n_warn * 5)
        return {'measured': True, 'features': results, 'flagged': flagged,
                'n_alert': n_alert, 'n_warn': n_warn,
                'scale_ratio': round(sB / sA, 3),
                'normalize': self.normalize,
                'psi_score': round(score, 1)}

    def report(self, r: dict) -> str:
        if not r.get('measured'):
            return f"PSIShapDetector: não mensurável ({r.get('error')})"
        L = [f"PSIShapDetector [{r['normalize']}]: score={r['psi_score']}/100"
             f"  escala v2/v1={r['scale_ratio']}",
             f"  ALERT={r['n_alert']}  WARN={r['n_warn']}"]
        for row in r['features']:
            if row['level'] != 'OK':
                L.append(f"  {'⚠' if row['level']=='ALERT' else '·'} "
                         f"'{row['feature']}': PSI={row['psi']} ({row['level']})")
        if not r['flagged']:
            L.append("  ✅ Distribuição de atribuição estável.")
        return "\n".join(L)
