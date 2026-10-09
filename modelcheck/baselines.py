"""
baselines.py — Detectores equivalentes às ferramentas existentes

Motivação (Revisão Crítica, lacuna D):
  Sem comparação com Evidently / Alibi Detect / NannyML / DeepChecks não é
  possível afirmar que o ModelGuard detecta algo novo.

Este módulo reimplementa os *mecanismos centrais* dessas ferramentas, para
comparação controlada no mesmo conjunto de cenários. Não substitui as
bibliotecas — permite um benchmark reprodutível sem depender de versões.

  KSDriftDetector       ~ Evidently (Data Drift), Alibi Detect (KS)
  MMDDriftDetector      ~ Alibi Detect (MMD)
  PSIDriftDetector      ~ Evidently (PSI), padrão da indústria de crédito
  PerformanceMonitor    ~ NannyML (performance monitoring)
  CorrelationDrift      ~ DeepChecks (feature correlation drift)
  PredictionDrift        ~ Evidently (Target/Prediction Drift)

Ponto central do argumento: todos estes comparam DISTRIBUIÇÕES (de X, de y
ou de ŷ) ou MÉTRICAS. Nenhum compara a RELAÇÃO FUNCIONAL aprendida. Quando
v1 e v2 são avaliados no MESMO X, P(X) é idêntico por construção — logo os
detectores de data drift não podem, por definição, alertar.
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats as scipy_stats


class KSDriftDetector:
    """Kolmogorov-Smirnov por feature (Evidently / Alibi Detect)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare(self, X_ref: pd.DataFrame, X_cur: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        if feature_names is None:
            feature_names = list(X_ref.columns)
        flagged = []
        for f in feature_names:
            _, p = scipy_stats.ks_2samp(X_ref[f].dropna(), X_cur[f].dropna())
            if p < self.alpha:
                flagged.append(f)
        n = len(feature_names)
        return {'tool': 'KS (Evidently/Alibi)', 'drifted': flagged,
                'n_drifted': len(flagged),
                'score': round(100 * (1 - len(flagged) / max(n, 1)), 1),
                'alert': len(flagged) > 0}


class MMDDriftDetector:
    """Maximum Mean Discrepancy com kernel RBF (Alibi Detect)."""

    def __init__(self, n_perm: int = 100, alpha: float = 0.05,
                 n_sample: int = 200):
        self.n_perm, self.alpha, self.n_sample = n_perm, alpha, n_sample

    def _mmd2(self, A, B, gamma):
        Kaa = np.exp(-gamma * ((A[:, None] - A[None]) ** 2).sum(-1))
        Kbb = np.exp(-gamma * ((B[:, None] - B[None]) ** 2).sum(-1))
        Kab = np.exp(-gamma * ((A[:, None] - B[None]) ** 2).sum(-1))
        return Kaa.mean() + Kbb.mean() - 2 * Kab.mean()

    def compare(self, X_ref: pd.DataFrame, X_cur: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        if feature_names is None:
            feature_names = list(X_ref.columns)
        n = min(self.n_sample, len(X_ref), len(X_cur))
        A = X_ref[feature_names].sample(n, random_state=0).values.astype(float)
        B = X_cur[feature_names].sample(n, random_state=1).values.astype(float)
        sd = np.concatenate([A, B]).std(0) + 1e-8
        A, B = A / sd, B / sd
        gamma = 1.0 / A.shape[1]
        obs = self._mmd2(A, B, gamma)
        pool = np.concatenate([A, B])
        rng = np.random.RandomState(0)
        cnt = 0
        for _ in range(self.n_perm):
            p = rng.permutation(len(pool))
            if self._mmd2(pool[p[:n]], pool[p[n:]], gamma) >= obs:
                cnt += 1
        pval = (cnt + 1) / (self.n_perm + 1)
        return {'tool': 'MMD (Alibi Detect)', 'mmd2': round(float(obs), 6),
                'p_value': round(float(pval), 4),
                'score': 100.0 if pval >= self.alpha else 0.0,
                'alert': bool(pval < self.alpha)}


class PSIDriftDetector:
    """Population Stability Index (Evidently / indústria de crédito)."""

    def __init__(self, n_bins: int = 10, threshold: float = 0.2):
        self.n_bins, self.threshold = n_bins, threshold

    def compare(self, X_ref: pd.DataFrame, X_cur: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        if feature_names is None:
            feature_names = list(X_ref.columns)
        psis, flagged = {}, []
        for f in feature_names:
            edges = np.unique(np.percentile(
                X_ref[f].dropna(), np.linspace(0, 100, self.n_bins + 1)))
            if len(edges) < 3:
                continue
            a, _ = np.histogram(X_ref[f].dropna(), bins=edges)
            b, _ = np.histogram(X_cur[f].dropna(), bins=edges)
            a = a / max(a.sum(), 1); b = b / max(b.sum(), 1)
            a = np.clip(a, 1e-6, None); b = np.clip(b, 1e-6, None)
            psi = float(((b - a) * np.log(b / a)).sum())
            psis[f] = round(psi, 4)
            if psi > self.threshold:
                flagged.append(f)
        return {'tool': 'PSI (Evidently)', 'psi': psis, 'drifted': flagged,
                'n_drifted': len(flagged),
                'score': round(100 * (1 - len(flagged) / max(len(psis), 1)), 1),
                'alert': len(flagged) > 0}


class PerformanceMonitor:
    """Monitoramento de performance (NannyML)."""

    def __init__(self, rel_threshold: float = 0.05):
        self.rel_threshold = rel_threshold

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame, y_eval) -> dict:
        p1 = np.asarray(model_v1.predict(X_eval), dtype=float)
        p2 = np.asarray(model_v2.predict(X_eval), dtype=float)
        y = np.asarray(y_eval, dtype=float)
        rmse1 = float(np.sqrt(((p1 - y) ** 2).mean()))
        rmse2 = float(np.sqrt(((p2 - y) ** 2).mean()))
        var = float(np.var(y)) + 1e-12
        r2_1 = 1 - ((p1 - y) ** 2).mean() / var
        r2_2 = 1 - ((p2 - y) ** 2).mean() / var
        rel = abs(rmse2 - rmse1) / max(rmse1, 1e-9)
        return {'tool': 'Performance (NannyML)',
                'rmse_v1': round(rmse1, 4), 'rmse_v2': round(rmse2, 4),
                'r2_v1': round(float(r2_1), 4), 'r2_v2': round(float(r2_2), 4),
                'delta_r2': round(abs(float(r2_1 - r2_2)), 4),
                'rel_change': round(rel, 4),
                'score': round(max(0.0, 100 * (1 - rel / max(self.rel_threshold, 1e-9) * 0.5)), 1)
                         if rel < 2 * self.rel_threshold else 0.0,
                'alert': bool(rel > self.rel_threshold)}


class PredictionDriftDetector:
    """Drift na distribuição das predições (Evidently Target Drift)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare(self, model_v1, model_v2, X_eval: pd.DataFrame) -> dict:
        p1 = np.asarray(model_v1.predict(X_eval), dtype=float)
        p2 = np.asarray(model_v2.predict(X_eval), dtype=float)
        _, p_ks = scipy_stats.ks_2samp(p1, p2)
        w = float(scipy_stats.wasserstein_distance(p1, p2))
        return {'tool': 'Prediction Drift (Evidently)',
                'ks_p': round(float(p_ks), 4),
                'wasserstein': round(w, 4),
                'score': 100.0 if p_ks >= self.alpha else 0.0,
                'alert': bool(p_ks < self.alpha)}


class CorrelationDriftDetector:
    """Drift na matriz de correlação features×predição (DeepChecks)."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold

    def compare(self, model_v1, model_v2, X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        if feature_names is None:
            feature_names = list(X_eval.columns)
        p1 = np.asarray(model_v1.predict(X_eval), dtype=float)
        p2 = np.asarray(model_v2.predict(X_eval), dtype=float)
        shifts, flagged = {}, []
        for f in feature_names:
            x = X_eval[f].values.astype(float)
            if np.std(x) < 1e-12:
                continue
            c1 = float(np.corrcoef(x, p1)[0, 1])
            c2 = float(np.corrcoef(x, p2)[0, 1])
            d = abs(c1 - c2)
            shifts[f] = round(d, 4)
            if d > self.threshold:
                flagged.append(f)
        return {'tool': 'Correlation Drift (DeepChecks)',
                'shifts': shifts, 'drifted': flagged,
                'n_drifted': len(flagged),
                'score': round(100 * (1 - len(flagged) / max(len(shifts), 1)), 1),
                'alert': len(flagged) > 0}


def run_all_baselines(model_v1, model_v2, X_eval, y_eval=None,
                      feature_names=None) -> dict:
    """Roda todos os baselines. X_ref == X_cur (mesmo conjunto de avaliação)."""
    if feature_names is None:
        feature_names = list(X_eval.columns)
    out = {}
    out['ks']          = KSDriftDetector().compare(X_eval, X_eval, feature_names)
    out['psi']         = PSIDriftDetector().compare(X_eval, X_eval, feature_names)
    out['mmd']         = MMDDriftDetector(n_perm=50, n_sample=150).compare(
                             X_eval, X_eval, feature_names)
    out['pred_drift']  = PredictionDriftDetector().compare(model_v1, model_v2, X_eval)
    out['corr_drift']  = CorrelationDriftDetector().compare(
                             model_v1, model_v2, X_eval, feature_names)
    if y_eval is not None:
        out['performance'] = PerformanceMonitor().compare(
                                 model_v1, model_v2, X_eval, y_eval)
    out['any_alert'] = any(v.get('alert') for v in out.values()
                           if isinstance(v, dict))
    return out
