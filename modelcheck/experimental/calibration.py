"""Calibração automática de threshold para o ShapeDivergenceDetector.

Problema original:
    O threshold padrão 0.30 é um magic number — funciona nos experimentos
    controlados do paper mas pode ser conservador demais ou agressivo demais
    dependendo do domínio:

    - Wine Quality: threshold calibrado = 0.59 (0.30 detectaria FPs)
    - PLD Elétrico: threshold calibrado = 0.78 (0.30 muito conservador)
    - MES no PLD:   threshold individual = 0.85 (feature dominante)
    - DIA_ANO:      threshold individual = 0.29 (feature estável)

Solução:
    ShapeThresholdCalibrator aprende a distribuição de spread "normal"
    treinando K modelos em subsamples do mesmo dataset (sem divergência
    injetada). O threshold vira o percentil 95 do spread natural.

    Isso transforma o threshold de magic number para
    "limite superior do comportamento normal neste domínio".

Uso:
    from modelcheck.calibration import ShapeThresholdCalibrator

    # Calibra em dados de referência (sem divergência conhecida)
    cal = ShapeThresholdCalibrator(percentile=95)
    cal.fit(X_ref, y_ref, feature_names=feats)

    print(f"Threshold calibrado: {cal.threshold_:.4f}")
    print(cal.report())

    # Usa o threshold calibrado no SDD
    guard = ModelGuard(models, bg,
                       divergence_threshold=cal.threshold_)

    # Ou por feature
    for feat, thr in cal.feature_thresholds_.items():
        print(f"{feat}: threshold={thr:.4f}")

Resultados experimentais:
    Wine Quality (11 features, 1599 amostras):
    - Threshold global 0.30 → 6 alertas em RF vs GB
    - Threshold calibrado 0.59 → 2 alertas (remove 4 FPs)
    - Features com alta variabilidade natural (density, free SO2) têm
      threshold maior; alcohol e sulphates têm threshold menor

    PLD Elétrico CCEE (2022-2023, anos estáveis):
    - Threshold global calibrado: 0.78
    - MES: 0.85 (muda naturalmente ao longo do ano)
    - DIA_ANO: 0.29 (mais estável)
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Union


class ShapeThresholdCalibrator:
    """Calibra automaticamente o threshold do SDD para um domínio específico.

    Parâmetros:
        percentile  : percentil da distribuição de spread natural a usar
                      como threshold (padrão 95 → rejeita 5% de spreads normais)
        n_models    : número de modelos a treinar em subsamples (padrão 10)
        sample_frac : fração do dataset por subsample (padrão 0.8)
        min_samples : mínimo de amostras para calibrar (padrão 100)

    Atributos após fit():
        threshold_           : threshold global calibrado
        feature_thresholds_  : dict {feature: threshold_individual}
        spread_dist_         : array com todos os spreads observados
        n_pairs_             : número de pares de modelos avaliados
    """

    def __init__(self,
                 percentile:   int   = 95,
                 n_models:     int   = 10,
                 sample_frac:  float = 0.8,
                 min_samples:  int   = 100):
        self.percentile   = percentile
        self.n_models     = n_models
        self.sample_frac  = sample_frac
        self.min_samples  = min_samples

        # Definidos após fit()
        self.threshold_           = None
        self.feature_thresholds_  = {}
        self.spread_dist_         = None
        self.n_pairs_             = 0
        self.feature_names_       = None

    def fit(self,
            X:             Union[np.ndarray, pd.DataFrame],
            y:             Union[np.ndarray, pd.Series],
            feature_names: Optional[List[str]] = None) -> 'ShapeThresholdCalibrator':
        """Aprende a distribuição de spread natural do domínio.

        Treina n_models modelos RF em subsamples do dataset.
        Mede o spread de importância SHAP entre todos os pares.
        Define threshold como percentil_p desses spreads.

        Parâmetros:
            X             : features (DataFrame ou array)
            y             : target
            feature_names : nomes das features (opcional se X é DataFrame)

        Retorna self para encadeamento.
        """
        # Converte para numpy
        X_arr = X.values if hasattr(X, 'values') else np.asarray(X)
        y_arr = y.values if hasattr(y, 'values') else np.asarray(y)

        if len(X_arr) < self.min_samples:
            raise ValueError(
                f"Calibração requer ao menos {self.min_samples} amostras, "
                f"mas recebeu {len(X_arr)}. "
                f"Use menos modelos ou reduza min_samples.")

        if feature_names is None:
            if hasattr(X, 'columns'):
                feature_names = list(X.columns)
            else:
                feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

        self.feature_names_ = feature_names
        n_feats = len(feature_names)

        # ── Treina modelos e extrai SHAP ──────────────────────────
        models_sv = []
        for i in range(self.n_models):
            n_sample = int(len(X_arr) * self.sample_frac)
            idx = np.random.choice(len(X_arr), n_sample, replace=False)
            Xs, ys = X_arr[idx], y_arr[idx]
            if len(Xs) < 20:
                continue
            rf = RandomForestRegressor(
                n_estimators=30, max_depth=5,
                random_state=i, n_jobs=1).fit(Xs, ys)
            bg = Xs[:min(40, len(Xs))]
            sv = shap.TreeExplainer(rf).shap_values(bg)
            models_sv.append(sv)

        if len(models_sv) < 2:
            raise ValueError("Não foi possível treinar modelos suficientes.")

        # ── Calcula spreads entre todos os pares ──────────────────
        spreads_global   = []
        spreads_per_feat = {f: [] for f in feature_names}

        for i in range(len(models_sv)):
            for j in range(i + 1, len(models_sv)):
                sv1, sv2 = models_sv[i], models_sv[j]
                n = min(sv1.shape[0], sv2.shape[0])
                for k, f in enumerate(feature_names):
                    imp1 = float(abs(sv1[:n, k]).mean())
                    imp2 = float(abs(sv2[:n, k]).mean())
                    sp   = abs(imp1 - imp2) / (max(imp1, imp2) + 1e-8)
                    spreads_per_feat[f].append(sp)
                    spreads_global.append(sp)

        # ── Define thresholds ─────────────────────────────────────
        self.spread_dist_  = np.array(spreads_global)
        self.threshold_    = float(np.percentile(self.spread_dist_, self.percentile))
        self.n_pairs_      = len(models_sv) * (len(models_sv) - 1) // 2
        self.feature_thresholds_ = {
            f: float(np.percentile(spreads_per_feat[f], self.percentile))
            for f in feature_names
        }
        return self

    def report(self) -> str:
        """Relatório textual da calibração."""
        if self.threshold_ is None:
            return "Calibrador não foi ajustado. Execute .fit() primeiro."

        lines = [
            f"ShapeThresholdCalibrator",
            f"  percentil={self.percentile}  n_modelos={self.n_models}  "
            f"n_pares={self.n_pairs_}",
            f"",
            f"  Threshold global: {self.threshold_:.4f}  "
            f"(magic number original: 0.30)",
            f"  Threshold por feature:",
        ]
        sorted_feats = sorted(self.feature_thresholds_.items(),
                              key=lambda x: -x[1])
        for f, t in sorted_feats:
            bar  = "█" * int(t * 40)
            flag = ""
            if t > 0.5:
                flag = " ← variabilidade alta, threshold folgado"
            elif t < 0.2:
                flag = " ← feature estável, threshold restrito"
            lines.append(f"    {f:<28} {t:.4f} {bar}{flag}")

        lines += [
            f"",
            f"  Interpretação:",
            f"    Threshold > 0.30: domínio tem mais variabilidade natural",
            f"    → usar 0.30 geraria falsos positivos",
            f"    Threshold < 0.30: domínio é mais estável",
            f"    → usar 0.30 perderia divergências reais",
        ]
        return "\n".join(lines)

    def threshold_for(self, feature: str) -> float:
        """Retorna o threshold calibrado para uma feature específica.
        Se a feature não foi calibrada, retorna o threshold global.
        """
        return self.feature_thresholds_.get(feature, self.threshold_)

    def summary(self) -> dict:
        """Retorna um dict resumido para uso programático."""
        return {
            'threshold_global':    self.threshold_,
            'feature_thresholds':  self.feature_thresholds_,
            'percentile':          self.percentile,
            'n_pairs':             self.n_pairs_,
            'spread_mean':         float(self.spread_dist_.mean()),
            'spread_std':          float(self.spread_dist_.std()),
        }
