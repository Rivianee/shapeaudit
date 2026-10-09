"""
zest.py — ArchitectureAgnosticComparator (inspirado no Zest)

Problema: comparar XGBoost vs MLP não é possível via pesos
nem via SHAP (arquiteturas diferentes).

Solução: compara os modelos pelas suas EXPLICAÇÕES LOCAIS.
Para cada amostra x, aproxima cada modelo com uma regressão
linear local (LIME-style). O vetor de coeficientes é a
"assinatura local" do modelo. Compara as distribuições
dessas assinaturas entre v1 e v2.

Uso:
    from modelcheck.zest import ArchitectureAgnosticComparator
    cmp = ArchitectureAgnosticComparator()
    result = cmp.compare(model_v1, model_v2, X_eval, features)
    print(cmp.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _local_explanation(model, x: np.ndarray,
                        X_ref: np.ndarray,
                        feature_names: List[str],
                        n_neighbors: int = 50,
                        sigma: float = 0.1) -> Optional[np.ndarray]:
    """
    Explicação linear local para o ponto x (LIME-style).
    Retorna vetor de coeficientes da regressão local.
    """
    n_feat = x.shape[0]
    # Perturbações gaussianas ao redor de x
    rng = np.random.RandomState(int(np.abs(x).sum() * 1000) % 2**31)
    noise = rng.normal(0, sigma * np.std(X_ref, axis=0) + 1e-8,
                       size=(n_neighbors, n_feat))
    X_local = x + noise

    # Peso por distância (kernel gaussiano)
    dists   = np.sqrt(((X_local - x)**2).sum(axis=1))
    weights = np.exp(-dists**2 / (2 * sigma**2))

    # Predições do modelo nos pontos perturbados
    try:
        df_local = pd.DataFrame(X_local, columns=feature_names)
        y_local  = model.predict(df_local)
    except Exception:
        y_local = model.predict(X_local)

    # Regressão linear ponderada
    sc = StandardScaler()
    X_sc = sc.fit_transform(X_local)
    lr = Ridge(alpha=1.0)
    lr.fit(X_sc, y_local, sample_weight=weights)
    return lr.coef_


class ArchitectureAgnosticComparator:
    """
    Compara dois modelos de qualquer arquitetura via explicações locais.

    Funciona para RF vs MLP, XGBoost vs SVM, etc.
    Não depende de SHAP nem de acesso aos pesos internos.

    Métricas:
      coef_distance  : distância L2 média entre coeficientes locais
      agreement_rate : % de amostras onde modelos concordam no sinal
      top_disagreement: features com maior discordância local
    """

    def __init__(self, n_sample: int = 80,
                 n_neighbors: int = 40,
                 sigma: float = 0.15):
        self.n_sample    = n_sample
        self.n_neighbors = n_neighbors
        self.sigma       = sigma

    def compare(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:

        if feature_names is None:
            feature_names = list(X_eval.columns)

        X_arr = X_eval[feature_names].values
        idx   = np.random.choice(len(X_arr),
                                 min(self.n_sample, len(X_arr)),
                                 replace=False)
        X_sub = X_arr[idx]

        coefs1, coefs2 = [], []
        for x in X_sub:
            c1 = _local_explanation(model_v1, x, X_arr,
                                    feature_names, self.n_neighbors,
                                    self.sigma)
            c2 = _local_explanation(model_v2, x, X_arr,
                                    feature_names, self.n_neighbors,
                                    self.sigma)
            if c1 is not None and c2 is not None:
                coefs1.append(c1); coefs2.append(c2)

        if not coefs1:
            return {'error': 'Sem explicações locais suficientes'}

        C1 = np.array(coefs1); C2 = np.array(coefs2)

        # Distância L2 média entre vetores de coeficientes
        dists = np.sqrt(((C1 - C2)**2).sum(axis=1))
        coef_dist = float(dists.mean())
        coef_dist_std = float(dists.std())

        # Taxa de concordância no sinal por feature
        sign_agree = (np.sign(C1) == np.sign(C2)).mean(axis=0)
        agreement_rate = float(sign_agree.mean())

        # Features com maior discordância
        disagreement = 1 - sign_agree
        top_idx = np.argsort(disagreement)[::-1][:4]
        top_disagree = [(feature_names[i], round(float(disagreement[i]), 3))
                        for i in top_idx if i < len(feature_names)]

        # Score: 0=modelos completamente diferentes, 100=idênticos
        # Normalizado pela dimensionalidade
        max_dist = np.sqrt(len(feature_names)) * 2
        arch_score = max(0.0, 100 * (1 - coef_dist / max_dist))

        return {
            'coef_distance':    round(coef_dist, 4),
            'coef_dist_std':    round(coef_dist_std, 4),
            'agreement_rate':   round(agreement_rate, 3),
            'top_disagreement': top_disagree,
            'arch_score':       round(arch_score, 1),
            'n_explained':      len(coefs1),
        }

    def report(self, result: dict) -> str:
        if 'error' in result:
            return f"ArchitectureAgnosticComparator: {result['error']}"
        lines = [
            "ArchitectureAgnosticComparator (Zest-style):",
            f"  Architecture Score: {result['arch_score']}/100",
            f"  (100=comportamento idêntico, 0=completamente diferente)",
            f"  Distância L2 média:  {result['coef_distance']:.4f} ± {result['coef_dist_std']:.4f}",
            f"  Concordância de sinal: {result['agreement_rate']:.1%}",
            f"  Amostras explicadas: {result['n_explained']}",
        ]
        if result['top_disagreement']:
            lines.append("  Features com maior discordância local:")
            for f, d in result['top_disagreement']:
                bar = '█' * int(d * 20)
                lines.append(f"    {f:<28} {bar} {d:.3f}")
        return "\n".join(lines)
