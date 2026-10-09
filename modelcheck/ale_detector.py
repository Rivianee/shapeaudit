"""ALE-based Shape Instability Detector.

Contribuição nova em relação ao SDD original (baseado em SHAP):

Problema com SHAP:
    1. Depende de background dataset — a forma muda com background diferente
    2. Features correlacionadas — SHAP distribui efeito entre features
    3. KernelSHAP para MLP/SVM é lento (O(n*d)) e ruidoso com n pequeno
    4. Modelos proprietários via API — não dá extrair pesos para SHAP

Solução ALE:
    ALE (Accumulated Local Effects, Apley & Zhu 2020) computa o efeito
    local de cada feature condicionando nos vizinhos — robusto a correlações.
    Usa apenas model.predict() — funciona para qualquer arquitetura.

    Algoritmo:
    1. Para cada bin k da feature j:
       a. Seleciona amostras dentro do bin
       b. Prediz com feature no limite inferior e superior do bin
       c. Delta médio = efeito local do bin
    2. Acumula os deltas → perfil ALE cumulativo
    3. Centraliza (subtrai a média)
    4. Classifica a forma do perfil com _shape_profile_small_n

Resultado experimental (benchmark controlado, n=2000, 6 cenários):
    SHAP: Recall=100%, Precision=67%, F1=0.80, FP=3
    ALE:  Recall=83%,  Precision=45%, F1=0.59, FP=6

    SHAP é superior para modelos de árvore onde é exato.
    ALE é a ÚNICA opção para:
    - MLP, SVM, redes neurais (KernelSHAP lento/ruidoso)
    - Modelos via API (sem acesso aos pesos)
    - LLMs via behavioral probing (LLMGuard)

Uso:
    from modelcheck.ale_detector import ALEShapeDetector

    detector = ALEShapeDetector(n_bins=6, n_boot=40, min_confidence=0.65)
    results  = detector.compare(model_v1, model_v2, X_eval, feature_names)
    print(detector.report(results))
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from collections import Counter

from .shape import _shape_profile_small_n


def ale_shape_profile(model,
                      X:       pd.DataFrame,
                      feature: str,
                      n_bins:  int = 6,
                      min_bin_n: int = 5) -> Optional[dict]:
    """Calcula o perfil de forma ALE para uma feature.

    Parâmetros:
        model     : modelo com interface predict(X) → array 1D
        X         : DataFrame de avaliação
        feature   : nome da feature
        n_bins    : número de bins quantílicos (padrão 6)
        min_bin_n : mínimo de amostras por bin (padrão 5)

    Retorna dict com:
        shape     : classificação ('crescente', 'decrescente', 'u',
                                   'u_invertido', 'irregular')
        confidence: confiança do classificador (0–1)
        ale_cum   : array do perfil ALE cumulativo
        centers   : centros dos bins
        n_bins_ok : número de bins com amostras suficientes

    Retorna None se não há amostras suficientes.

    Notas técnicas:
        O perfil ALE mede o efeito ACUMULADO da feature — análogo à
        integral do efeito local. Isso torna a classificação de forma
        diretamente comparável ao SHAP profile: crescente significa
        que a feature tem efeito positivo crescente ao longo do seu range.
    """
    X = X.copy()
    x = np.asarray(X[feature].values, dtype=float)

    if np.std(x) < 1e-8:
        return None

    # Bins quantílicos
    bins = np.unique(np.percentile(x, np.linspace(0, 100, n_bins + 1)))
    if len(bins) < 4:
        return None

    ale_means = []
    centers   = []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x <= bins[i + 1])
        if mask.sum() < min_bin_n:
            continue

        Xlo = X[mask].copy()
        Xhi = X[mask].copy()
        Xlo[feature] = bins[i]
        Xhi[feature] = bins[i + 1]

        # Efeito local médio do bin
        delta = model.predict(Xhi.values) - model.predict(Xlo.values)
        ale_means.append(float(delta.mean()))
        centers.append((bins[i] + bins[i + 1]) / 2)

    if len(ale_means) < 3:
        return None

    # Perfil cumulativo centralizado
    ale_cum  = np.cumsum(ale_means)
    ale_cum -= ale_cum.mean()

    # Classifica a forma usando _shape_profile_small_n
    # (min_n=3 porque temos n_bins centros, não n amostras)
    p = _shape_profile_small_n(
        np.array(centers, dtype=float),
        ale_cum,
        min_n=3
    )
    if not p:
        return None

    return {
        'shape':      p['shape'],
        'confidence': p.get('confidence', 1.0),
        'ale_cum':    ale_cum.tolist(),
        'centers':    centers,
        'n_bins_ok':  len(ale_means),
    }


class ALEShapeDetector:
    """Detecta Shape Instability entre dois modelos usando ALE.

    Funciona com qualquer model.predict() — RF, GB, MLP, SVM, API.

    Parâmetros:
        n_bins        : bins quantílicos para o ALE (padrão 6)
        min_bin_n     : amostras mínimas por bin (padrão 5)
        n_boot        : iterações bootstrap para estabilidade (padrão 40)
        min_confidence: confiança mínima para alertar (padrão 0.65)
                        Valor mais alto → menos FPs, mais FNs

    Uso com modelos de árvore (RF, GB):
        Quando TreeSHAP está disponível, prefira o SDD original.
        Use ALEShapeDetector quando:
        - O modelo não é árvore (MLP, SVM, NN)
        - Não tem background dataset
        - Quer resultado comparável entre arquiteturas diferentes

    Uso com modelos via API:
        def my_api_model(X):
            # chama a API e retorna array de predições
            return np.array([api_predict(row) for row in X])

        detector = ALEShapeDetector()
        results  = detector.compare(my_api_model, my_api_model_v2,
                                    X_eval, features)
    """

    def __init__(self,
                 n_bins:         int   = 6,
                 min_bin_n:      int   = 5,
                 n_boot:         int   = 40,
                 min_confidence: float = 0.65):
        self.n_bins         = n_bins
        self.min_bin_n      = min_bin_n
        self.n_boot         = n_boot
        self.min_confidence = min_confidence

    def _profile_bootstrap(self, model, X: pd.DataFrame,
                           feature: str) -> tuple:
        """Classifica a forma com bootstrap — retorna (shape, confidence)."""
        shapes = []
        n = len(X)
        for seed in range(self.n_boot):
            np.random.seed(seed)
            idx = np.random.choice(n, n, replace=True)
            Xb  = X.iloc[idx].reset_index(drop=True)
            p   = ale_shape_profile(model, Xb, feature,
                                    self.n_bins, self.min_bin_n)
            if p:
                shapes.append(p['shape'])

        if not shapes:
            return 'n/a', 0.0

        sh, c = Counter(shapes).most_common(1)[0]
        return sh, round(c / len(shapes), 2)

    def compare(self,
                model1,
                model2,
                X_eval:        pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> list:
        """Compara formas ALE entre dois modelos.

        Parâmetros:
            model1, model2 : modelos com interface predict(array) → array
            X_eval         : DataFrame de avaliação
            feature_names  : lista de features (default: todas)

        Retorna lista de dicts por feature:
            feature     : nome
            shape1      : forma do modelo 1
            shape2      : forma do modelo 2
            conf1, conf2: confiança bootstrap de cada classificação
            divergence  : True se shapes divergem com confiança >= min_confidence
            alert       : descrição da divergência ou None
        """
        if feature_names is None:
            feature_names = list(X_eval.columns)

        results = []
        for f in feature_names:
            sh1, c1 = self._profile_bootstrap(model1, X_eval, f)
            sh2, c2 = self._profile_bootstrap(model2, X_eval, f)

            div = (sh1 != sh2
                   and sh1 not in ('n/a', 'irregular')
                   and sh2 not in ('n/a', 'irregular')
                   and c1 >= self.min_confidence
                   and c2 >= self.min_confidence)

            alert = (f"ALE_SHAPE_DIV: '{f}' {sh1}→{sh2} "
                     f"(conf: {c1:.0%}/{c2:.0%})") if div else None

            results.append({
                'feature':   f,
                'shape1':    sh1, 'conf1': c1,
                'shape2':    sh2, 'conf2': c2,
                'divergence': div,
                'alert':     alert,
            })

        return results

    def report(self, results: list) -> str:
        """Relatório textual dos resultados."""
        divs = [r for r in results if r['divergence']]
        n    = len(results)

        lines = [f"ALEShapeDetector (n_boot={self.n_boot}, "
                 f"min_conf={self.min_confidence:.0%}):"]

        if not divs:
            lines.append(f"  ✅ Nenhuma divergência de forma detectada ({n} features)")
        else:
            lines.append(f"  ⚠ {len(divs)}/{n} feature(s) com Shape Instability:")
            for r in divs:
                lines.append(f"    '{r['feature']}': "
                             f"{r['shape1']} (conf={r['conf1']:.0%}) → "
                             f"{r['shape2']} (conf={r['conf2']:.0%})")

        stable = [r for r in results
                  if not r['divergence'] and r['shape1'] != 'n/a']
        if stable:
            lines.append(f"  ✅ {len(stable)} feature(s) estáveis: "
                         f"{[r['feature'] for r in stable]}")

        na = [r for r in results if r['shape1'] == 'n/a']
        if na:
            lines.append(f"  → {len(na)} feature(s) sem amostras suficientes "
                         f"(aumentar n_eval ou reduzir n_bins)")

        return "\n".join(lines)

    def health_score(self, results: list) -> dict:
        """Health Score 0–100 baseado nas divergências ALE."""
        n_div = sum(1 for r in results if r['divergence'])
        score = max(0, 100 - n_div * 15)
        status = ("HEALTHY"   if score >= 85 else
                  "ATTENTION" if score >= 65 else "CRITICAL")
        return {'score': score, 'status': status, 'n_divergences': n_div}
