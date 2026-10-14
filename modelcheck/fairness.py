"""
fairness.py — alerta quando variáveis sensíveis pesam demais na decisão.

⚠️  ESCOPO E LIMITES — leia antes de usar

Este módulo mede UMA coisa: a fração da atribuição total que uma variável
declarada como sensível recebe. Isso NÃO é uma avaliação de justiça do
modelo, e não substitui uma.

O que ele NÃO faz:

  - não detecta discriminação por proxy (CEP correlacionado com raça,
    nome correlacionado com gênero). Uma variável sensível com peso zero
    e um proxy com peso alto passa sem alerta.
  - não mede disparidade de resultado entre grupos (paridade demográfica,
    igualdade de oportunidade, calibração por grupo). Para isso use uma
    biblioteca dedicada.
  - não afirma que peso abaixo do limiar significa modelo justo.

O limiar padrão de 0.2 é arbitrário: não há base empírica para ele, e o
valor apropriado depende do domínio e do enquadramento legal. Trate como
ponto de partida para uma conversa, não como critério de aprovação.

Este módulo NÃO participa do veredito de `Guard.audit()` — por decisão
explícita, documentada em LIMITATIONS.md. Integrar ao veredito exigiria
validação com ground truth que ainda não temos, e em fairness um falso
negativo tem consequência que um defeito técnico não tem.
"""
import numpy as np
import pandas as pd


class FairnessChecker:
    """
    Parameters
    ----------
    sensitive_features : nomes das variáveis a monitorar
    max_weight : fração máxima da atribuição total (padrão 0.2, arbitrário)
    """

    def __init__(self, sensitive_features: list, max_weight: float = 0.2):
        if isinstance(sensitive_features, str):
            sensitive_features = [sensitive_features]
        if not sensitive_features:
            raise ValueError(
                "FairnessChecker: `sensitive_features` está vazio. Informe "
                "as variáveis a monitorar, ex: ['genero', 'idade'].")
        if not 0 < max_weight <= 1:
            raise ValueError(
                f"FairnessChecker: max_weight deve estar em (0, 1], "
                f"recebido {max_weight}.")
        self.sensitive_features = list(sensitive_features)
        self.max_weight = float(max_weight)

    @staticmethod
    def _as_array(sv):
        """
        BUG 15, de novo (v1.64): este módulo usava `sv.values` direto e
        quebrava com numpy.ndarray — que é o que shap.TreeExplainer
        devolve. Mesmo defeito já corrigido em shape.py; aqui passou
        despercebido porque o módulo não tinha teste.
        """
        from .frechet import matriz_shap
        return matriz_shap(sv)   # BUG 95: aceita a lista do shap antigo

    def check(self, shap_values, X) -> dict:
        """
        Returns
        -------
        dict com `violations`, `weights`, `missing` e `checked`.

        BUG 29 (v1.64): a versão anterior devolvia lista vazia quando a
        variável sensível NÃO EXISTIA em X — indistinguível de "nenhuma
        violação". Num módulo de fairness isso é o pior modo de falha
        possível: um erro de digitação no nome da coluna produzia um laudo
        de aprovação silencioso.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"FairnessChecker.check: X deve ser um DataFrame com nomes "
                f"de coluna, recebido {type(X).__name__}.")
        sv = self._as_array(shap_values)
        if sv.shape[1] != X.shape[1]:
            raise ValueError(
                f"FairnessChecker.check: atribuições têm {sv.shape[1]} "
                f"colunas e X tem {X.shape[1]}. Devem corresponder.")

        ausentes = [f for f in self.sensitive_features if f not in X.columns]
        if ausentes:
            raise ValueError(
                f"FairnessChecker.check: variáveis sensíveis ausentes em X: "
                f"{ausentes}. Colunas disponíveis: {list(X.columns)[:8]}. "
                f"Um nome errado produziria um laudo de aprovação sem que "
                f"nada tenha sido verificado.")

        imp = np.abs(sv).mean(axis=0)
        total = float(imp.sum())
        if total == 0:
            return {"violations": [], "weights": {}, "checked": [],
                    "status": "NAO_AVALIAVEL",
                    "reason": ("todas as atribuições são zero — o modelo pode "
                               "ser constante. Nada foi verificado.")}

        pesos = dict(zip(X.columns, imp / total))
        viol = [{"feature": f, "weight": round(float(pesos[f]), 4),
                 "max_allowed": self.max_weight}
                for f in self.sensitive_features
                if pesos[f] > self.max_weight]
        return {
            "violations": viol,
            "weights": {f: round(float(pesos[f]), 4)
                        for f in self.sensitive_features},
            "checked": list(self.sensitive_features),
            "status": "VIOLATION" if viol else "OK",
            "note": ("mede apenas o peso direto da variável declarada; não "
                     "detecta proxy nem disparidade de resultado entre "
                     "grupos."),
        }
