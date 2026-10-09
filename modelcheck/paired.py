"""
paired.py — PairedDivergenceAnalyzer

Responde: ONDE no espaço de dados os dois modelos divergem?
E QUAIS REGRAS descrevem essas regiões?

Ao invés de dizer "HORA divergiu" (SDD),
diz "os modelos divergem para amostras onde HORA > 14 e EAR < 30%"

Algoritmo:
1. Calcula divergência por amostra: |pred_v1(x) - pred_v2(x)|
2. Binariza: amostra é "divergente" se divergência > threshold
3. Treina decision tree para prever quais amostras são divergentes
4. Extrai as regras das folhas com maior concentração de divergências
5. Retorna regras em linguagem natural + estatísticas

Uso:
    from modelcheck.paired import PairedDivergenceAnalyzer
    pa = PairedDivergenceAnalyzer()
    result = pa.analyze(model_v1, model_v2, X_eval, feature_names)
    print(pa.report(result))
"""
import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.tree import DecisionTreeClassifier, export_text


def _extract_rules(tree, feature_names: List[str],
                   X: pd.DataFrame,
                   y_div: np.ndarray,
                   max_rules: int = 5) -> List[dict]:
    """Extrai regras das folhas mais divergentes da decision tree."""
    # v1.74: a árvore é treinada com `.values` (sem nomes de coluna). Passar
    # um DataFrame aqui gera UserWarning do sklearn em TODA auditoria,
    # poluindo a saída de quem usa a biblioteca. Convertemos para array.
    _X = X[feature_names].to_numpy() if hasattr(X, 'to_numpy') else X
    node_indicator = tree.decision_path(_X)
    leaf_ids = tree.apply(_X)
    rules = []

    # Para cada folha única
    for leaf_id in np.unique(leaf_ids):
        mask_leaf = (leaf_ids == leaf_id)
        n_div   = y_div[mask_leaf].sum()
        n_total = mask_leaf.sum()
        if n_total < 3 or n_div / n_total < 0.5:
            continue

        # Reconstrói o caminho da raiz até essa folha
        node_path = node_indicator[:, :].toarray()
        # Pega as amostras nessa folha
        sample_idx = np.where(mask_leaf)[0][0]
        path_nodes = np.where(node_path[sample_idx])[0]

        conditions = []
        for node_id in path_nodes[:-1]:  # exclui a folha
            feat_idx = tree.tree_.feature[node_id]
            thresh   = tree.tree_.threshold[node_id]
            feat_nm  = feature_names[feat_idx]
            # Descobre se foi pelo ramo esquerdo (<= thresh) ou direito (>)
            left_child  = tree.tree_.children_left[node_id]
            right_child = tree.tree_.children_right[node_id]
            next_node = path_nodes[list(path_nodes).index(node_id) + 1]
            if next_node == left_child:
                conditions.append(f"{feat_nm} ≤ {thresh:.2f}")
            else:
                conditions.append(f"{feat_nm} > {thresh:.2f}")

        if conditions:
            rules.append({
                'conditions':    conditions,
                'rule_text':     ' AND '.join(conditions),
                'n_total':       int(n_total),
                'n_divergent':   int(n_div),
                'divergence_rate': round(float(n_div / n_total), 3),
                'mean_div_magnitude': float(
                    np.abs(tree.tree_.value[leaf_id]).mean()),
            })

    # Ordena por taxa de divergência × cobertura
    rules.sort(key=lambda r: r['divergence_rate'] * r['n_total'],
               reverse=True)
    return rules[:max_rules]


class PairedDivergenceAnalyzer:
    """
    Identifica ONDE e POR QUE dois modelos divergem no espaço de dados.

    Complementa o SDD:
      - SDD: "a feature HORA mudou de forma crescente para U"
      - PDA: "os modelos divergem onde HORA > 18 AND PERIODO = 2"

    Casos de uso:
      - Diagnóstico de regiões problemáticas antes do deploy
      - Comunicação para gestores em linguagem de negócio
      - Identificação de subgrupos de risco no CCEE, crédito, LiDAR
    """

    def __init__(self,
                 div_threshold:  Optional[float] = None,
                 div_percentile: float = 75.0,
                 max_depth:      int   = 3,
                 min_samples:    int   = 10,
                 max_rules:      int   = 5):
        self.div_threshold  = div_threshold
        self.div_percentile = div_percentile
        self.max_depth      = max_depth
        self.min_samples    = min_samples
        self.max_rules      = max_rules

    def analyze(self, model_v1, model_v2,
                X_eval: pd.DataFrame,
                feature_names: Optional[List[str]] = None) -> dict:
        """
        Analisa onde os modelos divergem.

        Retorna dict com:
            divergence_per_sample : array de divergência por amostra
            threshold_used        : threshold de binarização
            pct_divergent         : % de amostras divergentes
            rules                 : lista de regras (conditions, stats)
            top_features          : features mais discriminantes
            mean_divergence       : divergência média global
            max_divergence        : divergência máxima
        """
        if feature_names is None:
            feature_names = list(X_eval.columns)

        # ── 1. Divergência por amostra ────────────────────────────
        pred1 = np.array(model_v1.predict(X_eval))
        pred2 = np.array(model_v2.predict(X_eval))
        div   = np.abs(pred1 - pred2)

        # Threshold adaptativo
        tau = (self.div_threshold if self.div_threshold is not None
               else np.percentile(div, self.div_percentile))
        y_div = (div > tau).astype(int)
        pct_div = float(y_div.mean())

        # ── 2. Decision Tree sobre divergências ───────────────────
        X_arr = X_eval[feature_names].values
        tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples,
            class_weight='balanced',
            random_state=42
        )
        # v1.74: treinar com `.values` e depois prever com DataFrame gera
        # UserWarning do sklearn em toda auditoria, poluindo a saída de
        # quem usa a biblioteca. Mantemos array nos dois lados.
        tree.fit(X_arr, y_div)

        # ── 3. Features mais discriminantes ──────────────────────
        importances = tree.feature_importances_
        top_idx = np.argsort(importances)[::-1][:4]
        top_features = [(feature_names[i], round(float(importances[i]), 3))
                        for i in top_idx if importances[i] > 0.01]

        # ── 4. Extrai regras ──────────────────────────────────────
        rules = _extract_rules(tree, feature_names, X_eval,
                               y_div, self.max_rules)

        # Enriquece regras com divergência média real nas regiões
        for rule in rules:
            # Aplica as condições manualmente para pegar a divergência real
            mask = np.ones(len(X_eval), dtype=bool)
            for cond in rule['conditions']:
                # Parseia "feature ≤ valor" ou "feature > valor"
                for sep, op in [(' ≤ ', '<='), (' > ', '>')]:
                    if sep in cond:
                        feat, val = cond.split(sep)
                        feat = feat.strip(); val = float(val.strip())
                        if op == '<=':
                            mask &= X_eval[feat].values <= val
                        else:
                            mask &= X_eval[feat].values > val
                        break
            if mask.sum() > 0:
                rule['mean_div_real']  = round(float(div[mask].mean()), 4)
                rule['n_real']         = int(mask.sum())
                rule['pct_divergent_real'] = round(float(y_div[mask].mean()), 3)
            else:
                rule['mean_div_real'] = 0.0
                rule['n_real'] = 0
                rule['pct_divergent_real'] = 0.0

        return {
            'divergence_per_sample': div,
            'threshold_used':        round(float(tau), 4),
            'pct_divergent':         round(pct_div, 3),
            'rules':                 rules,
            'top_features':          top_features,
            'mean_divergence':       round(float(div.mean()), 4),
            'max_divergence':        round(float(div.max()), 4),
            'n_eval':                len(X_eval),
        }

    def report(self, result: dict,
               unit: str = '') -> str:
        """Relatório em linguagem de negócio."""
        lines = [
            "PairedDivergenceAnalyzer:",
            f"  Amostras avaliadas:  {result['n_eval']}",
            f"  Divergência média:   {result['mean_divergence']:.4f}{unit}",
            f"  Divergência máxima:  {result['max_divergence']:.4f}{unit}",
            f"  Threshold:           {result['threshold_used']:.4f}{unit}",
            f"  % amostras divergentes: {result['pct_divergent']:.1%}",
        ]

        if result['top_features']:
            lines.append(f"\n  Features discriminantes:")
            for f, imp in result['top_features']:
                bar = '█' * int(imp * 20)
                lines.append(f"    {f:<28} {bar} {imp:.3f}")

        if result['rules']:
            lines.append(f"\n  Regiões de maior divergência:")
            for i, r in enumerate(result['rules'], 1):
                lines.append(
                    f"\n  [{i}] {r['rule_text']}"
                    f"\n      Cobertura: {r['n_real']} amostras  "
                    f"({r['pct_divergent_real']:.0%} divergentes)"
                    f"  div_média={r['mean_div_real']:.4f}{unit}")
        else:
            lines.append("\n  ✅ Nenhuma região de divergência sistemática.")

        return "\n".join(lines)
