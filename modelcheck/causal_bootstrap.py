"""ShapeCausalExplainer — identifica causas de mudança de forma com bootstrap.

Problema original (Limitação 5 do paper):
    O leave-group-out pontual identifica qual grupo de dados causa
    mudança de forma no modelo, mas sem quantificar a confiança.

    Exemplo real — LiDAR Guangxi:
        Análise pontual: Chinese fir causa 7/13 features
        Com bootstrap (n=15): Chinese fir causa 3/13 com 67-73% confiança
                               Eucalyptus causa 6/13 com 53-93% confiança
        O resultado pontual estava errado — Eucalyptus é a causa dominante

Solução:
    Para cada grupo, roda leave-group-out K vezes com subsamples bootstrap.
    Conta em quantas rodadas o grupo causou divergência de forma.
    Define confiança = proporção de rodadas onde o grupo causou divergência.
    Calcula intervalo de confiança de Wilson (95%) para cada estimativa.

    IC estreito com confiança alta → causa identificada de forma robusta
    IC largo ou confiança baixa → resultado instável, precisa mais dados

Uso:
    from modelcheck.causal import ShapeCausalExplainer

    explainer = ShapeCausalExplainer(n_bootstrap=20)
    explainer.fit(X_with_group_col, y, group_col='Species_name',
                  feature_names=['hp95', 'Hmean', ...])

    print(explainer.report(min_confidence=0.5))
    causes = explainer.top_causes(min_confidence=0.5)

Resultados experimentais (LiDAR Guangxi, n=1003):
    hp95     → Eucalyptus  93%  IC=[0.70, 0.99]  ← robusto
    LADmean  → Chinese fir 67%  IC=[0.42, 0.85]  ← moderado
    Hmean    → sem causa clara (< 40% para todas as espécies)
    dp50     → sem causa clara

    Interpretação ecológica confirmada:
    Eucalyptus domina estrutura de dossel (hp95, Hstd, CC)
    Chinese fir domina densidade foliar (LADmean, LADstd, LADcv)
"""
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from typing import List, Optional, Union

from .shape import _shape_profile


def _get_shapes(X: pd.DataFrame,
                y: pd.Series,
                feature_names: List[str],
                seed: int = 0,
                n_shap: int = 80) -> dict:
    """Treina RF e retorna dict {feature: shape}."""
    rf = RandomForestRegressor(
        n_estimators=60, max_depth=6,
        random_state=seed, n_jobs=1).fit(X, y)
    n = min(n_shap, len(X))
    sample = X.sample(n, random_state=seed)
    sv = shap.TreeExplainer(rf).shap_values(sample)
    shapes = {}
    for i, f in enumerate(feature_names):
        p = _shape_profile(sample[f].values, sv[:, i])
        shapes[f] = p['shape'] if p else 'n/a'
    return shapes


def _count_divs(sh_ref: dict, sh_test: dict, feature_names: List[str]) -> List[str]:
    """Lista de features com shapes diferentes."""
    return [f for f in feature_names
            if sh_ref.get(f, 'n/a') != sh_test.get(f, 'n/a')
            and sh_ref.get(f, 'n/a') != 'n/a'
            and sh_test.get(f, 'n/a') != 'n/a']


def _wilson_ci(p: float, n: int, z: float = 1.96):
    """
    Intervalo de confiança de Wilson para proporção.

    n=0 devolve [0, 1] (nenhuma informação) em vez de ZeroDivisionError.
    """
    if n <= 0:
        return 0.0, 1.0
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


class ShapeCausalExplainer:
    """Identifica causas de mudança de forma via leave-group-out com bootstrap.

    Parâmetros:
        n_bootstrap     : iterações bootstrap (padrão 20)
        sample_frac     : fração do dataset por subsample (padrão 0.8)
        min_group_size  : tamanho mínimo do grupo para ser considerado (30)
        n_shap          : amostras para cálculo SHAP (padrão 80)
        random_state    : seed para reprodutibilidade

    Atributos após fit():
        results_        : dict aninhado {feature: {grupo: stats}}
        feature_names_  : lista de features
        groups_         : lista de grupos
    """

    def __init__(self,
                 n_bootstrap:    int   = 20,
                 sample_frac:    float = 0.8,
                 min_group_size: int   = 30,
                 n_shap:         int   = 80,
                 random_state:   int   = 42,
                 min_rel_importance: float = 0.02):
        self.min_rel_importance = min_rel_importance
        self.n_bootstrap    = n_bootstrap
        self.sample_frac    = sample_frac
        self.min_group_size = min_group_size
        self.n_shap         = n_shap
        self.random_state   = random_state

        self.results_       = {}
        self.feature_names_ = []
        self.groups_        = []
        self.rel_importance_    = {}
        self.excluded_features_ = []

    def _relevant_features(self, X, y, feature_names):
        """
        Features com importância relativa >= min_rel_importance.

        Guarda contra causalidade espúria: o perfil SHAP de uma feature
        irrelevante é ruído, então sua forma vira ao remover qualquer
        subgrupo. Medido: feature com 0.08% da importância apontada como
        causa com confiança 0.75-0.92 em controle sem efeito nenhum.
        """
        try:
            import shap
            from sklearn.ensemble import RandomForestRegressor
            Xn = X[feature_names]
            mod = RandomForestRegressor(n_estimators=60, max_depth=6,
                                        random_state=self.random_state,
                                        n_jobs=1).fit(Xn, y)
            Xs = Xn.sample(min(self.n_shap * 2, len(Xn)),
                           random_state=self.random_state)
            sv = shap.TreeExplainer(mod).shap_values(Xs)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]
            sv = np.asarray(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
            imp = np.abs(sv).mean(axis=0)
            rel = imp / max(imp.sum(), 1e-12)
            self.rel_importance_ = {f: round(float(v), 5)
                                    for f, v in zip(feature_names, rel)}
            keep = [f for f, r in zip(feature_names, rel)
                    if r >= self.min_rel_importance]
            self.excluded_features_ = [f for f in feature_names
                                       if f not in keep]
            return keep
        except Exception:
            self.rel_importance_ = {}
            self.excluded_features_ = []
            return list(feature_names)

    def fit(self,
            X:             pd.DataFrame,
            y:             pd.Series,
            group_col:     str,
            feature_names: Optional[List[str]] = None,
            verbose:       bool = True) -> 'ShapeCausalExplainer':
        """Aprende causas via leave-group-out com bootstrap.

        Parâmetros:
            X             : DataFrame com features + coluna de grupo
            y             : target
            group_col     : coluna que define os grupos (ex: 'Species_name')
            feature_names : lista de features (default: todas exceto group_col)
            verbose       : imprime progresso

        Retorna self.
        """
        np.random.seed(self.random_state)
        if feature_names is None:
            feature_names = [c for c in X.columns if c != group_col]
        # FILTRO DE IMPORTÂNCIA (v1.36): remove features irrelevantes antes
        # da análise causal. Sem isso, o perfil SHAP de ruído é instável e
        # gera causalidade espúria com confiança alta (medido: 0.75-0.92 em
        # controle sem nenhum efeito).
        if self.min_rel_importance > 0:
            _keep = self._relevant_features(X, y, feature_names)
            if _keep:
                feature_names = _keep

        if feature_names is None:
            feature_names = [c for c in X.columns if c != group_col]

        groups   = list(X[group_col].unique())
        X_feats  = X[feature_names]
        g_series = X[group_col]

        if verbose:
            print(f"ShapeCausalExplainer: {self.n_bootstrap} bootstrap × "
                  f"{len(groups)} grupos × {len(feature_names)} features")

        # causas[feature][grupo] = lista de bool (causou divergência?)
        causas = {f: {g: [] for g in groups} for f in feature_names}

        for k in range(self.n_bootstrap):
            # Subsample com reposição
            n_sample = int(len(X) * self.sample_frac)
            idx      = np.random.choice(len(X), n_sample, replace=True)
            X_b      = X_feats.iloc[idx].reset_index(drop=True)
            y_b      = y.iloc[idx].reset_index(drop=True)
            g_b      = g_series.iloc[idx].reset_index(drop=True)

            # Modelo de referência
            sh_ref = _get_shapes(X_b, y_b, feature_names,
                                 seed=k, n_shap=self.n_shap)

            # Leave-group-out por grupo
            for g in groups:
                mask   = g_b != g
                X_sem  = X_b[mask].reset_index(drop=True)
                y_sem  = y_b[mask].reset_index(drop=True)

                if len(X_sem) < self.min_group_size:
                    for f in feature_names:
                        causas[f][g].append(None)
                    continue

                sh_sem = _get_shapes(X_sem, y_sem, feature_names,
                                     seed=k + 1000, n_shap=self.n_shap)
                divs   = _count_divs(sh_ref, sh_sem, feature_names)

                for f in feature_names:
                    causas[f][g].append(f in divs)

        # Calcula estatísticas por feature × grupo
        self.results_ = {}
        for f in feature_names:
            self.results_[f] = {}
            for g in groups:
                votes = [v for v in causas[f][g] if v is not None]
                if not votes:
                    self.results_[f][g] = {
                        'confidence': None,
                        'n_votes':    0,
                        'ci_lower':   None,
                        'ci_upper':   None,
                    }
                    continue
                conf      = float(np.mean(votes))
                ci_lo, ci_hi = _wilson_ci(conf, len(votes))
                self.results_[f][g] = {
                    'confidence': round(conf, 3),
                    'n_votes':    len(votes),
                    'ci_lower':   round(ci_lo, 3),
                    'ci_upper':   round(ci_hi, 3),
                }

        self.feature_names_ = feature_names
        self.groups_        = groups
        return self

    def top_causes(self, min_confidence: float = 0.5) -> dict:
        """Retorna causa mais provável por feature com confiança >= threshold.

        Retorna dict {feature: {'cause', 'confidence', 'ci_lower', 'ci_upper'}}
        """
        causes = {}
        for f in self.feature_names_:
            valid = {g: r for g, r in self.results_[f].items()
                     if r['confidence'] is not None}
            if not valid:
                continue
            best_g = max(valid, key=lambda g: valid[g]['confidence'])
            r = valid[best_g]
            if r['confidence'] >= min_confidence:
                causes[f] = {
                    'cause':      best_g,
                    'confidence': r['confidence'],
                    'ci_lower':   r['ci_lower'],
                    'ci_upper':   r['ci_upper'],
                    'ci_str':     f"[{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]",
                }
        return causes

    def report(self, min_confidence: float = 0.5) -> str:
        """Relatório textual com causas identificadas."""
        causes = self.top_causes(min_confidence)
        lines  = [f"ShapeCausalExplainer "
                  f"(bootstrap={self.n_bootstrap}, "
                  f"min_confidence={min_confidence:.0%}):"]

        if not causes:
            lines.append(f"  Nenhuma causa identificada com confiança >= "
                         f"{min_confidence:.0%}")
        else:
            # Agrupa por causa
            by_cause = {}
            for f, r in causes.items():
                by_cause.setdefault(r['cause'], []).append((f, r))

            for g, feats in sorted(
                    by_cause.items(),
                    key=lambda x: -np.mean([r['confidence']
                                            for _, r in x[1]])):
                mean_conf = np.mean([r['confidence'] for _, r in feats])
                lines.append(f"\n  Causa identificada: '{g}' "
                             f"(confiança média={mean_conf:.0%})")
                lines.append(f"  Features afetadas: {len(feats)}")
                for f, r in sorted(feats,
                                   key=lambda x: -x[1]['confidence']):
                    stability = ("robusto" if r['ci_lower'] >= 0.5
                                 else "moderado" if r['ci_lower'] >= 0.3
                                 else "instável")
                    lines.append(
                        f"    '{f}': {r['confidence']:.0%}  "
                        f"IC95%={r['ci_str']}  [{stability}]")

        # Features sem causa clara
        uncertain = [f for f in self.feature_names_ if f not in causes]
        if uncertain:
            lines.append(f"\n  Sem causa clara ({len(uncertain)}): "
                         f"{uncertain}")
            lines.append(f"  → Forma dessas features não é determinada "
                         f"por nenhum grupo específico")

        return "\n".join(lines)

    def all_confidences(self) -> pd.DataFrame:
        """Retorna DataFrame completo com confiança de cada feature × grupo."""
        rows = []
        for f in self.feature_names_:
            for g in self.groups_:
                r = self.results_[f][g]
                rows.append({
                    'feature':    f,
                    'group':      g,
                    'confidence': r['confidence'],
                    'ci_lower':   r['ci_lower'],
                    'ci_upper':   r['ci_upper'],
                    'n_votes':    r['n_votes'],
                })
        return pd.DataFrame(rows)
