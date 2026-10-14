"""Divergência de forma: modelos capturam formatos diferentes da mesma feature."""
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _adaptive_bins(n):
    return int(np.clip(n // 40, 5, 7))


def _shape_profile_small_n(feature_values, feature_shap, min_n=8):
    """Classifica forma da curva SHAP para amostras pequenas (n < 30).

    Combina 3 métodos por votação majoritária:
      1. Regressão polinomial grau 2 — detecta curvatura via coeficientes
      2. Bootstrap + bins — reamostragem para estabilizar bins com n pequeno
      3. Spearman split — correlação na metade inferior vs superior de x

    Retorna dict com 'shape', 'confidence' (0–1) e 'n', ou None se n < min_n.

    Limitações honestas:
      - Formas assimétricas (ex: sobe rápido e cai devagar) podem ser
        classificadas como crescente ou decrescente em vez de u_invertido
      - Ruído sigma > 2x amplitude do sinal pode causar misclassificação
      - Não substitui _shape_profile para n >= 30; usar como complemento
    """
    x = np.asarray(feature_values, dtype=float)
    y = np.asarray(feature_shap, dtype=float)
    n = len(x)

    if n < min_n or np.std(x) < 1e-8:
        return None

    # ── Método 1: Regressão polinomial grau 2 ──────────────────
    def _regression(x, y):
        try:
            x_n = (x - x.mean()) / (x.std() + 1e-8)
            coeffs = np.polyfit(x_n, y, 2)
            c, b = coeffs[0], coeffs[1]  # cx² + bx + a
            curvature = abs(c) / (abs(b) + abs(c) + 1e-8)
            if curvature < 0.25:
                return 'crescente' if b > 0 else 'decrescente'
            return 'u' if c > 0 else 'u_invertido'
        except Exception:
            return None

    # ── Método 2: Bootstrap + bins ─────────────────────────────
    def _bootstrap(x, y, n_boot=80, n_bins=4):
        shapes = []
        # BUG 78 (v1.0.4): sem semente, o nome da forma mudava de uma
        # execução para outra quando 20 <= n < 30.
        rng = np.random.RandomState(0)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            xb, yb = x[idx], y[idx]
            try:
                bins = pd.qcut(xb, n_bins, duplicates='drop')
                means = pd.Series(yb).groupby(bins, observed=True).mean().values
                if len(means) < 3:
                    continue
                diffs = np.diff(means)
                up = (diffs > 0).sum(); down = (diffs < 0).sum()
                if max(up, down) / len(diffs) >= 0.7:
                    shapes.append('crescente' if up > down else 'decrescente')
                else:
                    km = np.argmin(means); kM = np.argmax(means)
                    if 0 < kM < len(means) - 1:
                        shapes.append('u_invertido')
                    elif 0 < km < len(means) - 1:
                        shapes.append('u')
                    else:
                        shapes.append('irregular')
            except Exception:
                pass
        if not shapes:
            return None
        from collections import Counter
        return Counter(shapes).most_common(1)[0][0]

    # ── Método 3: Spearman split ────────────────────────────────
    def _spearman_split(x, y):
        idx = np.argsort(x); mid = n // 2
        r1, _ = scipy_stats.spearmanr(x[idx[:mid]], y[idx[:mid]])
        r2, _ = scipy_stats.spearmanr(x[idx[mid:]], y[idx[mid:]])
        if np.isnan(r1) or np.isnan(r2):
            return None
        if r1 > 0.3 and r2 > 0.3:   return 'crescente'
        if r1 < -0.3 and r2 < -0.3: return 'decrescente'
        if r1 > 0.3 and r2 < -0.3:  return 'u_invertido'
        if r1 < -0.3 and r2 > 0.3:  return 'u'
        return 'irregular'

    # ── Votação majoritária ─────────────────────────────────────
    votes = [v for v in [_regression(x, y), _bootstrap(x, y), _spearman_split(x, y)]
             if v is not None]
    if not votes:
        return None

    from collections import Counter
    shape, count = Counter(votes).most_common(1)[0]
    return {
        'shape':      shape,
        'confidence': round(count / 3, 2),
        'n':          n,
        'method':     'small_n_ensemble',
        'votes':      votes,
    }


def _shape_profile(feature_values, feature_shap, n_bins=None,
                   per_bin_test=False, alpha=0.05):
    """Perfil por bins com flag opcional de significância por bin."""
    n = len(feature_values)
    if np.std(feature_values) == 0 or n < 8:
        return None
    # Para amostras pequenas usa ensemble de métodos alternativos
    if n < 30:
        return _shape_profile_small_n(feature_values, feature_shap, min_n=8)
    if n_bins is None:
        n_bins = _adaptive_bins(n)
    try:
        bins = pd.qcut(feature_values, n_bins, duplicates="drop")
    except ValueError:
        return None

    df = pd.DataFrame({"b": bins, "s": feature_shap})
    groups = df.groupby("b", observed=True)["s"]
    means = groups.mean().to_numpy()
    if len(means) < 3:
        return None

    if per_bin_test:
        # Wilcoxon signed-rank por bin: sinal real vs ruído
        bin_vals = [grp.to_numpy() for _, grp in df.groupby("b", observed=True)["s"]]
        significant = []
        for bv in bin_vals:
            if len(bv) < 5:
                significant.append(False)
                continue
            try:
                _, pv = scipy_stats.wilcoxon(bv)
                significant.append(pv < alpha)
            except Exception:
                significant.append(False)
        # usar apenas bins significativos para monotonia
        sig_means = [m for m, s in zip(means, significant) if s]
        if len(sig_means) >= 3:
            means = np.array(sig_means)

    diffs = np.diff(means)
    up = int((diffs > 0).sum())
    down = int((diffs < 0).sum())
    mono_ratio = max(up, down) / len(diffs)
    monotonic = mono_ratio >= 0.8

    k_min = int(np.argmin(means))
    k_max = int(np.argmax(means))
    interior_min = 0 < k_min < len(means) - 1
    interior_max = 0 < k_max < len(means) - 1

    if monotonic:
        shape = "crescente" if up >= down else "decrescente"
    elif interior_max and not interior_min:
        shape = "u_invertido"
    elif interior_min and not interior_max:
        shape = "u"
    else:
        shape = "irregular"

    return {"means": means, "monotonic": monotonic, "shape": shape}


class ShapeDivergenceDetector:
    """Alerta quando modelos capturam FORMAS diferentes da mesma relação.

    Suporta SHAP e ALE como entrada de atribuição.
    Suporta per-bin significance testing para relaxar o requisito n>=50.
    """

    def __init__(self, min_weight=0.01, per_bin_test=False, alpha=0.05,
                 confirm_bootstrap=False, min_concordance=0.85, n_boot=12,
                 tau=0.30):
        """
        BUG 28 (v1.60): esta classe e `audit()` implementam o mesmo cálculo em
        código separado, e os dois divergiam. Duas causas:

          - `min_weight` era 0.03 aqui e `min_rel_importance` 0.01 no audit()
          - o filtro de espalhamento (tau) existia no audit() e NÃO aqui

        Resultado medido no Wine (RF vs GB): audit() reportava 3 features,
        esta classe reportava 2, e no caso de feature neutralizada os dois
        conjuntos eram DIFERENTES, não apenas de tamanhos diferentes.

        Os defaults agora são idênticos aos do audit(), e há um teste de
        consistência que falha se voltarem a divergir.
        """
        self.min_weight = min_weight
        self.tau = tau
        self.per_bin_test = per_bin_test
        self.alpha = alpha
        # v1.59: confirmação por bootstrap. Ver `_forma_estavel` para as
        # medições que motivaram substituir o filtro de importância.
        self.confirm_bootstrap = confirm_bootstrap
        self.min_concordance = min_concordance
        self.n_boot = n_boot

    @staticmethod
    def _as_array(sv):
        """
        BUG 15 (v1.46): `check()` assumia que os SHAP values eram DataFrame
        (usava `sv.values` direto) e levantava
        `AttributeError: 'numpy.ndarray' object has no attribute 'values'`
        com array — que é justamente o que `shap.TreeExplainer.shap_values()`
        devolve por padrão. Quem chamasse a classe diretamente com a saída
        crua do SHAP recebia exceção.
        """
        from .frechet import matriz_shap
        return matriz_shap(sv)   # BUG 95: aceita a lista do shap antigo

    @staticmethod
    def _forma_estavel(values, sv_col, n_boot=8, frac=0.7, seed=0):
        """
        Forma modal e concordância entre subamostras (v1.59).

        Substitui o filtro de espalhamento de importância como critério de
        confirmação. Medido em 160 pares de benchmark com importância
        preservada por construção:

            filtro de importância (tau=0.30) : det  6%, FP  0%, F1 0.12
            sem filtro nenhum                : det 100%, FP 58%, F1 0.77
            confirmacao por bootstrap        : det 100%, FP  8%, F1 0.96

        O filtro anterior exigia que a IMPORTANCIA mudasse junto com a forma,
        o que por construção nao acontece em Shape Instability pura -- por
        isso detectava apenas 6%. A confirmacao por bootstrap testa se a forma
        e estavel sob reamostragem, que e a propriedade relevante.
        """
        rng = np.random.RandomState(seed)
        n = len(values)
        # v1.59b: a fração reamostrada precisa crescer quando n é pequeno.
        # Medido: com n=177 e frac=0.70 restam ~124 pontos por réplica, a
        # forma fica instável e a confirmação a gamma=0.85 zera a detecção
        # (0 divergências onde o critério anterior achava 1). Com n grande
        # (n>=400) a fração menor é preferível porque dá réplicas mais
        # independentes.
        if n < 250:
            frac = 0.90
        elif n < 400:
            frac = 0.80
        k = max(int(n * frac), 50)
        if n < 60:
            p = _shape_profile(values, sv_col)
            return (p['shape'], 1.0) if p else (None, 0.0)
        formas = []
        for _ in range(n_boot):
            idx = rng.choice(n, min(k, n), replace=False)
            p = _shape_profile(np.asarray(values)[idx],
                               np.asarray(sv_col)[idx])
            if p:
                formas.append(p['shape'])
        if not formas:
            return None, 0.0
        vals, cnts = np.unique(formas, return_counts=True)
        j = int(np.argmax(cnts))
        return str(vals[j]), float(cnts[j] / len(formas))

    def check(self, shap_by_model: dict, X):
        alerts = []
        weights = {}
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(
                np.asarray(X),
                columns=[f"f{i}" for i in range(np.asarray(X).shape[1])])
        shap_by_model = {m: self._as_array(sv)
                         for m, sv in shap_by_model.items()}
        for m, sv in shap_by_model.items():
            imp = np.abs(sv).mean(axis=0)
            tot = imp.sum()
            weights[m] = imp / tot if tot > 0 else imp

        for i, feature in enumerate(X.columns):
            if not pd.api.types.is_numeric_dtype(X[feature]):
                continue
            # alerta se pelo menos UM modelo vê a feature como relevante
            if all(w[i] < self.min_weight for w in weights.values()):
                continue
            values = X.iloc[:, i].to_numpy(dtype=float)
            profiles = {}
            for model, sv in shap_by_model.items():
                p = _shape_profile(values, sv[:, i],
                                   per_bin_test=self.per_bin_test,
                                   alpha=self.alpha)
                if p:
                    profiles[model] = p
            if len(profiles) < 2:
                continue

            shapes = {m: p["shape"] for m, p in profiles.items()}
            if len(set(shapes.values())) <= 1:
                continue

            # v1.59: confirma que cada forma divergente é ESTÁVEL sob
            # reamostragem antes de reportar. Substitui o filtro que exigia
            # mudança de importância — este detectava só 6% da Instabilidade
            # de Forma pura (F1 0.12); a confirmação por bootstrap detecta
            # 100% com 8% de FP (F1 0.96), medido em 160 pares.
            if self.confirm_bootstrap:
                estaveis, concord = {}, {}
                for m, sv_m in shap_by_model.items():
                    fm, cc = self._forma_estavel(
                        values, sv_m[:, i], n_boot=self.n_boot,
                        seed=abs(hash(str(m))) % (2 ** 31))
                    estaveis[m] = fm
                    concord[m] = cc
                if any(v is None or concord[m] < self.min_concordance
                       for m, v in estaveis.items()):
                    continue                    # forma instável: não reporta
                if len(set(estaveis.values())) <= 1:
                    continue                    # discordância era ruído
                shapes = estaveis
            elif self.tau is not None:
                # BUG 28: o audit() exige espalhamento de importância >= tau
                # e esta classe não exigia. Mesmo critério agora.
                imps = {m: float(np.abs(sv_m[:, i]).mean())
                        for m, sv_m in shap_by_model.items()}
                vals = list(imps.values())
                spread = ((max(vals) - min(vals)) / (max(vals) + 1e-12)
                          if vals else 0.0)
                if spread < self.tau:
                    continue

            # BUG 16 (v1.46): com n pequeno o perfil vem de
            # `_shape_profile_small_n`, que NÃO tem a chave 'monotonic' —
            # gerava KeyError e derrubava o check() inteiro. Qualquer usuário
            # com conjunto de avaliação pequeno recebia exceção.
            # Derivamos a monotonicidade da própria forma classificada.
            _MONO = {'crescente', 'decrescente'}

            def _e_monotonico(p):
                if 'monotonic' in p:
                    return bool(p['monotonic'])
                return p.get('shape') in _MONO

            nonlinear = [m for m, p in profiles.items() if not _e_monotonico(p)]
            linearish = [m for m, p in profiles.items() if _e_monotonico(p)]

            if nonlinear and linearish:
                msg = (f"\'{feature}\': {nonlinear} capturam relação "
                       f"não-monotônica ({shapes[nonlinear[0]]}) que "
                       f"{linearish} tratam como linear. Possível "
                       f"underfitting em {linearish}.")
                sugg = (f"Adicione \'{feature}²\' como feature para os "
                        "modelos lineares ou prefira modelo não-linear.")
            else:
                msg = (f"\'{feature}\': modelos capturam formas "
                       f"DIFERENTES: {shapes}.")
                sugg = ("Compare curvas de dependência parcial e valide "
                        "contra conhecimento de domínio.")

            alerts.append({
                "feature": feature,
                "shapes": shapes,
                "nonlinear_models": nonlinear,
                "monotonic_models": linearish,
                "message": msg,
                "suggestion": sugg,
            })
        return alerts
