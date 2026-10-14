"""
Fairness (v1.64) — módulo que estava sem teste.

Achado ao verificar a pergunta "o que acontece quando há integridade +
fairness ao mesmo tempo?": o módulo existia, não tinha teste, e tinha o
mesmo defeito do bug 15 mais um novo (bug 29).

NOTA DE ESCOPO: este módulo mede o peso direto de uma variável declarada
como sensível. Não detecta proxy nem disparidade entre grupos, e NÃO
participa do veredito de `Guard.audit()`. Os testes abaixo verificam o que
ele faz, não fazem dele uma avaliação de justiça.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from modelcheck.fairness import FairnessChecker


@pytest.fixture(scope='module')
def cenario():
    """Modelo com dependência REAL de 'genero' — ground truth conhecido."""
    import shap
    rng = np.random.RandomState(0)
    n = 1200
    g = rng.binomial(1, .5, n)
    X = pd.DataFrame({'genero': g, 'renda': rng.lognormal(9, .6, n),
                      'idade': rng.randint(21, 70, n),
                      'score': rng.uniform(0, 1, n)})
    y = pd.Series(((0.9 * g + 0.5 * X.score + rng.randn(n) * .25) > .7)
                  .astype(int))
    m = RandomForestClassifier(n_estimators=50, max_depth=6,
                               random_state=0, n_jobs=1).fit(X, y)
    s = shap.TreeExplainer(m).shap_values(X)
    from modelcheck.frechet import matriz_shap
    return X, matriz_shap(s)   # shap antigo devolve lista (BUG 95)


# ═══ BUG 29: coluna inexistente dava aprovação silenciosa ═════════════
def test_variavel_sensivel_inexistente_levanta_erro(cenario):
    """
    A versão anterior devolvia lista vazia — indistinguível de "nenhuma
    violação". Num módulo de fairness é o pior modo de falha: um erro de
    digitação no nome da coluna produzia laudo de aprovação sem que nada
    tivesse sido verificado.
    """
    X, sv = cenario
    with pytest.raises(ValueError, match='ausentes'):
        FairnessChecker(['raca']).check(sv, X)


def test_lista_de_sensiveis_vazia_levanta_erro():
    with pytest.raises(ValueError, match='vazio'):
        FairnessChecker([])


def test_max_weight_invalido_levanta_erro():
    for w in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match='max_weight'):
            FairnessChecker(['a'], max_weight=w)


# ═══ BUG 15 de novo: numpy cru ════════════════════════════════════════
def test_aceita_shap_cru_numpy(cenario):
    """shap.TreeExplainer devolve ndarray; a versão anterior quebrava."""
    X, sv = cenario
    r = FairnessChecker(['genero']).check(sv, X)
    assert r['status'] == 'VIOLATION'


def test_aceita_dataframe(cenario):
    X, sv = cenario
    r = FairnessChecker(['genero']).check(pd.DataFrame(sv, columns=X.columns), X)
    assert r['status'] == 'VIOLATION'


# ═══ Ground truth ═════════════════════════════════════════════════════
def test_detecta_variavel_com_peso_real(cenario):
    X, sv = cenario
    r = FairnessChecker(['genero']).check(sv, X)
    assert r['weights']['genero'] > 0.2
    assert r['violations'][0]['feature'] == 'genero'


def test_nao_acusa_variavel_de_peso_baixo(cenario):
    X, sv = cenario
    r = FairnessChecker(['renda']).check(sv, X)
    assert r['status'] == 'OK'
    assert r['violations'] == []


def test_modelo_constante_e_nao_avaliavel(cenario):
    """Atribuições todas zero: nada foi verificado, e o retorno diz isso."""
    X, sv = cenario
    r = FairnessChecker(['genero']).check(np.zeros_like(sv), X)
    assert r['status'] == 'NAO_AVALIAVEL'
    assert 'reason' in r


# ═══ Contrato ═════════════════════════════════════════════════════════
def test_x_sem_nomes_de_coluna_levanta_erro(cenario):
    X, sv = cenario
    with pytest.raises(TypeError, match='DataFrame'):
        FairnessChecker(['genero']).check(sv, X.to_numpy())


def test_dimensoes_incompativeis_levantam_erro(cenario):
    X, sv = cenario
    with pytest.raises(ValueError, match='colunas'):
        FairnessChecker(['genero']).check(sv[:, :2], X)


def test_retorno_declara_o_escopo(cenario):
    """O usuário precisa saber o que NÃO foi verificado."""
    X, sv = cenario
    r = FairnessChecker(['genero']).check(sv, X)
    assert 'proxy' in r['note']


# ═══ Decisão de projeto: fora do veredito ═════════════════════════════
def test_fairness_nao_participa_do_veredito():
    """
    Decisão explícita: integrar exigiria validação com ground truth que não
    temos, e em fairness um falso negativo tem consequência que um defeito
    técnico não tem. Este teste falha se alguém integrar sem revisar.
    """
    import inspect, re
    from modelcheck import api, orchestrator
    for mod in (api, orchestrator):
        src = inspect.getsource(mod)
        # procura USO real (import ou instanciação), não menção em texto —
        # o `not_evaluated` cita 'fairness' de propósito, para declarar que
        # a camada não rodou.
        usa = (re.search(r'from\s+modelcheck\.fairness\s+import', src) or
               re.search(r'FairnessChecker\s*\(', src))
        assert not usa, (
            f"{mod.__name__} passou a USAR FairnessChecker. Antes de "
            f"integrar ao veredito, valide com ground truth e revise a "
            f"nota de escopo em fairness.py.")
