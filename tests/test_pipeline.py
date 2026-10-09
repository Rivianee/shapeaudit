"""
Composição com ferramentas de deriva (v1.75).

Nenhuma equipe usa uma ferramenta só. O ModelGuard responde "o modelo mudou?";
Evidently, NannyML e Alibi respondem "os dados mudaram?". As duas perguntas
são independentes — é o argumento central do método — e a leitura útil vem da
combinação.

Os testes abaixo não dependem do Evidently instalado: passam a lista de
colunas derivadas diretamente, que é como a API foi desenhada. Só o teste
marcado com `importorskip` exercita a integração real.
"""
import warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
import modelcheck as M
from modelcheck.pipeline import combinar, exit_code, evidently_drifted_columns


@pytest.fixture(scope='module')
def auditorias():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 500), 'b': rng.uniform(0, 1, 500)})
    y = pd.Series(3 * X.a + 2 * X.b + rng.randn(500) * .2)
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=40).fit(Xc, y)
    g = M.Guard()
    return {
        # X_train é o treino do modelo NOVO: em `Xc` a coluna 'a' é
        # constante, que é o que caracteriza falha de pipeline (v0.12.0).
        'quebrado': g.audit(m1, m2, X, y, list(X.columns), X_train=Xc),
        'saudavel': g.audit(m1, RF(7, n=40).fit(X, y), X, y, list(X.columns),
                            X_train=X),
    }


# ═══ As quatro linhas da matriz ═══════════════════════════════════════
@pytest.mark.parametrize('caso,cols,esperado,tipo', [
    ('quebrado', [],          'BLOQUEADO', 'PIPELINE_FAILURE'),
    ('saudavel', ['idade'],   'ATENCAO',   'DATA_SHIFT'),
    ('quebrado', ['idade'],   'BLOQUEADO', 'BOTH'),
    ('saudavel', [],          'APROVADO',  'HEALTHY'),
])
def test_matriz_de_decisao(auditorias, caso, cols, esperado, tipo):
    r = combinar(auditorias[caso], cols)
    assert r['verdict'] == esperado
    assert r['problem_type'] == tipo


def test_leitura_explica_a_acao(auditorias):
    """Cada combinação tem uma ação diferente, e o texto precisa dizer qual."""
    r = combinar(auditorias['quebrado'], [])
    assert 'ETL' in r['reading']
    r = combinar(auditorias['saudavel'], ['x'])
    assert 'Retreine' in r['reading'] or 'recalibr' in r['reading'].lower()


# ═══ "não rodou" é diferente de "não achou" ═══════════════════════════
def test_drift_nao_executado_nao_vira_ausencia_de_drift(auditorias):
    """
    O defeito mais frequente deste projeto é confundir "não verifiquei" com
    "está tudo bem". Aqui: sem a ferramenta de deriva, não dá para separar
    falha de pipeline de mudança real — e a matriz não deve ser aplicada.
    """
    r = combinar(auditorias['quebrado'], drift_ran=False)
    assert r['problem_type'] == 'MODEL_ONLY'
    assert r['data_drifted'] is None, \
        'sem execução, o sinal de deriva é desconhecido, não False'
    lac = [i for i in r['not_evaluated'] if i['layer'] == 'data drift']
    assert lac and lac[0]['severity'] == 'alta'


def test_lista_vazia_significa_rodou_e_nao_achou(auditorias):
    r = combinar(auditorias['quebrado'], [])
    assert r['data_drifted'] is False
    assert not [i for i in r['not_evaluated'] if i['layer'] == 'data drift']


def test_lacunas_do_audit_sao_preservadas(auditorias):
    """A combinação não pode descartar o que o audit já declarava."""
    orig = {i['layer'] for i in auditorias['quebrado'].not_evaluated}
    comb = {i['layer'] for i in combinar(auditorias['quebrado'], [])['not_evaluated']}
    assert orig.issubset(comb)


# ═══ CI ═══════════════════════════════════════════════════════════════
@pytest.mark.parametrize('cols,caso,esperado', [
    ([], 'quebrado', 1), ([], 'saudavel', 0), (['x'], 'saudavel', 0)])
def test_exit_code(auditorias, cols, caso, esperado):
    assert exit_code(combinar(auditorias[caso], cols), fail_on='high') == esperado


# ═══ Integração real ══════════════════════════════════════════════════
def test_extrator_do_evidently():
    """
    Exercita o formato real da 0.7.x. Se o Evidently mudar o formato do
    dicionário, este teste falha — melhor que devolver lista vazia
    silenciosamente, que a matriz leria como "não achou nada".
    """
    ev = pytest.importorskip('evidently', reason='dependência opcional')
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'renda': rng.lognormal(9, .5, 800),
                      'idade': rng.randint(21, 70, 800)})
    Xs = X.copy(); Xs['idade'] = Xs['idade'] + 15
    dd = DataDefinition()
    rep = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(Xs, data_definition=dd),
        Dataset.from_pandas(X, data_definition=dd))
    cols = evidently_drifted_columns(rep.dict())
    assert cols == ['idade'], f'esperava idade, veio {cols}'


def test_formato_desconhecido_devolve_vazio():
    assert evidently_drifted_columns({'metrics': [{'foo': 'bar'}]}) == []
    assert evidently_drifted_columns({}) == []


# ═══ fairlearn: disparidade entre grupos (v1.76) ══════════════════════
# Cobre o que o FairnessChecker deste pacote NÃO mede e declara que não
# mede. Delegar é deliberado: medir disparidade bem é um campo inteiro, e
# um detector nosso não validado num tema onde o falso negativo tem
# consequência não seria responsável.

def test_fairlearn_detecta_disparidade():
    pytest.importorskip('fairlearn', reason='dependência opcional')
    from modelcheck.pipeline import fairlearn_disparity
    rng = np.random.RandomState(0)
    n = 1500
    g = rng.binomial(1, .5, n)
    yt = rng.binomial(1, .4, n)
    yp = np.where(g == 1, rng.binomial(1, .7, n), rng.binomial(1, .2, n))
    r = fairlearn_disparity(yt, yp, g)
    assert r['status'] == 'DISPARITY'
    assert r['difference'] > 0.10
    assert len(r['by_group']) == 2


def test_fairlearn_nao_acusa_modelo_equilibrado():
    pytest.importorskip('fairlearn', reason='dependência opcional')
    from modelcheck.pipeline import fairlearn_disparity
    rng = np.random.RandomState(1)
    n = 1500
    g = rng.binomial(1, .5, n)
    yt = rng.binomial(1, .4, n)
    yp = rng.binomial(1, .4, n)
    assert fairlearn_disparity(yt, yp, g)['status'] == 'OK'


def test_fairlearn_com_um_grupo_e_nao_avaliavel():
    """Comparar exige dois grupos — devolver OK seria falso."""
    pytest.importorskip('fairlearn', reason='dependência opcional')
    from modelcheck.pipeline import fairlearn_disparity
    rng = np.random.RandomState(2)
    n = 500
    r = fairlearn_disparity(rng.binomial(1, .4, n), rng.binomial(1, .4, n),
                            np.zeros(n))
    assert r['status'] == 'NAO_AVALIAVEL'
    assert 'grupo' in r['reason']


def test_fairlearn_ausente_e_declarado(monkeypatch):
    """
    Sem a dependência, o retorno precisa dizer que nada foi verificado —
    e não confundir "não instalado" com "sem disparidade".
    """
    import builtins
    from modelcheck.pipeline import fairlearn_disparity
    real = builtins.__import__

    def falso(nome, *a, **k):
        if nome.startswith('fairlearn'):
            raise ImportError('simulado')
        return real(nome, *a, **k)

    monkeypatch.setattr(builtins, '__import__', falso)
    r = fairlearn_disparity([0, 1], [0, 1], [0, 1])
    assert r['status'] == 'NAO_AVALIAVEL'
    assert 'pip install' in r['reason']


def test_fairlearn_declara_o_que_nao_mede():
    pytest.importorskip('fairlearn', reason='dependência opcional')
    from modelcheck.pipeline import fairlearn_disparity
    rng = np.random.RandomState(3)
    n = 400
    r = fairlearn_disparity(rng.binomial(1, .4, n), rng.binomial(1, .4, n),
                            rng.binomial(1, .5, n))
    assert 'taxa de seleção sozinha pode esconder' in r['note']


# ═══ v0.9.1 — bug 35: o dano do FMMO parecia melhora ═════════════════
def _cenario_fmmo():
    rng = np.random.RandomState(0); n = 4000
    g = rng.binomial(1, .3, n); y = rng.binomial(1, .25, n)
    pred_a = np.where(y == 1, rng.binomial(1, .7, n), rng.binomial(1, .02, n))
    pred_d = pred_a.copy()
    m = (g == 1) & (y == 0)                       # só o grupo 1, só negativos
    pred_d[m] = rng.binomial(1, .08, m.sum())     # FPR do grupo 1 sobe ~4x
    return y, pred_a, pred_d, g


def test_evolucao_detecta_fpr_subindo_num_grupo():
    pytest.importorskip('fairlearn')
    from modelcheck.pipeline import fairness_evolucao
    y, a, d, g = _cenario_fmmo()
    r = fairness_evolucao(y, a, g, y, d, g)
    assert r['status'] == 'PIOROU'
    assert any(p['grupo'] == '1' and p['metrica'] == 'fpr' for p in r['pioras'])


def test_disparidade_reporta_fpr_e_nao_so_selecao():
    """Taxa de seleção sozinha fazia o dano parecer melhora (bug 35)."""
    pytest.importorskip('fairlearn')
    from modelcheck.pipeline import fairlearn_disparity
    y, a, d, g = _cenario_fmmo()
    r = fairlearn_disparity(y, d, g)
    assert {'selection_rate', 'fpr', 'fnr'} <= set(r['metrics'])


def test_evolucao_sem_mudanca_e_ok():
    pytest.importorskip('fairlearn')
    from modelcheck.pipeline import fairness_evolucao
    y, a, _, g = _cenario_fmmo()
    assert fairness_evolucao(y, a, g, y, a, g)['status'] == 'OK'


def test_limiar_fixo_nao_volta():
    """BUG 36: limiar fixo de +0.02 dava alarme em 95% dos retreinos
    saudáveis no German Credit (grupo de 68 pessoas). Exige teste."""
    import inspect
    from modelcheck import pipeline
    src = inspect.getsource(pipeline.fairness_evolucao)
    assert 'binomtest' in src and 'excesso' in src


def test_combinar_declara_fairness_ausente():
    from modelcheck.pipeline import combinar
    class R: root_cause_features = []; verdict = 'APROVADO'; not_evaluated = []
    c = combinar(R(), [])
    assert any(i['layer'] == 'fairness evolution' for i in c['not_evaluated'])


def test_combinar_eleva_quando_grupo_piora():
    from modelcheck.pipeline import combinar
    class R: root_cause_features = []; verdict = 'APROVADO'; not_evaluated = []
    f = {'status': 'PIOROU', 'pioras': [{'grupo': '1', 'metrica': 'fpr',
                                          'antes': .01, 'depois': .05}]}
    c = combinar(R(), [], fairness=f)
    assert c['verdict'] == 'ATENCAO' and 'FAIRNESS_SHIFT' in c['problem_type']
