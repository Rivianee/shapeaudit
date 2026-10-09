"""
Testes da API PÚBLICA como um usuário externo a chamaria (v1.47).

Lacuna que os bugs 15 e 16 expuseram: os ~40 experimentos do projeto sempre
chamavam os detectores através de `audit()`, que já normalizava a entrada.
Ninguém nunca instanciou as classes direto, do jeito que a documentação do
SHAP ensina. Os dois bugs viviam exatamente aí.

Este arquivo cobre três coisas para cada classe pública:
  1. a chamada funciona com a saída CRUA do shap
  2. o report() devolve string não vazia (é o que o usuário lê)
  3. entradas inválidas geram erro com instrução, não resultado plausível
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
import modelcheck as M
from conftest import RF

DETECTORES_COMPARE = [
    'GeometricCanaryDetector', 'GeometryDivergenceDetector',
    'ImportanceCollapseDetector', 'PSIShapDetector', 'RHALEDetector',
    'SelectiveALEDetector', 'TopologicalDriftDetector',
    'TopologicalDriftGudhi', 'UniversalCanaryDetector',
    'ALEShapeDetector', 'ArchitectureAgnosticComparator',
]


@pytest.fixture(scope='module')
def cenario():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400),
                      'b': rng.uniform(0, 1, 400),
                      'c': rng.uniform(0, 1, 400)})
    y = pd.Series(3 * X.a + 2 * X.b + rng.randn(400) * 0.2)
    m1 = RF(0, n=30).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=30).fit(Xc, y)
    return m1, m2, X, y, list(X.columns)


@pytest.fixture(scope='module')
def shap_cru(cenario):
    """Saída crua de shap.TreeExplainer — o que a doc do SHAP ensina."""
    import shap
    m1, m2, X, _, _ = cenario

    def sv(m):
        s = shap.TreeExplainer(m).shap_values(X)
        return s[:, :, 1] if getattr(s, 'ndim', 2) == 3 else s

    return sv(m1), sv(m2)


# ═══ 1. A chamada funciona ════════════════════════════════════════════
@pytest.mark.parametrize('nome', DETECTORES_COMPARE)
def test_compare_funciona_pela_api_publica(nome, cenario):
    m1, m2, X, _, fs = cenario
    det = getattr(M, nome)()
    r = det.compare(m1, m2, X, fs)
    assert r is not None


def test_sdd_aceita_shap_cru_numpy(cenario, shap_cru):
    """
    BUG 15: `check()` usava `sv.values` e quebrava com numpy.ndarray — que é
    o que shap.TreeExplainer.shap_values() devolve por padrão.
    """
    from modelcheck.shape import ShapeDivergenceDetector
    _, _, X, _, _ = cenario
    sv1, sv2 = shap_cru
    r = ShapeDivergenceDetector().check({'v1': sv1, 'v2': sv2}, X)
    assert r is not None


def test_slope_e_silence_aceitam_shap_cru(cenario, shap_cru):
    from modelcheck.slope import ShapeSlopeDetector
    from modelcheck.silence import ShapeSilenceAnalyzer
    _, _, X, _, fs = cenario
    sv1, sv2 = shap_cru
    assert ShapeSlopeDetector().check(sv1, sv2, X, fs) is not None
    assert ShapeSilenceAnalyzer().analyze(sv1, sv2, X, X, fs) is not None


# ═══ 2. report() devolve texto ════════════════════════════════════════
@pytest.mark.parametrize('nome', DETECTORES_COMPARE)
@pytest.mark.parametrize('cen', ['controle', 'corrompido'])
def test_report_devolve_texto(nome, cen, cenario):
    """O report() é o que o usuário lê. Nunca deve ser vazio nem quebrar."""
    m1, m2, X, _, fs = cenario
    alvo = m1 if cen == 'controle' else m2
    det = getattr(M, nome)()
    txt = det.report(det.compare(m1, alvo, X, fs))
    assert isinstance(txt, str) and len(txt) > 10, \
        f"{nome}/{cen}: report vazio ou não-string"


# ═══ 3. Entradas inválidas geram erro com instrução ═══════════════════
VALIDADOS = ['GeometricCanaryDetector', 'ImportanceCollapseDetector',
             'PSIShapDetector', 'UniversalCanaryDetector',
             'SelectiveALEDetector']


@pytest.mark.parametrize('nome', VALIDADOS)
def test_feature_inexistente_levanta_erro(nome, cenario):
    """
    BUG 17: vários detectores devolviam resultado NORMAL com nome de feature
    errado. Um número plausível para entrada inválida é a pior falha.
    """
    m1, m2, X, _, _ = cenario
    with pytest.raises((ValueError, KeyError)) as e:
        getattr(M, nome)().compare(m1, m2, X, ['nao_existe'])
    assert len(str(e.value)) > 20, "erro deve ser informativo"


@pytest.mark.parametrize('nome', VALIDADOS)
def test_lista_de_features_vazia_levanta_erro(nome, cenario):
    m1, m2, X, _, _ = cenario
    with pytest.raises(ValueError):
        getattr(M, nome)().compare(m1, m2, X, [])


@pytest.mark.parametrize('nome', VALIDADOS)
def test_x_vazio_levanta_erro(nome, cenario):
    """X vazio dava frechet_distance=nan e score 100."""
    m1, m2, X, _, fs = cenario
    with pytest.raises(ValueError):
        getattr(M, nome)().compare(m1, m2, X.iloc[0:0], fs)


@pytest.mark.parametrize('nome', VALIDADOS)
def test_x_com_uma_linha_levanta_erro(nome, cenario):
    m1, m2, X, _, fs = cenario
    with pytest.raises(ValueError):
        getattr(M, nome)().compare(m1, m2, X.head(1), fs)


def test_audit_modelo_none_levanta_erro(cenario):
    m1, _, X, _, _ = cenario
    with pytest.raises(ValueError, match='None'):
        M.audit(m1, None, X, verbose=False)


def test_audit_x_none_levanta_erro(cenario):
    m1, m2, _, _, _ = cenario
    with pytest.raises(ValueError, match='None'):
        M.audit(m1, m2, None, verbose=False)


def test_audit_objeto_sem_predict_levanta_typeerror(cenario):
    m1, _, X, _, _ = cenario
    with pytest.raises(TypeError, match='predict'):
        M.audit(m1, "não é modelo", X, verbose=False)


def test_audit_aceita_numpy_como_x(cenario):
    """numpy 2D deve ser convertido, não rejeitado."""
    m1, m2, X, _, _ = cenario
    r = M.audit(m1, m2, X.values, verbose=False)
    assert r['health_score'] is not None


def test_check_features_aceita_string_unica(cenario):
    """monitored='a' não deve virar ['a','']. Aceita string única."""
    m1, m2, X, _, fs = cenario
    r = M.Guard().check_features(m1, m2, X, monitored='a', feature_names=fs)
    assert r['monitored'] == ['a']


def test_check_features_monitored_vazio_levanta_erro(cenario):
    m1, m2, X, _, fs = cenario
    with pytest.raises(ValueError, match='monitored'):
        M.Guard().check_features(m1, m2, X, monitored=[], feature_names=fs)


# ═══ 4. Contrato de retorno ════════════════════════════════════════════
def test_audit_retorna_chaves_documentadas(cenario):
    m1, m2, X, _, _ = cenario
    r = M.audit(m1, m2, X, verbose=False)
    for k in ('health_score', 'status', 'divergences', 'method_used',
              'warnings', 'model_collapsed', 'report_text'):
        assert k in r, f"chave documentada ausente: {k}"


def test_guard_audit_retorna_auditresult(cenario):
    m1, m2, X, y, fs = cenario
    r = M.Guard().audit(m1, m2, X, y, fs)
    assert r.verdict in ('APROVADO', 'ATENCAO', 'REVISAR', 'BLOQUEADO',
                         'INDETERMINADO')
    assert isinstance(r.plain, str) and len(r.plain) > 20
    assert isinstance(r.to_mlflow(), dict)
    assert isinstance(r.to_slack(), dict)


# ═══ 5. Health Score aposentado da interface (v1.47) ══════════════════
def test_plain_nao_mostra_o_score(cenario):
    """
    O índice de saúde saiu da saída primária. Cinco medições mostraram que é
    o sinal mais fraco — inclusive INVERTEU a ordem num caso medido
    (retreino legítimo 43 BLOQUEADO vs pipeline quebrado 57 REVISAR).
    """
    m1, m2, X, y, fs = cenario
    txt = M.Guard().audit(m1, m2, X, y, fs).plain
    primeira = txt.split('\n')[0]
    assert 'índice' not in primeira.lower(), \
        f"o score não deve aparecer na primeira linha: {primeira}"
    assert 'de 100' not in txt, f"o score não deve aparecer no plain: {txt[:120]}"


def test_plain_debug_mostra_o_score_com_ressalva(cenario):
    m1, m2, X, y, fs = cenario
    txt = M.Guard().audit(m1, m2, X, y, fs).plain_debug()
    assert 'índice de saúde' in txt
    assert 'não compare' in txt, "o modo depuração deve avisar sobre o uso"
    assert 'LIMITATIONS' in txt


def test_summary_e_a_saida_primaria(cenario):
    m1, m2, X, y, fs = cenario
    s = M.Guard().audit(m1, m2, X, y, fs).summary
    for k in ('verdict', 'problem_type', 'affected_features', 'action', 'mode'):
        assert k in s, f"campo primário ausente: {k}"
    assert 'score' not in s, "o score não pertence à saída primária"


def test_repr_prioriza_diagnostico_nao_numero(cenario):
    m1, m2, X, y, fs = cenario
    r = repr(M.Guard().audit(m1, m2, X, y, fs))
    assert 'type=' in r and 'features=' in r
    assert 'score=' not in r, f"repr não deve destacar o score: {r}"


def test_slack_sem_score_no_cabecalho(cenario):
    m1, m2, X, y, fs = cenario
    txt = M.Guard().audit(m1, m2, X, y, fs).to_slack('modelo-teste')['text']
    cabecalho = txt.split('\n')[0]
    assert '/100' not in cabecalho, f"cabeçalho do Slack com score: {cabecalho}"


def test_score_segue_acessivel_para_compatibilidade(cenario):
    """Aposentar da interface não é remover da API."""
    m1, m2, X, y, fs = cenario
    r = M.Guard().audit(m1, m2, X, y, fs)
    assert r.score is not None
    assert r.debug_score == r.score
    assert 'modelguard.health_score' in r.to_mlflow()


# ═══ 5. Hierarquia da saída (v1.47) ═══════════════════════════════════
# O score é o componente mais fraco (LIMITATIONS §3) e estava no lugar mais
# visível. Estes testes travam a inversão: diagnóstico e features primeiro,
# score como detalhe técnico no rodapé.

def test_relatorio_comeca_pelo_diagnostico_nao_pelo_score(cenario):
    from modelcheck.orchestrator import ModelGuardOrchestrator
    m1, m2, X, y, fs = cenario
    r = ModelGuardOrchestrator().diagnose(m1, m2, X, y, fs)
    cabecalho = "\n".join(r['report_text'].split('\n')[:8])
    assert 'HEALTH SCORE GLOBAL' not in cabecalho, \
        "o score não deve mais dominar o cabeçalho"
    assert r['interpretation']['problem_type'] in cabecalho, \
        "o tipo de anomalia deve aparecer no cabeçalho"
    assert 'Features afetadas' in cabecalho, \
        "as features afetadas devem aparecer no cabeçalho"


def test_relatorio_mantem_score_no_rodape_com_ressalva(cenario):
    """O score continua acessível — só deixa de ser a manchete."""
    from modelcheck.orchestrator import ModelGuardOrchestrator
    m1, m2, X, y, fs = cenario
    r = ModelGuardOrchestrator().diagnose(m1, m2, X, y, fs)
    rodape = "\n".join(r['report_text'].split('\n')[-6:])
    assert 'índice de saúde' in rodape
    assert 'LIMITATIONS' in rodape, \
        "o rodapé deve apontar para a limitação documentada"
    assert r['global_score'] is not None, \
        "o score continua no dict de retorno para uso programático"


def test_features_afetadas_aparecem_no_cabecalho(cenario):
    """Com corrupção conhecida em 'a', o cabeçalho deve citar 'a'."""
    from modelcheck.orchestrator import ModelGuardOrchestrator
    m1, m2, X, y, fs = cenario
    r = ModelGuardOrchestrator().diagnose(m1, m2, X, y, fs)
    cabecalho = "\n".join(r['report_text'].split('\n')[:8])
    assert "'a'" in cabecalho, f"'a' (corrompida) deve estar no topo"
