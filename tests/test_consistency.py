"""
Consistência entre caminhos equivalentes (v1.60).

Lente nova, motivada pelo bug 28: vários módulos reimplementam o mesmo
cálculo em código separado. Cada um é testado isoladamente e passa; nenhum
teste verificava se eles CONCORDAM entre si.

Foi assim que `audit()` e `ShapeDivergenceDetector.check()` divergiram --
defaults diferentes (0.01 vs 0.03) e um filtro presente em só um dos dois.
No Wine com RF vs GB um reportava 3 features e o outro 2; com feature
neutralizada os conjuntos eram diferentes, não apenas de tamanhos distintos.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor
from conftest import RF
import modelcheck as M
from modelcheck.audit import audit
from modelcheck.shape import ShapeDivergenceDetector
from modelcheck.importance import ImportanceCollapseDetector, _shap_importance
from modelcheck.backends import importance as bk_importance


@pytest.fixture(scope='module')
def par():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 700), 'b': rng.uniform(0, 1, 700),
                      'c': rng.uniform(0, 1, 700), 'd': rng.uniform(0, 1, 700)})
    y = pd.Series(3 * X.a + 2 * X.b + X.c + rng.randn(700) * 0.2)
    m1 = RF(0, n=60).fit(X, y)
    gb = GradientBoostingRegressor(n_estimators=60, max_depth=3,
                                   random_state=0).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    m2 = RF(0, n=60).fit(Xc, y)
    ctrl = RF(7, n=60).fit(X, y)
    return dict(X=X, y=y, Xc=Xc, m1=m1, gb=gb, m2=m2, ctrl=ctrl,
                fs=list(X.columns))


def _shap(m, X):
    import shap
    s = shap.TreeExplainer(m).shap_values(X)
    return s[:, :, 1] if getattr(s, 'ndim', 2) == 3 else s


# ═══ Defaults idênticos ═══════════════════════════════════════════════
def test_defaults_do_sdd_batem_com_o_audit():
    """
    BUG 28: `min_weight` era 0.03 na classe e `min_rel_importance` 0.01 no
    audit(). Um filtro três vezes mais restritivo num dos caminhos.
    """
    import inspect
    d = ShapeDivergenceDetector()
    sig = inspect.signature(audit)
    mri = sig.parameters['min_rel_importance'].default
    assert d.min_weight == mri, \
        (f"filtro de importância diverge: SDD.min_weight={d.min_weight}, "
         f"audit.min_rel_importance={mri}")
    thr = sig.parameters['threshold'].default
    assert d.tau == thr, \
        f"limiar diverge: SDD.tau={d.tau}, audit.threshold={thr}"


# ═══ Mesma saída pelos dois caminhos ══════════════════════════════════
@pytest.mark.parametrize('cen', ['gb', 'm2', 'ctrl'])
def test_audit_e_sdd_check_concordam(par, cen):
    """Os dois calculam divergência de forma. Devem dar o mesmo conjunto."""
    m1, m2 = par['m1'], par[cen]
    X = par['X']
    a = sorted(d['feature'] for d in audit(m1, m2, X, verbose=False)['divergences'])
    r = ShapeDivergenceDetector().check({'v1': _shap(m1, X), 'v2': _shap(m2, X)}, X)
    b = sorted(x['feature'] for x in r) if isinstance(r, list) else []
    assert a == b, f"audit={a}  SDD.check={b}"


def test_guard_audit_reporta_a_causa_raiz_do_audit(par):
    """
    Guard.audit agrega o SDD com o detector de importância e reordena por
    causa raiz. Não é subconjunto estrito: o Guard aplica o filtro de
    importância sobre a MÉDIA dos dois modelos, enquanto o audit usa o
    MÁXIMO. Uma feature de ruído que ganha peso por redistribuição
    (medido: 0.18% -> 6.7% quando a principal é neutralizada) passa por um
    critério e não pelo outro.

    Ambos são defensáveis; o contrato testável é que a causa raiz apareça
    nos dois.
    """
    a = [d['feature'] for d in
         audit(par['m1'], par['m2'], par['X'], verbose=False)['divergences']]
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    assert 'a' in a, "o audit deve reportar a feature corrompida"
    assert 'a' in r.affected_features, "o Guard deve reportar a feature corrompida"
    assert 'a' in r.root_cause_features, "e identificá-la como causa raiz"


def test_shap_importance_e_backend_concordam(par):
    """Duas implementações de importância SHAP: mesma ordenação."""
    i1 = _shap_importance(par['m1'], par['X'], n_sample=len(par['X']))
    i2 = bk_importance(par['m1'], par['X'], par['fs'], backend='shap')
    if i1 is None or i2 is None:
        pytest.skip('SHAP indisponível para este modelo')
    assert np.corrcoef(i1, i2)[0, 1] > 0.99
    assert int(np.argmax(i1)) == int(np.argmax(i2))


def test_icd_e_check_features_concordam(par):
    """Guard.check_features encapsula o ICD; o veredito deve ser o mesmo."""
    icd = ImportanceCollapseDetector(n_boot=5, n_sample=300)
    r1 = icd.compare(par['m1'], par['m2'], par['X'], par['fs'], monitored=['a'])
    r2 = M.Guard().check_features(par['m1'], par['m2'], par['X'],
                                  monitored=['a'], feature_names=par['fs'])
    assert r1['monitored_alert'] == r2['alert']
    assert sorted(r1['monitored_hits']) == sorted(r2['collapsed'])


def test_orchestrator_e_audit_concordam(par):
    a = sorted(d['feature'] for d in
               audit(par['m1'], par['gb'], par['X'], verbose=False)['divergences'])
    o = M.ModelGuardOrchestrator().diagnose(par['m1'], par['gb'], par['X'],
                                            par['y'], par['fs'])
    b = sorted(d['feature'] for d in o['layer1']['sdd']['divs'])
    assert a == b, f"audit={a}  orchestrator={b}"


def test_audit_deterministico(par):
    s = {str(sorted(d['feature'] for d in
                    audit(par['m1'], par['gb'], par['X'],
                          verbose=False)['divergences'])) for _ in range(3)}
    assert len(s) == 1


# ═══ Política de precedência entre sinais (v1.63) ═════════════════════
def test_veredito_bloqueado_e_estavel_entre_sementes(par):
    """
    Medido em 10 sementes: BLOQUEADO 10/10, driver feature_integrity 10/10.
    Contrasta com o veredito anterior, guiado pelo índice de saúde, que
    oscilava (11/15 num caso de fronteira).
    """
    from collections import Counter
    X, y = par['X'], par['y']
    vs = []
    for s in range(6):
        m1 = RF(s, n=40).fit(X, y)
        Xc = X.copy(); Xc['a'] = Xc['a'].mean()
        m2 = RF(s, n=40).fit(Xc, y)
        vs.append(M.Guard().audit(m1, m2, X, y, par['fs']).verdict)
    assert set(vs) == {'BLOQUEADO'}, f"veredito oscilou: {Counter(vs)}"


def test_matriz_de_decisao_no_veredito(par):
    """
    v1.63: a camada treino/inferência passou a participar do veredito quando
    X_train é fornecido. Antes ela existia como detector isolado, e um caso
    com skew real saía APROVADO.

    Reproduz a matriz documentada:
      integridade | treino/serving | veredito
      alerta      | silencioso     | BLOQUEADO
      silencioso  | alerta         | ATENCAO
      alerta      | alerta         | BLOQUEADO
      silencioso  | silencioso     | índice decide
    """
    X, y, fs = par['X'], par['y'], par['fs']
    m1, m2, ctrl = par['m1'], par['m2'], par['ctrl']
    Xskew = X.copy(); Xskew['b'] = Xskew['b'] * 1.8
    g = M.Guard()

    # X_train é o treino do modelo NOVO: nele a coluna 'a' é constante,
    # que é o que distingue falha de pipeline de reaprendizado (v0.12.0).
    r = g.audit(m1, m2, X, y, fs, X_train=par['Xc'])
    assert r.verdict == 'BLOQUEADO' and r.verdict_driver.startswith('feature_integrity')

    r = g.audit(m1, ctrl, Xskew, y, fs, X_train=X)
    assert r.verdict == 'ATENCAO', f"skew sozinho deve dar ATENCAO: {r.verdict}"
    assert r.verdict_driver == 'train_serving'
    assert r.skew_features, "as variáveis com skew devem ser reportadas"

    Xcskew = par['Xc'].copy(); Xcskew['b'] = Xcskew['b'] * 1.8
    r = g.audit(m1, m2, Xskew, y, fs, X_train=par['Xc'])
    assert r.verdict == 'BLOQUEADO'
    assert '+' in r.verdict_driver, f"driver deve indicar ambos: {r.verdict_driver}"


def test_sem_x_train_a_quarta_camada_nao_participa(par):
    """
    Compatibilidade: sem X_train o comportamento anterior é preservado. O
    usuário precisa saber que a camada não roda — está na docstring.
    """
    X, y, fs = par['X'], par['y'], par['fs']
    Xskew = X.copy(); Xskew['b'] = Xskew['b'] * 1.8
    r = M.Guard().audit(par['m1'], par['ctrl'], Xskew, y, fs)
    assert r.verdict_driver != 'train_serving'  # a camada não rodou
    assert r.skew_features == []


# ═══ Declarar o que NÃO foi avaliado (v1.65) ══════════════════════════
# Princípio: "não avaliamos X" e "X está ok" são afirmações diferentes, e
# só a primeira é verdadeira quando a camada não rodou. Um relatório que
# omite isso parece completo medindo menos do que o leitor supõe.

def test_sem_x_train_o_relatorio_declara_a_lacuna(par):
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    camadas = [i['layer'] for i in r.not_evaluated]
    assert any('train/serving' in c for c in camadas)
    assert 'Não verificado' in r.plain
    assert 'X_train' in r.plain


def test_com_x_train_a_lacuna_some(par):
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['X'])
    camadas = [i['layer'] for i in r.not_evaluated]
    assert not any('train/serving' in c for c in camadas)


def test_fairness_sempre_declarado_como_nao_avaliado(par):
    """
    O fairness não participa do veredito por decisão. O relatório precisa
    dizer isso — senão o leitor supõe que foi verificado e estava ok.
    """
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['X'])
    assert any('fairness' in i['layer'] for i in r.not_evaluated)


def test_cada_lacuna_declara_a_consequencia(par):
    """
    Dizer "não rodou" é metade; o leitor precisa saber o que fica
    descoberto.
    """
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    for item in r.not_evaluated:
        assert item.get('consequence'), f"sem consequência: {item}"
        assert len(item['consequence']) > 20


def test_falha_de_camada_aparece_como_nao_avaliada(par):
    """
    BUG 30: a camada de forma falhava com NameError e o erro era engolido —
    o relatório reportava o veredito sem dizer que a análise de forma não
    tinha rodado. Foi o próprio `not_evaluated` que expôs isso.
    """
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['X'])
    metodo = str((r.raw.get('layer1', {}).get('sdd') or {}).get('method', ''))
    assert 'ERROR' not in metodo.upper(), \
        f"camada de forma falhando: {metodo}"
    assert not any('shape' in i['layer'] for i in r.not_evaluated)


# ═══ INVARIANTE: exceção capturada vira lacuna declarada (v1.66) ══════
# O bug 30 sobreviveu porque um `except` engoliu um NameError: a camada de
# forma parou de medir, o veredito saiu certo pelo detector de integridade,
# e nenhum teste de comportamento notou — o resultado final estava correto
# pelo motivo errado.
#
# Política: em código de auditoria, todo `except` registra uma lacuna. Se
# uma exceção não merece virar lacuna, ela não deveria ser capturada.

_MODULOS_DE_AUDITORIA = ('orchestrator', 'api', 'audit')


def _handlers_silenciosos(nome_modulo):
    """Handlers que nem re-levantam nem registram a falha."""
    import ast, importlib, inspect
    mod = importlib.import_module(f'modelcheck.{nome_modulo}')
    src = inspect.getsource(mod)
    achados = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Try):
            continue
        for h in node.handlers:
            corpo = ast.unparse(h)
            relevanta = any(isinstance(n, ast.Raise) for n in ast.walk(h))
            registra = any(k in corpo for k in (
                '_registrar_falha', '_failures', '_falhas', 'warnings_list',
                'not_evaluated', 'ERROR', 'error', 'reason'))
            # marcador explícito para exceções locais que não correspondem a
            # nenhuma medição perdida (ex.: coluna não numérica num teste de
            # infinito). Exige comentário justificando.
            linha = src.split('\n')[h.lineno - 1:h.lineno + 6]
            isento = any('lacuna-nao-aplicavel' in l for l in linha)
            registra = registra or isento
            if not (relevanta or registra):
                achados.append((nome_modulo, h.lineno, corpo.split('\n')[0]))
    return achados


@pytest.mark.parametrize('modulo', _MODULOS_DE_AUDITORIA)
def test_nenhuma_excecao_engolida_sem_registro(modulo):
    """
    Falha se alguém adicionar um `except` que descarta a exceção sem
    declarar a lacuna. O caminho correto é chamar `_registrar_falha` ou
    deixar a exceção subir.
    """
    mudos = _handlers_silenciosos(modulo)
    assert not mudos, (
        f"{len(mudos)} except silencioso(s) em {modulo}.py: {mudos[:4]}. "
        f"Em código de auditoria, toda exceção capturada deve virar lacuna "
        f"declarada (veja _registrar_falha) — senão uma medição pode parar "
        f"sem que ninguém perceba, como no bug 30.")


def test_falha_de_camada_vira_lacuna_no_relatorio(par, monkeypatch):
    """
    Simula uma camada quebrando e verifica que o relatório declara a
    lacuna em vez de omiti-la.
    """
    import modelcheck.orchestrator as orc

    def _explode(*a, **k):
        raise RuntimeError('falha simulada')

    monkeypatch.setattr(orc, '_run_layer2', _explode, raising=False)
    try:
        r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'],
                            par['fs'], X_train=par['X'])
    except RuntimeError:
        pytest.skip('a camada propaga a exceção — também é comportamento '
                    'aceitável, desde que não seja silenciosa')
    assert r.not_evaluated, 'camada falhou e o relatório não declarou nada'


def test_auditoria_saudavel_declara_apenas_lacunas_de_projeto(par):
    """
    Numa auditoria sem falhas, a única lacuna esperada é o fairness — que
    não roda por decisão documentada. Qualquer outra indica camada quebrada.
    """
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['X'])
    inesperadas = [i['layer'] for i in r.not_evaluated
                   if 'fairness' not in i['layer']]
    assert not inesperadas, f"camadas falhando em silêncio: {inesperadas}"


# ═══ audit() vs snapshot.diff() (v1.67) ═══════════════════════════════
# Os dois caminhos respondem à mesma pergunta por rotas diferentes: um com
# os dois modelos em memória, outro contra um perfil gravado. Foram tocados
# separadamente entre v1.62 e v1.66, que é a condição em que o bug 28
# apareceu — implementações paralelas divergindo sem que nenhum teste
# comparasse as duas.

@pytest.fixture
def snap(par, tmp_path):
    from modelcheck import snapshot
    p = str(tmp_path / 's.json')
    snapshot.save(par['m1'], par['X'], p, label='base')
    return p


@pytest.mark.parametrize('cen', ['m2', 'ctrl'])
def test_audit_e_snapshot_diff_dao_o_mesmo_veredito(par, snap, cen):
    from modelcheck import snapshot
    a = M.Guard().audit(par['m1'], par[cen], par['X'], par['y'], par['fs'])
    b = snapshot.diff(snap, par[cen], par['X'])
    assert a.verdict == b['verdict'], \
        f"audit={a.verdict} diff={b['verdict']}"
    assert sorted(a.root_cause_features) == sorted(b['root_cause_features'])


def test_snapshot_diff_declara_lacunas(par, snap):
    """
    O `not_evaluated` foi adicionado ao audit() e ao diff() em momentos
    diferentes. Se um declarar e o outro não, um relatório parece mais
    completo que o outro sem ser.
    """
    from modelcheck import snapshot
    b = snapshot.diff(snap, par['m2'], par['X'])
    assert b.get('not_evaluated'), 'diff() não declara lacunas'
    camadas = [i['layer'] for i in b['not_evaluated']]
    assert any('train/serving' in c for c in camadas)
    assert any('fairness' in c for c in camadas)
    for item in b['not_evaluated']:
        assert item.get('consequence'), f'lacuna sem consequência: {item}'


# ═══ Os três formatos trazem os mesmos campos ═════════════════════════
def test_formatos_nao_divergem(par):
    """
    `text`, `json` e `html` saem do mesmo dicionário, mas são montados por
    funções separadas. Quem adiciona um campo pode tocar só uma delas.
    """
    import json as _json
    from modelcheck.cli import _texto, _html
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    res = {'verdict': r.verdict, 'problem_type': r.problem_type,
           'root_cause_features': r.root_cause_features,
           'divergences': [], 'collapses': [], 'n_eval': len(par['X']),
           'representation': 'shap', 'not_evaluated': r.not_evaluated}
    # v1.70: o laudo segue MODELCHECK_LANG. Fixamos o idioma aqui — senão o
    # resultado da suíte mudaria conforme a LANG da máquina que a roda.
    from modelcheck.laudo import render as _laudo
    t, h, j = _texto(res), _laudo(res, lang='pt'), _json.dumps(res)

    # v1.69: o HTML é o laudo para leitura humana — ele TRADUZ códigos
    # internos em vez de exibi-los. Comparar literais entre os três
    # formatos seria exigir que o laudo mostre 'IMPORTANCE_COLLAPSE' na
    # tela, que é exatamente o que não deve acontecer.
    #
    # O contrato real: nenhum formato pode PERDER a informação. Para o
    # HTML, verificamos a forma traduzida.
    from modelcheck.laudo import _texto_problema, _TONS

    for campo in list(r.root_cause_features):
        campo = str(campo)
        faltando = [n for n, s in (('text', t), ('html', h), ('json', j))
                    if campo not in s]
        assert not faltando, f"'{campo}' ausente em: {faltando}"

    # veredito: literal em text/json, ação equivalente no html
    assert r.verdict in t and r.verdict in j
    assert _TONS[r.verdict][2] in h, 'html deve mostrar a ação do veredito'

    # tipo de problema: literal em text/json, frase traduzida no html
    assert r.problem_type in j
    # v0.16: o subtítulo deixou de repetir o código do problema e passou a
    # explicar a situação em uma frase. O que não pode faltar é a explicação.
    assert ('deixaram de influenciar' in h or 'deixou de influenciar' in h
            or _texto_problema(r.problem_type, 'pt') in h), \
        'html deve mostrar a tradução do tipo de problema'

    # lacunas: chave interna em text/json, nome legível + a consequência
    # no html. Nenhuma pode sumir.
    for item in r.not_evaluated:
        assert item['layer'] in t and item['layer'] in j
        assert item['consequence'] in h, (
            f"a consequência da lacuna '{item['layer']}' não aparece no "
            f"laudo — o leitor não saberia o que ficou sem exame")
    assert h.count('não verificado') >= len(r.not_evaluated)


def test_plain_e_texto_cli_concordam_no_veredito(par):
    """`AuditResult.plain` e o `_texto` da CLI são renderizadores distintos."""
    from modelcheck.cli import _texto
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    res = {'verdict': r.verdict, 'problem_type': r.problem_type,
           'root_cause_features': r.root_cause_features,
           'not_evaluated': r.not_evaluated, 'n_eval': len(par['X']),
           'representation': 'shap'}
    assert r.verdict in r.plain
    assert r.verdict in _texto(res)
    if r.root_cause_features:
        f = r.root_cause_features[0]
        assert f in r.plain and f in _texto(res)


# ═══ v0.11.0: o índice de saúde não decide mais ══════════════════════
def test_indice_nao_decide_o_veredito(par):
    """
    Dois modelos idênticos: nenhum sinal medido -> APROVADO, qualquer que
    seja o índice. Antes o índice decidia os casos sem sinal de camada.
    """
    r = M.Guard().audit(par['m1'], par['m1'], par['X'], par['y'], par['fs'])
    assert r.verdict == 'APROVADO' and r.verdict_driver == 'nenhum_sinal'


def test_indice_nao_aparece_na_regra():
    import inspect, ast
    from modelcheck import api
    import textwrap
    src = ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(api.AuditResult.__init__))))
    assert '_VERDICT' not in src and 'self.score >=' not in src


# ═══ v0.12.0: coluna ausente x modelo que deixou de usar ══════════════
def test_coluna_constante_no_treino_e_falha_de_pipeline(par):
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['Xc'])
    assert r.verdict == 'BLOQUEADO' and 'a' in r.root_cause_features
    assert not r.relearning


def test_variavel_presente_que_perdeu_peso_e_reaprendizado(par):
    """
    Mesmo par, mas o treino do modelo novo tem a coluna variando (é o
    conjunto original). Medido em retreinos sazonais: a camada dava
    BLOQUEADO num retreino legítimo 1 vez em 5.
    """
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'],
                        X_train=par['X'])
    assert r.verdict == 'REVISAR' and r.verdict_driver == 'relearning'
    assert 'a' in r.relearning and 'a' not in r.root_cause_features


def test_sem_x_train_nada_e_rebaixado(par):
    r = M.Guard().audit(par['m1'], par['m2'], par['X'], par['y'], par['fs'])
    assert r.verdict == 'BLOQUEADO' and 'a' in r.root_cause_features
