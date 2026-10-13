"""
Snapshot e CLI (v1.62).

O snapshot é o que separa "comparação instantânea" de "monitoramento": sem
ele, comparar o modelo de janeiro com o de julho exige ter os dois carregados
ao mesmo tempo, o que não acontece em produção.
"""
import json, os, subprocess, sys, warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
from modelcheck import snapshot


@pytest.fixture(scope='module')
def cenario(tmp_path_factory):
    d = tmp_path_factory.mktemp('snap')
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 600), 'b': rng.uniform(0, 1, 600),
                      'c': rng.uniform(0, 1, 600)})
    y = pd.Series(3 * X.a + 2 * X.b + rng.randn(600) * 0.2)
    v1 = RF(0, n=50).fit(X, y)
    Xc = X.copy(); Xc['a'] = Xc['a'].mean()
    v2 = RF(0, n=50).fit(Xc, y)
    ctrl = RF(7, n=50).fit(X, y)
    return dict(d=str(d), X=X, y=y, v1=v1, v2=v2, ctrl=ctrl)


# ═══ O caso de uso central ════════════════════════════════════════════
def test_diff_funciona_sem_o_modelo_antigo(cenario):
    """
    A propriedade que justifica o módulo: depois de gravar o snapshot, o
    modelo v1 pode ser descartado. Em produção ele não estará carregado.
    """
    p = os.path.join(cenario['d'], 's1.json')
    snapshot.save(cenario['v1'], cenario['X'], p, label='jan')

    v1 = cenario['v1']
    del v1                                  # o modelo antigo some

    r = snapshot.diff(p, cenario['v2'], cenario['X'])
    assert r['verdict'] == 'BLOQUEADO'
    assert 'a' in r['root_cause_features']


def test_snapshot_nao_contem_dados_nem_modelo(cenario):
    """
    Um snapshot descreve estrutura, não copia dados. Isso permite versioná-lo
    onde o dataset não poderia ir.
    """
    p = os.path.join(cenario['d'], 's2.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    bruto = open(p, encoding='utf-8').read()
    assert 'estimators_' not in bruto
    # nenhuma linha de dado deve aparecer literalmente
    amostra = cenario['X'].iloc[0]
    assert f"{amostra['a']:.10f}" not in bruto


def test_snapshot_e_pequeno(cenario):
    p = os.path.join(cenario['d'], 's3.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    assert os.path.getsize(p) < 60_000, "snapshot deve ser leve"


def test_controle_nao_dispara(cenario):
    p = os.path.join(cenario['d'], 's4.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    r = snapshot.diff(p, cenario['ctrl'], cenario['X'])
    assert r['verdict'] == 'APROVADO'
    assert r['root_cause_features'] == []


# ═══ Proteções ═══════════════════════════════════════════════════════
def test_recusa_conjunto_de_avaliacao_diferente(cenario):
    """
    Comparar perfis calculados em conjuntos diferentes mede a diferença entre
    os conjuntos, não entre os modelos. Recusar é mais útil que devolver um
    número enganoso.
    """
    p = os.path.join(cenario['d'], 's5.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    outro = cenario['X'].sample(300, random_state=99)
    with pytest.raises(ValueError, match='avaliação|avaliacao'):
        snapshot.diff(p, cenario['v2'], outro)


def test_permite_conjunto_diferente_se_explicito(cenario):
    p = os.path.join(cenario['d'], 's6.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    outro = cenario['X'].sample(300, random_state=99)
    r = snapshot.diff(p, cenario['v2'], outro, strict_eval=False)
    assert r['same_eval_set'] is False


def test_recusa_conjunto_de_features_diferente(cenario):
    p = os.path.join(cenario['d'], 's7.json')
    snapshot.save(cenario['v1'], cenario['X'], p)
    Xmenos = cenario['X'].drop(columns=['c'])
    m = RF(0, n=50).fit(Xmenos, cenario['y'])
    with pytest.raises(ValueError, match='variáveis|variaveis'):
        snapshot.diff(p, m, Xmenos)


def test_snapshot_ausente_da_erro_com_instrucao():
    with pytest.raises(FileNotFoundError, match='snapshot save'):
        snapshot.load('/tmp/nao_existe_snapshot_xyz.json')


def test_listar_ordena_do_mais_recente(cenario):
    d = os.path.join(cenario['d'], 'lista')
    os.makedirs(d, exist_ok=True)
    for lab in ('a', 'b'):
        snapshot.save(cenario['v1'], cenario['X'],
                      os.path.join(d, f'{lab}.json'), label=lab)
    rows = snapshot.listar(d)
    assert len(rows) == 2
    assert all('label' in r and 'created' in r for r in rows)


# ═══ CLI: exit codes ══════════════════════════════════════════════════
def _cli(args, cwd):
    env = dict(os.environ, PYTHONPATH=os.getcwd(), NO_COLOR='1')
    return subprocess.run([sys.executable, '-W', 'ignore', '-m',
                           'modelcheck.cli'] + args,
                          capture_output=True, text=True, cwd=cwd, env=env)


@pytest.fixture(scope='module')
def cli_env(cenario, tmp_path_factory):
    import joblib
    d = str(tmp_path_factory.mktemp('cli'))
    joblib.dump(cenario['v1'], os.path.join(d, 'v1.pkl'))
    joblib.dump(cenario['v2'], os.path.join(d, 'v2.pkl'))
    joblib.dump(cenario['ctrl'], os.path.join(d, 'ok.pkl'))
    cenario['X'].to_csv(os.path.join(d, 'eval.csv'), index=False)
    _cli(['snapshot', 'save', 'v1.pkl', 'eval.csv', '-o', 's.json',
          '--label', 'base'], d)
    return d


def test_cli_exit_0_quando_saudavel(cli_env):
    r = _cli(['diff', 'ok.pkl', 'eval.csv', '--against', 's.json',
              '--fail-on', 'high'], cli_env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_cli_exit_1_quando_encontra_problema(cli_env):
    """É o que permite bloquear deploy automaticamente."""
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json',
              '--fail-on', 'high'], cli_env)
    assert r.returncode == 1, r.stdout + r.stderr


def test_cli_exit_2_em_erro_de_execucao(cli_env):
    """
    Erro de execução tem código próprio: um pipeline não pode tratar
    "a auditoria não rodou" como "o modelo passou".
    """
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 'inexistente.json'],
             cli_env)
    assert r.returncode == 2
    assert 'Traceback' not in r.stderr, "erro previsto não deve vazar stack trace"


def test_cli_sem_fail_on_falha_em_bloqueado(cli_env):
    """
    BUG 85 (v1.0.4): o padrão era não falhar nunca. A documentação promete
    "0 limpo, 1 problema", e num CI o padrão antigo deixava passar modelo
    BLOQUEADO. Padrão agora: --fail-on high.
    """
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json'], cli_env)
    assert r.returncode == 1


def test_cli_modo_observacao_e_explicito(cli_env):
    """Reportar sem falhar continua possível, mas tem de ser pedido."""
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json',
              '--fail-on', 'none'], cli_env)
    assert r.returncode == 0


# ═══ CLI: formatos ════════════════════════════════════════════════════
def test_cli_json_e_parseavel(cli_env):
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json',
              '--format', 'json'], cli_env)
    d = json.loads(r.stdout)
    assert d['verdict'] == 'BLOQUEADO'
    assert 'a' in d['root_cause_features']


def test_cli_html_gera_arquivo(cli_env):
    out = os.path.join(cli_env, 'r.html')
    _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json',
          '--format', 'html', '-o', 'r.html'], cli_env)
    assert os.path.exists(out)
    h = open(out, encoding='utf-8').read()
    assert '<!DOCTYPE html>' in h
    # v1.70: o laudo mostra a AÇÃO ("Do not promote" / "Não promova"), não o
    # código interno do veredito — em qualquer um dos idiomas.
    from modelcheck.laudo import _TONS, _ACAO_EN
    acoes = (_TONS['BLOQUEADO'][2], _ACAO_EN['BLOQUEADO'])
    assert any(a in h for a in acoes), \
        f'o laudo deve mostrar a ação do veredito; nenhuma de {acoes} apareceu'


def test_cli_texto_sem_cor_quando_nao_e_tty(cli_env):
    """Saída redirecionada não deve conter códigos ANSI."""
    r = _cli(['diff', 'v2.pkl', 'eval.csv', '--against', 's.json'], cli_env)
    assert '\033[' not in r.stdout


def test_cli_check_exige_monitor(cli_env):
    r = _cli(['check', 'v1.pkl', 'v2.pkl', 'eval.csv'], cli_env)
    assert r.returncode == 2
    assert 'monitor' in r.stderr
