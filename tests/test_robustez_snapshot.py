"""
Robustez do snapshot e da cadeia de migração (v1.69).

Os cinco cenários priorizados: camadas diferentes entre auditorias, falha no
meio da migração, arquivo corrompido, bordas do histórico, determinismo.
"""
import json, os, warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
from modelcheck import snapshot
from modelcheck.snapshot import SCHEMA_VERSION, normalizar


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    y = pd.Series(3 * X.a + rng.randn(400) * .2)
    m = RF(0, n=40).fit(X, y)
    d = tmp_path_factory.mktemp('rob')
    p = str(d / 'base.json')
    snapshot.save(m, X, p, label='base')
    return dict(X=X, y=y, m=m, path=p, raw=json.load(open(p)), dir=str(d))


# ═══ 1. Auditorias que verificaram coisas diferentes ══════════════════
def test_auditorias_com_camadas_diferentes_sao_distinguiveis(base, tmp_path):
    """
    O caso central do histórico: uma auditoria de janeiro sem a camada de
    dados e uma de julho com ela. O registro precisa distinguir "não
    alertou" de "não verificou" — senão o histórico sugere que janeiro
    estava limpo quando janeiro nem olhou.
    """
    import modelcheck as M
    m1, m2 = base['m'], RF(7, n=40).fit(base['X'], base['y'])
    X, y = base['X'], base['y']
    fs = list(X.columns)

    antiga = M.Guard().audit(m1, m2, X, y, fs)                 # sem X_train
    nova = M.Guard().audit(m1, m2, X, y, fs, X_train=X)        # com

    lac_antiga = [i['layer'] for i in antiga.not_evaluated]
    lac_nova = [i['layer'] for i in nova.not_evaluated]

    assert any('train/serving' in c for c in lac_antiga), \
        'a auditoria antiga deve declarar que não verificou a camada'
    assert not any('train/serving' in c for c in lac_nova), \
        'a auditoria nova verificou — não deve declarar lacuna'
    assert lac_antiga != lac_nova, (
        'as duas auditorias cobriram coisas diferentes e o registro precisa '
        'refletir isso')


def test_lacuna_distingue_nao_alertou_de_nao_verificou(base):
    """
    Ambas as auditorias podem dar APROVADO. A diferença está no que foi
    coberto, e só o `not_evaluated` carrega essa informação.
    """
    import modelcheck as M
    m1, ok = base['m'], RF(7, n=40).fit(base['X'], base['y'])
    X, y, fs = base['X'], base['y'], list(base['X'].columns)
    sem = M.Guard().audit(m1, ok, X, y, fs)
    com = M.Guard().audit(m1, ok, X, y, fs, X_train=X)
    assert sem.verdict == com.verdict == 'APROVADO'
    assert len(sem.not_evaluated) > len(com.not_evaluated), (
        'mesmo veredito, coberturas diferentes — o histórico precisa poder '
        'mostrar isso')


# ═══ 2. Falha no meio da cadeia de migração ═══════════════════════════
def test_falha_no_meio_da_cadeia_aponta_a_migracao(base, monkeypatch):
    """
    v1 → v2 → v3 com a segunda migração quebrando. A mensagem deve dizer
    QUAL migração falhou — "erro ao migrar" sem o número obriga o mantenedor
    a bissectar à mão.
    """
    import modelcheck.snapshot as sn

    def _ok(d):
        d = dict(d); d['schema_version'] = 2; d['campo_v2'] = 1
        return d

    def _quebra(d):
        raise RuntimeError('estrutura inesperada no campo `bins`')

    monkeypatch.setattr(sn, 'SCHEMA_VERSION', 3)
    monkeypatch.setattr(sn, '_MIGRACOES', {1: _ok, 2: _quebra})

    with pytest.raises(ValueError) as e:
        sn.normalizar(dict(base['raw']))
    msg = str(e.value)
    assert 'v2' in msg and 'v3' in msg, \
        f'a mensagem deve identificar a migração que falhou: {msg}'
    assert 'estrutura inesperada' in msg, \
        'a causa original deve ser preservada'


def test_migracao_parcial_nao_devolve_snapshot_meio_migrado(base, monkeypatch):
    """
    Se a cadeia falha no meio, não pode devolver o snapshot parcialmente
    convertido — seria um formato que não existe em versão nenhuma.
    """
    import modelcheck.snapshot as sn

    def _ok(d):
        d = dict(d); d['schema_version'] = 2
        return d

    def _quebra(d):
        raise RuntimeError('falha')

    monkeypatch.setattr(sn, 'SCHEMA_VERSION', 3)
    monkeypatch.setattr(sn, '_MIGRACOES', {1: _ok, 2: _quebra})
    with pytest.raises(ValueError):
        sn.normalizar(dict(base['raw']))


# ═══ 3. Arquivo corrompido ════════════════════════════════════════════
def test_json_truncado_da_mensagem_clara(base, tmp_path):
    """Arquivo cortado no meio — erro legível, não JSONDecodeError cru."""
    bruto = open(base['path']).read()
    p = str(tmp_path / 'trunc.json')
    open(p, 'w').write(bruto[:len(bruto) // 2])
    with pytest.raises(ValueError) as e:
        snapshot.load(p)
    assert 'corrompido' in str(e.value).lower() or 'inválido' in str(e.value).lower()


def test_snapshot_vazio_e_recusado(tmp_path):
    p = str(tmp_path / 'vazio.json')
    json.dump({}, open(p, 'w'))
    with pytest.raises(ValueError, match='sem versão|corrompido'):
        snapshot.load(p)


def test_schema_version_nao_numerica_e_recusada(base, tmp_path):
    ruim = dict(base['raw'], schema_version='abc')
    p = str(tmp_path / 'ruim.json')
    json.dump(ruim, open(p, 'w'))
    with pytest.raises(ValueError, match='inválida'):
        snapshot.load(p)


def test_arquivo_que_nao_e_json(tmp_path):
    p = str(tmp_path / 'naojson.json')
    open(p, 'w').write('isto nao e json {{{')
    with pytest.raises(ValueError):
        snapshot.load(p)


# ═══ 4. Campos desconhecidos aninhados ════════════════════════════════
def test_campo_desconhecido_aninhado_e_preservado(base, tmp_path):
    """
    `aditivo é seguro` precisa valer também dentro de subestruturas — uma
    versão futura pode acrescentar campo em `meta` sem mudar o esquema.
    """
    novo = json.loads(json.dumps(base['raw']))
    novo['meta']['campo_futuro'] = {'detalhe': [1, 2, 3]}
    p = str(tmp_path / 'aninhado.json')
    json.dump(novo, open(p, 'w'))
    s = snapshot.load(p)
    assert s['meta']['campo_futuro'] == {'detalhe': [1, 2, 3]}


# ═══ 5. Determinismo ══════════════════════════════════════════════════
def test_snapshot_do_mesmo_modelo_tem_conteudo_estavel(base, tmp_path):
    """
    Gravar duas vezes o mesmo modelo e dados deve dar o mesmo conteúdo,
    exceto o timestamp. Sem isso, um histórico mostraria "mudanças" que são
    ruído de serialização.
    """
    p2 = str(tmp_path / 'again.json')
    snapshot.save(base['m'], base['X'], p2, label='base')
    a, b = dict(base['raw']), json.load(open(p2))
    for d in (a, b):
        d.pop('created_utc', None)
    assert a['shapes'] == b['shapes']
    assert a['importance'] == b['importance']
    assert a['meta']['eval_fingerprint'] == b['meta']['eval_fingerprint']


def test_ordem_das_colunas_nao_muda_o_fingerprint(base, tmp_path):
    """
    O fingerprint identifica o conjunto de avaliação. Reordenar colunas não
    muda o conjunto — se mudasse, o diff recusaria comparações válidas.
    """
    p = str(tmp_path / 'reord.json')
    Xr = base['X'][list(reversed(list(base['X'].columns)))]
    snapshot.save(base['m'], Xr, p, label='reord',
                  feature_names=list(base['X'].columns))
    novo = json.load(open(p))
    assert novo['meta']['eval_fingerprint'] == \
        base['raw']['meta']['eval_fingerprint']
