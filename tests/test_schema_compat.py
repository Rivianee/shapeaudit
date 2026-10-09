"""
Compatibilidade do esquema de snapshot (v1.68).

POR QUE ISTO EXISTE
-------------------
Um histórico de auditoria atravessa versões da ferramenta. Um snapshot
gravado em janeiro precisa abrir em julho — recusá-lo perde o histórico
exatamente quando ele é necessário.

Antes da v1.68, `load()` recusava qualquer `snapshot_version` diferente da
corrente. Isso teria quebrado o `report --last N` na primeira mudança de
formato: o comando que existe para olhar o passado não conseguiria lê-lo.

A REGRA
-------
    aditivo é seguro    campo novo é opcional, esquema não muda
    renomear            alias por uma versão, depois deprecia
    remover/quebrar     SCHEMA_VERSION nova + migração registrada
"""
import json, os, warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from conftest import RF
from modelcheck import snapshot
from modelcheck.snapshot import SCHEMA_VERSION, CANONICO, normalizar


@pytest.fixture(scope='module')
def base(tmp_path_factory):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({'a': rng.uniform(0, 1, 400), 'b': rng.uniform(0, 1, 400)})
    y = pd.Series(3 * X.a + rng.randn(400) * .2)
    m = RF(0, n=40).fit(X, y)
    p = str(tmp_path_factory.mktemp('sc') / 'base.json')
    snapshot.save(m, X, p, label='base')
    return dict(X=X, y=y, m=m, path=p, raw=json.load(open(p)))


# ═══ O contrato ═══════════════════════════════════════════════════════
def test_schema_version_e_explicito(base):
    """Versão em campo visível, não inferida por heurística de conteúdo."""
    assert base['raw']['schema_version'] == SCHEMA_VERSION
    assert isinstance(base['raw']['schema_version'], int)


def test_snapshot_da_versao_anterior_abre(base, tmp_path):
    """
    Snapshots gravados até a v1.67 só têm `snapshot_version`. Precisam abrir
    sem migração manual — é o caso que o `report` histórico vai encontrar.
    """
    antigo = dict(base['raw'])
    del antigo['schema_version']            # como era até a v1.67
    p = str(tmp_path / 'v167.json')
    json.dump(antigo, open(p, 'w'))
    s = snapshot.load(p)
    assert s['schema_version'] == SCHEMA_VERSION
    assert s['features'] == base['raw']['features']


def test_campo_novo_nao_quebra_leitura(base, tmp_path):
    """`aditivo é seguro`: um campo desconhecido não invalida o snapshot."""
    novo = dict(base['raw'], campo_que_ainda_nao_existe={'x': 1})
    p = str(tmp_path / 'aditivo.json')
    json.dump(novo, open(p, 'w'))
    s = snapshot.load(p)
    assert s['campo_que_ainda_nao_existe'] == {'x': 1}, \
        'campo desconhecido deve ser preservado, não descartado'


def test_esquema_futuro_e_recusado_com_instrucao(base, tmp_path):
    """
    Ler um formato que ainda não existe exigiria adivinhar o significado
    dos campos. Recusar com instrução é mais honesto que tentar.
    """
    fut = dict(base['raw'], schema_version=SCHEMA_VERSION + 50)
    p = str(tmp_path / 'futuro.json')
    json.dump(fut, open(p, 'w'))
    with pytest.raises(ValueError, match='Atualize'):
        snapshot.load(p)


def test_sem_versao_da_erro_claro(tmp_path):
    p = str(tmp_path / 'lixo.json')
    json.dump({'features': ['a']}, open(p, 'w'))
    with pytest.raises(ValueError, match='sem versão'):
        snapshot.load(p)


def test_campos_canonicos_presentes(base):
    for k in CANONICO:
        assert k in base['raw'], f'campo canônico ausente: {k}'


# ═══ A cadeia de migração ═════════════════════════════════════════════
def test_cadeia_de_migracao_funciona(base, monkeypatch):
    """
    Simula SCHEMA_VERSION=2 com uma migração registrada, e verifica que um
    snapshot v1 é convertido automaticamente.

    Este teste é o que garante que a política não é só documentação: se a
    máquina de migração não funcionar, ela falha aqui e não em produção.
    """
    import modelcheck.snapshot as sn

    def _v1_para_v2(d):
        d = dict(d)
        d['campo_adicionado_na_v2'] = 'valor padrão'
        d['schema_version'] = 2
        return d

    monkeypatch.setattr(sn, 'SCHEMA_VERSION', 2)
    monkeypatch.setitem(sn._MIGRACOES, 1, _v1_para_v2)

    s = sn.normalizar(dict(base['raw']))
    assert s['schema_version'] == 2
    assert s['campo_adicionado_na_v2'] == 'valor padrão'
    assert s['features'] == base['raw']['features'], \
        'a migração não pode perder dados existentes'


def test_migracao_ausente_e_defeito_explicito(base, monkeypatch):
    """
    Se alguém subir SCHEMA_VERSION sem registrar a migração, o erro precisa
    dizer que é defeito do código — não sugerir que o usuário regrave.
    """
    import modelcheck.snapshot as sn
    monkeypatch.setattr(sn, 'SCHEMA_VERSION', 2)
    monkeypatch.setattr(sn, '_MIGRACOES', {})
    with pytest.raises(ValueError, match='defeito'):
        sn.normalizar(dict(base['raw']))


# ═══ O que o `report` vai precisar ════════════════════════════════════
def test_snapshots_de_versoes_diferentes_sao_comparaveis(base, tmp_path):
    """
    O caso de uso do `report --last N`: comparar snapshots gravados em
    momentos diferentes, possivelmente com formatos diferentes. Depois de
    normalizar, os dois expõem o mesmo esquema.
    """
    antigo = dict(base['raw'])
    del antigo['schema_version']
    p1 = str(tmp_path / 'a.json'); json.dump(antigo, open(p1, 'w'))
    p2 = base['path']

    a, b = snapshot.load(p1), snapshot.load(p2)
    assert set(CANONICO).issubset(a) and set(CANONICO).issubset(b)
    assert a['schema_version'] == b['schema_version']
    assert set(a['shapes']) == set(b['shapes']), \
        'snapshots normalizados devem expor as mesmas variáveis'


def test_diff_aceita_snapshot_de_versao_anterior(base, tmp_path):
    """A comparação em si precisa funcionar, não só a leitura."""
    antigo = dict(base['raw'])
    del antigo['schema_version']
    p = str(tmp_path / 'ant.json'); json.dump(antigo, open(p, 'w'))
    r = snapshot.diff(p, base['m'], base['X'])
    assert r['verdict'] in ('APROVADO', 'ATENCAO', 'BLOQUEADO')
