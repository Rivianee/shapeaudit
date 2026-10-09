"""
Condições que invalidam ou enfraquecem a comparação (v1.71).

ORIGEM
------
O caso da imputação: duas estratégias diferentes (-999 vs mediana) faziam o
R² cair de 0.897 para 0.334 na avaliação cruzada, e o veredito era APROVADO.
Não era defeito de detecção — era a ferramenta produzindo um veredito num
caso onde o veredito não significa o que o leitor supõe.

A varredura por essa classe encontrou mais três. Este arquivo trava as
guardas.

A REGRA
-------
    recusa   a comparação não é válida — nenhum número salva
    aviso    é válida mas frágil; o resultado sai com a ressalva
"""
import warnings
import numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from conftest import RF
import modelcheck as M
from modelcheck._validate import (checar_comparabilidade, natureza_da_saida,
                                  N_MINIMO_CONFIAVEL)


@pytest.fixture(scope='module')
def dados():
    rng = np.random.RandomState(0)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, 700) for i in range(4)})
    y = pd.Series(3 * X.x0 + 2 * X.x1 + rng.randn(700) * .2)
    return X, y


# ═══ RECUSA: naturezas de saída diferentes ════════════════════════════
def test_regressor_vs_classificador_e_recusado(dados):
    """
    Um regressor prevê valores contínuos e um classificador prevê classes.
    As atribuições ficam em unidades distintas, e qualquer veredito seria
    sobre a diferença de unidade — não sobre o modelo.

    Antes da v1.71 isto devolvia ATENCAO.
    """
    X, y = dados
    reg = RF(0, n=40).fit(X, y)
    clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0,
                                 n_jobs=1).fit(X, (y > y.median()).astype(int))
    with pytest.raises(ValueError, match='naturezas diferentes'):
        M.Guard().audit(reg, clf, X, y, list(X.columns))


def test_mensagem_de_recusa_diz_o_que_fazer(dados):
    X, y = dados
    reg = RF(0, n=40).fit(X, y)
    clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0,
                                 n_jobs=1).fit(X, (y > y.median()).astype(int))
    with pytest.raises(ValueError) as e:
        M.Guard().audit(reg, clf, X, y, list(X.columns))
    assert 'mesmo tipo' in str(e.value)


def test_natureza_da_saida_classifica_corretamente(dados):
    X, y = dados
    reg = RF(0, n=40).fit(X, y)
    clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=0,
                                 n_jobs=1).fit(X, (y > y.median()).astype(int))
    assert natureza_da_saida(reg, X) == 'continuo'
    assert natureza_da_saida(clf, X) == 'classe'


def test_dois_regressores_nao_sao_recusados(dados):
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y), X, y,
                        list(X.columns))
    assert r.verdict in ('APROVADO', 'ATENCAO', 'BLOQUEADO')


def test_dois_classificadores_nao_sao_recusados(dados):
    X, y = dados
    yc = (y > y.median()).astype(int)
    mk = lambda s: RandomForestClassifier(n_estimators=40, max_depth=6,
                                          random_state=s, n_jobs=1).fit(X, yc)
    r = M.Guard().audit(mk(0), mk(7), X, yc, list(X.columns))
    assert r.verdict in ('APROVADO', 'ATENCAO', 'BLOQUEADO')


# ═══ AVISO: conjunto de avaliação pequeno ═════════════════════════════
def test_n_pequeno_gera_ressalva(dados):
    """
    Com 15 amostras a classificação de forma oscila. O veredito sai, mas
    com a ressalva — antes saía APROVADO sem nenhuma qualificação.
    """
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y),
                        X.head(15), y.head(15), list(X.columns))
    av = [i for i in r.not_evaluated if 'evaluation size' in i['layer']]
    assert av, 'avaliação com 15 amostras deve gerar ressalva'
    assert av[0].get('severity') == 'alta'
    assert '15' in av[0]['reason']


def test_n_suficiente_nao_gera_ressalva(dados):
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y), X, y,
                        list(X.columns))
    assert not [i for i in r.not_evaluated if 'evaluation size' in i['layer']]


@pytest.mark.parametrize('n', [5, 20, 59])
def test_limiar_de_tamanho(dados, n):
    X, y = dados
    res = checar_comparabilidade(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y),
                                 X.head(n))
    assert any('evaluation size' in r['layer'] for r in res)
    assert n < N_MINIMO_CONFIAVEL


# ═══ A ressalva chega ao relatório ════════════════════════════════════
def test_ressalva_aparece_no_texto(dados):
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y),
                        X.head(15), y.head(15), list(X.columns))
    assert 'Não verificado' in r.plain or 'amostras' in r.plain


def test_toda_ressalva_tem_consequencia(dados):
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y),
                        X.head(15), y.head(15), list(X.columns))
    for i in r.not_evaluated:
        assert i.get('consequence'), f'ressalva sem consequência: {i}'


# ═══ Sinal de imputação eleva o peso do aviso ═════════════════════════
def test_sinal_de_imputacao_eleva_severidade():
    """
    Medido: com -999 numa versão e mediana na outra, o R² cai de 0.897 para
    0.334 na avaliação cruzada e o veredito sem X_train era APROVADO.
    """
    rng = np.random.RandomState(0)
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, 800) for i in range(4)})
    y = pd.Series(2 * X.x0 + rng.randn(800) * .3)
    Xa = X.copy(); Xa.loc[Xa.index[:120], 'x0'] = -999
    m1 = RF(0, n=40).fit(Xa, y); m2 = RF(7, n=40).fit(Xa, y)
    r = M.Guard().audit(m1, m2, Xa, y, list(X.columns))
    ts = [i for i in r.not_evaluated if 'train/serving' in i['layer']]
    assert ts and ts[0].get('severity') == 'alta', \
        'com sinal de imputação, a lacuna de X_train deve ter peso alto'
    assert 'imputação' in ts[0]['reason']


def test_sem_sinal_de_imputacao_severidade_normal(dados):
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y), X, y,
                        list(X.columns))
    ts = [i for i in r.not_evaluated if 'train/serving' in i['layer']]
    assert ts and ts[0].get('severity') != 'alta'


# ═══ As guardas valem também no snapshot.diff (v1.72) ═════════════════
def test_diff_declara_ressalva_de_tamanho(dados, tmp_path):
    """
    As guardas nasceram no Guard.audit. O diff é um caminho paralelo, e
    caminhos paralelos divergem (bug 28). Um diff com 15 amostras não
    avisava.
    """
    from modelcheck import snapshot
    X, y = dados
    Xe = X.head(15)
    p = str(tmp_path / 's.json')
    snapshot.save(RF(0, n=40).fit(X, y), Xe, p, label='b')
    r = snapshot.diff(p, RF(7, n=40).fit(X, y), Xe)
    assert any('evaluation size' in i['layer'] for i in r['not_evaluated'])


def test_diff_nao_inventa_ressalva_com_n_suficiente(dados, tmp_path):
    from modelcheck import snapshot
    X, y = dados
    p = str(tmp_path / 's2.json')
    snapshot.save(RF(0, n=40).fit(X, y), X, p, label='b')
    r = snapshot.diff(p, RF(7, n=40).fit(X, y), X)
    assert not any('evaluation size' in i['layer'] for i in r['not_evaluated'])


def test_snapshot_com_amostra_pequena_nao_quebra(dados, tmp_path):
    """
    BUG 31: `_perfis` assumia a chave "means", que o perfil de amostra
    pequena não devolve — KeyError com n < 30. Mesma família do bug 16.
    """
    from modelcheck import snapshot
    X, y = dados
    p = str(tmp_path / 's3.json')
    s = snapshot.save(RF(0, n=40).fit(X, y), X.head(15), p, label='pequeno')
    assert s['features']
    assert snapshot.load(p)['schema_version']


# ═══ Interação entre guardas ══════════════════════════════════════════
def test_ressalvas_sobrevivem_empilhadas(dados):
    """
    Cada guarda foi testada isolada. Empilhar duas não pode fazer uma
    sumir — é onde sistemas com muitas verificações costumam surpreender.
    """
    X, y = dados
    Xi = X.copy()
    Xi.loc[Xi.index[:150], 'x0'] = -999
    Xe = Xi.head(15)
    m1, m2 = RF(0, n=40).fit(Xi, y), RF(7, n=40).fit(Xi, y)
    r = M.Guard().audit(m1, m2, Xe, y.head(15), list(X.columns))
    camadas = [i['layer'] for i in r.not_evaluated]
    assert any('evaluation size' in c for c in camadas), 'perdeu a de tamanho'
    assert any('train/serving' in c for c in camadas), 'perdeu a de imputação'


def test_severidade_da_imputacao_nao_cai_com_n_pequeno(dados):
    """
    O piso de 20 amostras fazia a heurística de imputação SILENCIAR
    justamente quando a amostra é pequena — menos aviso quando o resultado
    é menos confiável. Medido: severidade caía de 'alta' para normal com
    n=15.
    """
    X, y = dados
    Xi = X.copy()
    Xi.loc[Xi.index[:150], 'x0'] = -999
    m1, m2 = RF(0, n=40).fit(Xi, y), RF(7, n=40).fit(Xi, y)
    sev = {}
    for n in (15, 700):
        r = M.Guard().audit(m1, m2, Xi.head(n), y.head(n), list(X.columns))
        ts = [i for i in r.not_evaluated if 'train/serving' in i['layer']]
        sev[n] = ts[0].get('severity') if ts else None
    assert sev[15] == sev[700] == 'alta', \
        f'severidade mudou com o tamanho: {sev}'


def test_causa_raiz_sobrevive_a_todas_as_ressalvas(dados):
    """A corrupção precisa ser reportada mesmo com três ressalvas ativas."""
    X, y = dados
    Xi = X.copy()
    Xi.loc[Xi.index[:150], 'x0'] = -999
    m1 = RF(0, n=40).fit(Xi, y)
    Xc = Xi.copy(); Xc['x1'] = Xc['x1'].mean()
    m2 = RF(0, n=40).fit(Xc, y)
    r = M.Guard().audit(m1, m2, Xi.head(15), y.head(15), list(X.columns))
    assert r.not_evaluated, 'devia haver ressalvas'
    assert r.verdict == 'BLOQUEADO'
    assert r.root_cause_features, 'a causa raiz sumiu sob as ressalvas'


# ═══ Condições sobre o RESULTADO (v1.73) ══════════════════════════════
# A varredura anterior cobriu condições sobre a ENTRADA. Estas são sobre a
# saída: casos em que o número sai certo e a interpretação não se sustenta.

def test_classe_rara_gera_ressalva(dados):
    """
    Um classificador avaliado com 2 exemplos de uma classe produzia
    APROVADO sem ressalva. Com 2 amostras nada se conclui sobre aquela
    classe — e quem lê APROVADO supõe que houve verificação.
    """
    X, _ = dados
    yb = pd.Series([0] * (len(X) - 2) + [1, 2])
    mk = lambda s: RandomForestClassifier(n_estimators=30, max_depth=5,
                                          random_state=s, n_jobs=1).fit(X, yb)
    r = M.Guard().audit(mk(0), mk(7), X, yb, list(X.columns))
    cr = [i for i in r.not_evaluated if 'class representation' in i['layer']]
    assert cr, 'classe com 2 exemplos deve gerar ressalva'
    assert cr[0]['severity'] == 'alta'
    assert 'np.' not in cr[0]['reason'], 'o repr do numpy vazou na mensagem'


def test_classe_bem_representada_nao_gera_ressalva(dados):
    X, _ = dados
    yb = pd.Series([0] * (len(X) - 60) + [1] * 60)
    mk = lambda s: RandomForestClassifier(n_estimators=30, max_depth=5,
                                          random_state=s, n_jobs=1).fit(X, yb)
    r = M.Guard().audit(mk(0), mk(7), X, yb, list(X.columns))
    assert not [i for i in r.not_evaluated
                if 'class representation' in i['layer']]


def test_regressao_nao_dispara_ressalva_de_classe(dados):
    """y contínuo não é classificação — a heurística não pode confundir."""
    X, y = dados
    r = M.Guard().audit(RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y), X, y,
                        list(X.columns))
    assert not [i for i in r.not_evaluated
                if 'class representation' in i['layer']]


def test_ordem_de_colunas_trocada_e_recusada(dados):
    """
    BUG 32: o sklearn recusa X com colunas reordenadas ("Feature names must
    be in the same order as they were in fit"). Alguma camada engolia esse
    erro e o veredito saía — apontando a variável ERRADA: corrompemos x0 e
    o laudo dizia x3.

    Reordenar colunas é acidente comum: um `select` diferente, um join que
    muda a ordem, um CSV regravado.
    """
    X, y = dados
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
    m2 = RF(0, n=40).fit(Xc, y)
    perm = list(reversed(X.columns))
    with pytest.raises(ValueError, match='ordem diferente'):
        M.Guard().audit(m1, m2, X[perm], y, perm)


def test_coluna_ausente_e_recusada(dados):
    X, y = dados
    m1, m2 = RF(0, n=40).fit(X, y), RF(7, n=40).fit(X, y)
    menos = list(X.columns)[:-1]
    with pytest.raises(ValueError, match='não correspondem'):
        M.Guard().audit(m1, m2, X[menos], y, menos)


def test_ordem_correta_aponta_a_variavel_certa(dados):
    """O contraponto: com a ordem certa, a causa raiz é a corrompida."""
    X, y = dados
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
    m2 = RF(0, n=40).fit(Xc, y)
    r = M.Guard().audit(m1, m2, X, y, list(X.columns))
    assert r.root_cause_features == ['x0'], \
        f'apontou {r.root_cause_features} em vez de x0'


def test_variavel_abaixo_do_filtro_nao_vira_causa_raiz(dados):
    """
    Uma variável com importância de 0,005 (abaixo do filtro de 1%) não pode
    ser apontada como causa — o perfil dela é ruído.
    """
    X, _ = dados
    rng = np.random.RandomState(3)
    y = pd.Series(3 * X.x0 + 2 * X.x1 + 0.02 * X.x3 + rng.randn(len(X)) * .2)
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['x3'] = Xc['x3'].mean()
    m2 = RF(0, n=40).fit(Xc, y)
    r = M.Guard().audit(m1, m2, X, y, list(X.columns))
    assert 'x3' not in r.root_cause_features, \
        'variável irrelevante virou causa raiz'


# ═══ v0.9.1 — bug 33 e redistribuição entre clones ════════════════════
def test_clone_nao_vira_causa_raiz():
    """
    Breast Cancer: RF vs XGBoost nos mesmos dados apontavam `mean_radius`
    como colapso. Ele tem r=0.99 com `mean_area`/`mean_perimeter`: o modelo
    novo só preferiu o clone. Não é falha de pipeline.
    """
    rng = np.random.RandomState(0)
    a = rng.uniform(0, 1, 800)
    X = pd.DataFrame({'a': a, 'a_clone': a + rng.randn(800) * .01,
                      'b': rng.uniform(0, 1, 800)})
    y = pd.Series(3 * a + X.b + rng.randn(800) * .2)
    m1 = RF(0, n=40).fit(X[['a', 'a_clone', 'b']], y)
    Xs = X.copy(); Xs['a'] = Xs['a'].mean()      # v2 aprende pelo clone
    m2 = RF(0, n=40).fit(Xs, y)
    r = M.Guard().audit(m1, m2, X, y, list(X.columns))
    assert 'a' not in r.root_cause_features
    assert 'a' in r.clone_redistribution
    assert any('clone' in i['reason'] for i in r.not_evaluated)


def test_colapso_sem_clone_continua_bloqueando(dados):
    X, y = dados
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
    r = M.Guard().audit(m1, RF(0, n=40).fit(Xc, y), X, y, list(X.columns))
    assert r.verdict == 'BLOQUEADO' and r.root_cause_features == ['x0']


def test_metodo_de_importancia_nao_e_misturado():
    """BUG 33: o método do v2 era descartado — SHAP vs permutação."""
    import inspect
    from modelcheck import importance
    src = inspect.getsource(importance.ImportanceCollapseDetector.compare)
    assert 'i2, _ =' not in src, 'o método do segundo modelo voltou a ser descartado'
    assert 'm1 != m2' in src


def test_clones_que_colapsam_juntos_nao_se_desculpam():
    """
    BUG 34: com temp e atemp (r=0.99) quebrados juntos no Bike Sharing, um
    servia de explicação para o outro e a causa raiz ficava vazia. O clone
    só explica um colapso se ele próprio não colapsou.
    """
    rng = np.random.RandomState(1)
    a = rng.uniform(0, 1, 800)
    X = pd.DataFrame({'a': a, 'a2': a + rng.randn(800) * .01,
                      'b': rng.uniform(0, 1, 800)})
    y = pd.Series(3 * a + X.b + rng.randn(800) * .2)
    m1 = RF(0, n=40).fit(X, y)
    Xq = X.copy(); Xq['a'] = Xq['a'].mean(); Xq['a2'] = Xq['a2'].mean()
    r = M.Guard().audit(m1, RF(0, n=40).fit(Xq, y), X, y, list(X.columns))
    assert r.verdict == 'BLOQUEADO'
    assert set(r.root_cause_features) >= {'a', 'a2'}
    assert not r.clone_redistribution


def test_piso_de_relevancia_se_adapta_ao_numero_de_variaveis():
    """
    BUG 37: com muitas variáveis, a fatia de cada uma fica abaixo do piso fixo
    de 2% e a camada de integridade não examinava nenhuma. Medido no SECOM
    (475 variáveis): a mais importante pesava 0,3% e uma falha total passava.
    """
    rng = np.random.RandomState(0)
    n, p = 700, 120
    X = pd.DataFrame({f'x{i}': rng.uniform(0, 1, n) for i in range(p)})
    y = pd.Series(sum(X[f'x{i}'] * (1.0 if i < 8 else .05) for i in range(p))
                  + rng.randn(n) * .1)
    m1 = RF(0, n=40).fit(X, y)
    Xc = X.copy(); Xc['x0'] = Xc['x0'].mean()
    r = M.Guard().audit(m1, RF(0, n=40).fit(Xc, y), X, y, list(X.columns),
                        X_train=Xc)
    assert r.verdict == 'BLOQUEADO', f'com {p} variáveis deveria bloquear'
    assert 'x0' in r.root_cause_features
