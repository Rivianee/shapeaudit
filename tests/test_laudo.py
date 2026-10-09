"""
Renderização do laudo (v1.69).

O HTML é o que o usuário não-técnico vê. Um identificador interno vazando
para a tela ("Ood only") custa mais credibilidade que um bug de cálculo,
porque é visível para quem não sabe avaliar o cálculo.
"""
import re, warnings
import pytest
warnings.filterwarnings('ignore')
from modelcheck.laudo import render, _texto_problema, _CAMADAS


# Os testes abaixo foram escritos antes do i18n (v1.70) e assumem os textos
# em português. Fixamos o idioma em vez de depender do ambiente — senão o
# resultado mudaria conforme a LANG da máquina que roda a suíte.
def _r(res, lang='pt'):
    return render(res, lang=lang)


def _base(**kw):
    d = dict(verdict='BLOQUEADO', problem_type='IMPORTANCE_COLLAPSE',
             root_cause_features=['alcohol'], divergences=[], collapses=[],
             n_eval=640, representation='shap', not_evaluated=[])
    d.update(kw)
    return d


# ═══ Nenhum identificador interno na tela ═════════════════════════════
@pytest.mark.parametrize('codigo', [
    'IMPORTANCE_COLLAPSE', 'SHAPE_DIVERGENCE', 'OOD_ONLY', 'SHAPE_ONLY',
    'SLOPE_ONLY', 'CANARY_ONLY', 'TOPO_ONLY', 'MULTIPLE', 'NONE', ''])
def test_tipo_de_problema_nunca_aparece_cru(codigo):
    h = _r(_base(problem_type=codigo))
    assert codigo not in h or codigo == '', \
        f'o código interno {codigo!r} vazou para o laudo'
    assert '_' not in _texto_problema(codigo)


def test_codigo_desconhecido_tem_fallback_legivel():
    """Um tipo novo não pode virar 'Ood only' na tela."""
    t = _texto_problema('SINAL_INEDITO_ONLY')
    assert '_' not in t and t[0].isupper() and t.endswith('.')


def test_sem_placeholder_nem_variavel_nao_substituida():
    h = _r(_base())
    assert '{' not in h.split('<style>')[0]
    assert 'None' not in re.sub(r'<style>.*?</style>', '', h, flags=re.S)


# ═══ A cobertura — o elemento central ═════════════════════════════════
def test_cobertura_conta_verificacoes_executadas():
    h = _r(_base(not_evaluated=[
        {'layer': 'fairness', 'reason': 'x', 'consequence': 'y'}]))
    assert f'/{len(_CAMADAS)}' in h
    assert f'>{len(_CAMADAS) - 1}<' in h


def test_lacuna_aparece_como_nao_verificada():
    h = _r(_base(not_evaluated=[
        {'layer': 'train/serving consistency', 'reason': 'sem X_train',
         'consequence': 'escala não foi verificada'}]))
    assert 'não verificado' in h
    assert 'escala não foi verificada' in h


def test_texto_distingue_nao_verificado_de_sem_problema():
    """A frase que impede a leitura errada do laudo."""
    h = _r(_base(not_evaluated=[
        {'layer': 'fairness', 'reason': 'x', 'consequence': 'y'}]))
    assert 'ausência de problema' in h and 'ausência de exame' in h


# ═══ Conteúdo ═════════════════════════════════════════════════════════
def test_causa_raiz_aparece_em_destaque():
    h = _r(_base())
    assert 'Comece por aqui' in h and 'alcohol' in h


def test_causa_raiz_nao_se_repete_na_tabela_de_padrao():
    """Ver a mesma variável em duas tabelas sugere dois problemas."""
    h = _r(_base(divergences=[
        {'feature': 'alcohol', 'shape_before': 'u', 'shape_after': 'irregular'},
        {'feature': 'pH', 'shape_before': 'u', 'shape_after': 'crescente'}]))
    # só a seção das curvas — a causa raiz pode (e deve) aparecer em
    # "Comece por aqui" e em "Próxima ação"
    corpo = h.split('Variáveis que mudaram de padrão')[-1].split('<section')[0]
    assert 'pH' in corpo
    assert corpo.count('alcohol') == 0


def test_aprovado_nao_mostra_secoes_de_problema():
    h = _r(_base(verdict='APROVADO', problem_type='NONE',
                     root_cause_features=[]))
    assert 'Pode promover' in h
    assert 'Comece por aqui' not in h


@pytest.mark.parametrize('v,esperado', [
    ('BLOQUEADO', 'Não promova'), ('ATENCAO', 'Promova com atenção'),
    ('APROVADO', 'Pode promover'), ('INDETERMINADO', 'Não foi possível medir')])
def test_veredito_vira_acao_e_nao_rotulo(v, esperado):
    """O usuário precisa saber o que fazer, não o nome interno do estado."""
    assert esperado in _r(_base(verdict=v))


# ═══ Segurança e robustez ═════════════════════════════════════════════
def test_nome_de_variavel_com_html_e_escapado():
    h = _r(_base(root_cause_features=['<script>alert(1)</script>']))
    assert '<script>alert' not in h
    assert '&lt;script&gt;' in h


def test_renderiza_sem_campos_opcionais():
    assert _r({'verdict': 'APROVADO'}).startswith('<!DOCTYPE html>')


def test_html_e_autocontido():
    """Sem rede: o laudo precisa abrir de um anexo de e-mail."""
    h = _r(_base())
    assert 'http://' not in h and 'https://' not in h
    assert '<link' not in h and '<script' not in h


# ═══ Idiomas (v1.70) ══════════════════════════════════════════════════
# O laudo é distribuído internacionalmente. Texto em português num relatório
# em inglês não é detalhe: é a primeira coisa que um avaliador estrangeiro
# nota, antes de avaliar qualquer número.

import os
from modelcheck.laudo import _idioma, _forma, _T, _ACAO_EN


@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_renderiza_nos_dois_idiomas(lang):
    h = render(_base(), lang=lang)
    assert f'<html lang="{lang}"' in h


def test_nenhum_texto_portugues_no_laudo_em_ingles():
    """O teste que impede o vazamento mais provável."""
    h = render(_base(
        divergences=[{'feature': 'pH', 'shape_before': 'crescente',
                      'shape_after': 'u_invertido'}],
        collapses=[{'feature': 'a', 'importance_before': .2,
                    'importance_after': .01, 'ratio': .05}],
        not_evaluated=[{'layer': 'fairness', 'reason': 'x',
                        'consequence': 'not checked'}]), lang='en')
    vazou = [w for w in (
        'Variável', 'Comece', 'verificações', 'Peso antes', 'Resultado',
        'amostras', 'crescente', 'u_invertido', 'Não promova',
        'Atribuições', 'verificado') if w in h]
    assert not vazou, f'português vazou no laudo em inglês: {vazou}'


def test_nenhum_texto_ingles_no_laudo_em_portugues():
    h = render(_base(
        divergences=[{'feature': 'pH', 'shape_before': 'crescente',
                      'shape_after': 'u'}]), lang='pt')
    vazou = [w for w in ('Start here', 'checks performed', 'Feature integrity',
                         'increasing', 'Do not promote') if w in h]
    assert not vazou, f'inglês vazou no laudo em português: {vazou}'


def test_os_dois_dicionarios_tem_as_mesmas_chaves():
    """Chave faltando num idioma vira KeyError na hora de renderizar."""
    assert set(_T['pt']) == set(_T['en'])


def test_todo_veredito_tem_acao_em_ingles():
    from modelcheck.laudo import _TONS
    faltando = [v for v in _TONS if v not in _ACAO_EN]
    assert not faltando, f'veredito sem tradução: {faltando}'


@pytest.mark.parametrize('env,esperado', [
    ('pt_BR.UTF-8', 'pt'), ('en_US.UTF-8', 'en'), ('', 'en')])
def test_idioma_segue_o_ambiente(env, esperado, monkeypatch):
    for v in ('MODELCHECK_LANG', 'LC_ALL', 'LANG'):
        monkeypatch.delenv(v, raising=False)
    if env:
        monkeypatch.setenv('MODELCHECK_LANG', env)
    assert _idioma() == esperado


def test_argumento_explicito_vence_o_ambiente(monkeypatch):
    monkeypatch.setenv('MODELCHECK_LANG', 'pt_BR')
    assert _idioma('en') == 'en'


def test_idioma_desconhecido_cai_em_ingles():
    assert _idioma('klingon') == 'en'


def test_forma_desconhecida_nao_quebra():
    """Uma categoria nova do detector não pode virar erro de renderização."""
    assert _forma('forma_inedita', 'en') == 'forma_inedita'
    assert _forma(None, 'pt') == '\u2014'


# ═══ v0.14.0: model diff e seções de decisão ═════════════════════════
def test_laudo_tem_as_secoes_de_decisao():
    h = _r(_base(divergences=[{'feature': 'pH', 'shape_v1': 'u',
                               'shape_v2': 'crescente'}]))
    for secao in ('Resumo', 'Leitura em uma frase', 'Por que este veredito',
                  'A mudança afetou a decisão',
                  'Por onde começar'):
        assert secao in h, secao


def test_model_diff_separa_peso_estavel_de_forma_mudou():
    h = _r(_base(divergences=[{'feature': 'age', 'shape_v1': 'crescente',
                               'shape_v2': 'u_invertido'}],
                 model_diff=[
        {'feature': 'monthly', 'peso_v1': .18, 'peso_v2': .36, 'delta_pp': 18.0,
         'categoria': 'ganhou_influencia', 'forma_mudou': False},
        {'feature': 'tenure', 'peso_v1': .32, 'peso_v2': .14, 'delta_pp': -18.0,
         'categoria': 'perdeu_influencia', 'forma_mudou': False},
        {'feature': 'age', 'peso_v1': .21, 'peso_v2': .22, 'delta_pp': 1.0,
         'categoria': 'peso_estavel_forma_mudou', 'forma_mudou': True,
         'shape_v1': 'crescente', 'shape_v2': 'u_invertido'}]))
    assert 'Ganhou influência' in h and 'Perdeu influência' in h
    assert 'Mesmo peso, comportamento diferente' in h
    assert 'Olhando só importância, ninguém veria estas' in h


def test_impacto_na_decisao_declara_quando_nao_medido():
    assert 'Não avaliado' in _r(_base())
    h = _r(_base(decisao={'trocados': 17, 'selecionados': 128, 'pct': 13.3}))
    assert '17' in h and '13.3' in h


def test_distancia_vem_qualificada():
    """O número sozinho não diz nada a quem chega agora (v0.16)."""
    h = _r(_base(divergences=[{'feature': 'pH', 'shape_v1': 'u',
                               'shape_v2': 'crescente', 'frechet': 2.9,
                               'limiar': 0.33}]))
    assert 'divergência alta' in h and 'distância' in h


def test_legenda_das_curvas_aparece():
    h = _r(_base(divergences=[{'feature': 'pH', 'shape_v1': 'u',
                               'shape_v2': 'crescente'}]))
    assert 'versão atual' in h and 'versão candidata' in h


def test_nao_conta_o_mesmo_fato_duas_vezes():
    """
    A coluna que quebrou aparece como colapso de importância E como diferença
    entre treino e inferência. É o mesmo fato: o resumo não pode listar os
    dois como problemas separados (v0.17).
    """
    h = _r(_base(verdict='BLOQUEADO', problem_type='IMPORTANCE_COLLAPSE',
                 root_cause_features=['Contract'], skew_features=['Contract']))
    assert 'nenhuma outra diferença' in h
    assert 'chegam diferentes do treino' not in h


def test_topo_nomeia_a_variavel_e_nao_repete_a_leitura():
    h = _r(_base(verdict='BLOQUEADO', problem_type='IMPORTANCE_COLLAPSE',
                 root_cause_features=['Contract']))
    assert 'Contract' in h.split('Leitura em uma frase')[0]
    leitura = h.split('Leitura em uma frase')[1][:400]
    assert 'aponta primeiro para um problema de dados' in leitura


def test_linguagem_nao_e_categorica_demais():
    """'É a assinatura de' virou 'é compatível com' — o próprio artigo mostra
    que o sistema não separa perfeitamente perda legítima de falha."""
    h = _r(_base(verdict='BLOQUEADO', problem_type='IMPORTANCE_COLLAPSE',
                 root_cause_features=['Contract']))
    assert 'compatível com falha no pipeline' in h
    assert 'assinatura de falha' not in h
