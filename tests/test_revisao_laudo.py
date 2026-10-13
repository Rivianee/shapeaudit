"""
Revisão do laudo e da CLI (v1.0.4) — BUG 50 a BUG 65.

O laudo é um instrumento de decisão: alguém promove ou não um modelo com base
nele. Cada "✓" e cada frase tranquilizadora precisa ser verdadeira. Um teste
por defeito confirmado na revisão; quase todos renderizam a partir de
dicionários montados à mão, sem treinar modelo.
"""
import io
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings('ignore')

import modelcheck
from modelcheck.laudo import render

RAIZ = Path(modelcheck.__file__).resolve().parents[1]


def _base(**kw):
    d = dict(verdict='APROVADO', problem_type='NONE', root_cause_features=[],
             divergences=[], collapses=[], model_diff=[], skew_features=[],
             skew_status='ok', clone_redistribution={}, relearning=[],
             decisao={}, n_eval=500, representation='shap', not_evaluated=[])
    d.update(kw)
    return d


def _texto(h):
    h = re.sub(r'<style.*?</style>', '', h, flags=re.S)
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', h))


def _secao(h, titulo):
    t = _texto(h)
    i = t.find(titulo)
    assert i >= 0, titulo
    return t[i:i + 600]


def _css(h):
    return h.split('<style>')[1].split('</style>')[0]


# ═══ BUG 50: reaprendizado recebia "✓ nenhuma variável deixou de influenciar"
def test_bug50_reaprendizado_nao_vira_visto_de_integridade():
    res = _base(verdict='REVISAR', problem_type='RELEARNING',
                relearning=['a'],
                divergences=[{'feature': 'b', 'frechet': 0.9}],
                model_diff=[{'feature': 'a', 'peso_v1': .55, 'peso_v2': 0.0,
                             'delta_pp': -55, 'categoria': 'perdeu_influencia'}])
    h = render(res, 'pt')
    assert 'nenhuma variável deixou de influenciar o modelo' not in h
    assert 'Não há sinal de variável perdida' not in h
    resumo = _secao(h, 'Resumo')
    assert 'a deixou de ser usada (reaprendizado)' in resumo
    assert '55% → 0%' in resumo
    he = render(res, 'en')
    assert 'no feature stopped influencing the model' not in he
    assert 'stopped being used (relearning)' in he


# ═══ BUG 51: BLOQUEADO afirmava constância sem ter verificado ══════════
def test_bug51_causa_raiz_so_afirma_o_que_foi_verificado():
    sem = _base(verdict='BLOQUEADO', problem_type='IMPORTANCE_COLLAPSE',
                root_cause_features=['a'], skew_status='ausente')
    def afirmacoes(h):
        """subtítulo do veredito + "por que" — onde a afirmação aparecia"""
        acao = h.split('<div class="acao">')[1].split('</div>')[0]
        pq = h.split('<section class="porque">')[1].split('</section>')[0]
        return _texto(acao + pq).lower()
    for lang in ('pt', 'en'):
        t = afirmacoes(render(sem, lang))
        assert 'constant' not in t, lang
        assert 'vai receber' not in t
        assert 'confirm' not in t
    com = dict(sem, skew_status='ok', skew_features=['a'])
    tp, te = _texto(render(com, 'pt')), _texto(render(com, 'en'))
    # pt e en dizem a mesma coisa: dados de TREINO da nova versão
    assert 'confirmou' in tp and 'dados de treino da nova versão' in tp
    assert 'confirmed' in te and "new version's training data" in te
    assert 'vai receber' not in tp


# ═══ BUG 52: resultado fora da faixa usual escondido ═══════════════════
def test_bug52_ood_aparece_e_frase_nao_generaliza():
    h = render(_base(ood_features=['renda']), 'pt')
    assert 'Nenhuma camada medida' not in h
    assert 'renda' in _secao(h, 'Resumo')
    h = render(_base(), 'pt')
    assert 'Nenhuma camada medida' not in h
    assert 'Nenhuma das camadas principais' in h


# ═══ BUG 53: "✓" para camada listada em not_evaluated ══════════════════
def test_bug53_sem_visto_para_camada_que_nao_rodou():
    forma = _base(not_evaluated=[{'layer': 'shape divergence',
                                  'reason': 'SHAP ERROR',
                                  'consequence': 'curvas não comparadas'}])
    h = render(forma, 'pt')
    assert 'nenhuma relação mudou de forma' not in h
    assert 'forma das relações: não verificado' in h
    integ = _base(not_evaluated=[{'layer': 'feature integrity',
                                  'reason': 'MemoryError na camada',
                                  'consequence': 'não examinado'}])
    h = render(integ, 'pt')
    assert 'nenhuma variável deixou de influenciar o modelo' not in h
    assert 'nenhuma substituição por variável quase idêntica' not in h
    assert 'não verificado (MemoryError na camada)' in h
    he = render(integ, 'en')
    assert 'no feature stopped influencing the model' not in he
    # a nota de clone (também sob 'feature integrity') não é lacuna
    clone = _base(verdict='ATENCAO', clone_redistribution={'a': 'a2'},
                  not_evaluated=[{'layer': 'feature integrity',
                                  'reason': 'a perdeu peso, mas tem clone '
                                            'quase idêntico (a2, |r|>=0.95)',
                                  'consequence': 'x'}])
    assert 'variáveis que deixaram de influenciar: não verificado' \
        not in render(clone, 'pt')


# ═══ BUG 54: explicação contradizia o veredito ═════════════════════════
def test_bug54_explicacao_segue_o_que_puxa_o_veredito():
    res = _base(verdict='ATENCAO', problem_type='DATA_SHIFT',
                skew_features=['c'],
                divergences=[{'feature': 'c', 'frechet': 1.2}])
    h = render(res, 'pt')
    porque = _secao(h, 'Por que este veredito')
    assert 'nem de inconsistência' not in h
    assert porque.index('treino') < porque.index('forma')
    he = render(res, 'en')
    assert 'no sign of a pipeline failure' not in he
    assert 'train/serving inconsistency' not in he
    # "passou nos checks principais" só quando todos rodaram
    sem = _base(verdict='REVISAR', skew_status='ausente',
                divergences=[{'feature': 'c', 'frechet': 1.2}])
    assert 'passou nos checks principais' not in render(sem, 'pt')
    assert 'passes the main checks' not in render(sem, 'en')
    assert 'nem de inconsistência' not in render(sem, 'pt')


# ═══ BUG 55: CLI compare montava dicionário parcial ════════════════════
class _ResultadoFalso:
    verdict = 'ATENCAO'
    problem_type = 'CLONE_REDISTRIBUTION'
    root_cause_features = []
    raw = {'layer1': {'sdd': {'divs': [], 'method': 'ALE (fallback)'}}}
    not_evaluated = []
    clone_redistribution = {'a': 'a_copy'}
    relearning = []
    skew_features = []
    skew_status = 'ok'
    skew_error = ''
    model_diff = []
    shape_divergences = []


def test_bug55_cli_compare_usa_o_dicionario_completo(tmp_path, monkeypatch):
    import modelcheck.cli as cli
    import pandas as pd

    class _G:
        def __init__(self, *a, **k):      # a CLI passa mode= desde a 1.0.4
            pass

        def audit(self, *a, **k):
            return _ResultadoFalso()
    monkeypatch.setattr(modelcheck, 'Guard', _G)
    monkeypatch.setattr(cli, '_carregar_modelo', lambda p: object())
    monkeypatch.setattr(cli, '_carregar_dados',
                        lambda *a: (pd.DataFrame({'a': [1.0, 2.0]}), None))
    monkeypatch.setenv('MODELCHECK_LANG', 'pt')
    out = tmp_path / 'r.html'
    cli.main(['compare', 'a', 'b', 'd', '--format', 'html', '-o', str(out)])
    h = out.read_text(encoding='utf-8')
    assert 'nenhuma substituição por variável quase idêntica' not in h
    assert 'substituída(s) por quase-cópia' in h
    assert 'ALE (fallback)' in h        # representation não é mais fixa


# ═══ BUG 56: XSS no explorer ═══════════════════════════════════════════
def test_bug56_explorer_escapa_tudo(monkeypatch):
    import pandas as pd
    import modelcheck.explorer as ex
    x = '<img src=x onerror=alert(1)>'
    fecha = '</script><script>alert(2)</script>'
    cols = [x, fecha]
    prof = {c: {'x': [0, 1], 'y_mean': [0, 1], 'y_std': [0, 0],
                'importance': .5} for c in cols}
    monkeypatch.setattr(ex, '_shap_profiles', lambda *a, **k: prof)
    X = pd.DataFrame({c: [0.0, 1.0] for c in cols})
    h = ex.InteractiveDivergenceExplorer().generate(
        None, None, X, title=x, label_v1="O'Brien" + x, label_v2=fecha,
        paired_result={'rules': [{'rule_text': x}]})
    assert '<img src=x onerror=' not in h
    assert 'alert(2)</script>' not in h
    assert "'O'Brien" not in h
    assert 'innerHTML' not in h
    assert 'MIT' in h and 'Apache' not in h


# ═══ BUG 57: Windows (cp1252) e README ════════════════════════════════
def test_bug57_cli_nao_quebra_em_stdout_cp1252(monkeypatch):
    import modelcheck.cli as cli
    buf = io.BytesIO()
    fake = io.TextIOWrapper(buf, encoding='cp1252')
    monkeypatch.setattr(sys, 'stdout', fake)
    cli._emitir(_base(verdict='BLOQUEADO', root_cause_features=['a']),
                'html', None)
    fake.flush()
    assert '✓'.encode('utf-8') in buf.getvalue() \
        or '⚠'.encode('utf-8') in buf.getvalue()


def test_bug57_readme_abre_arquivo_com_utf8():
    for nome in ('README.md', 'README_pt.md'):
        p = RAIZ / nome
        if not p.exists():
            continue
        for linha in p.read_text(encoding='utf-8').splitlines():
            if re.search(r"open\([^)]*['\"]w['\"]", linha):
                assert "encoding='utf-8'" in linha or \
                    'encoding="utf-8"' in linha, (nome, linha)


# ═══ BUG 58: segundo laudo recolore o primeiro no notebook ═════════════
def test_bug58_cor_do_veredito_fica_na_raiz_do_laudo():
    h = render(_base(verdict='BLOQUEADO', root_cause_features=['a']), 'pt')
    css = re.sub(r'/\*.*?\*/', '', _css(h), flags=re.S)
    assert '--marca:' not in css and '--marca-fundo:' not in css
    assert ':root' not in css
    assert re.search(r'<div class="folha" style="--marca:#8E2226;'
                     r'--marca-fundo:#FBF3F2">', h)


# ═══ BUG 59: escopo do CSS ═════════════════════════════════════════════
def _seletores(css):
    css = re.sub(r'/\*.*?\*/', '', css, flags=re.S)
    for m in re.finditer(r'([^{}]+)\{([^{}]*)\}', css):
        sel = m.group(1).strip()
        if sel.startswith('@'):
            continue
        for s in sel.split(','):
            yield s.strip()


def test_bug59_toda_regra_presa_a_folha_e_reset_sem_especificidade():
    css = _css(render(_base(verdict='REVISAR',
                            divergences=[{'feature': 'x', 'frechet': 1}]),
                      'pt')).replace('{{', '{').replace('}}', '}')
    sels = list(_seletores(css))
    assert sels
    soltos = [s for s in sels
              if not (s.startswith('.folha') or s.startswith(':where(.folha'))]
    assert not soltos, soltos
    assert ':where(.folha *)' in css
    assert not re.search(r'(^|[\s,}])\.folha \*\s*[{,]', css)
    # rodapé não colide com `ul.md .pe`
    assert not any(re.fullmatch(r'\.folha \.pe', s) for s in sels)
    h = render(_base(), 'pt')
    assert '<footer class="pe"' not in h


def test_bug59_margens_aparecem_no_arquivo_solto():
    pw = pytest.importorskip('playwright.sync_api')
    h = render(_base(verdict='REVISAR',
                     divergences=[{'feature': 'x', 'frechet': 1}]), 'pt')
    host = ('<html><head><style>h2{font-size:26px}</style></head><body>'
            '<h2 id="hh">host</h2><div id="o"></div></body></html>')
    try:
        with pw.sync_playwright() as p:
            kw = {}
            if os.path.exists('/opt/pw-browsers/chromium'):
                kw['executable_path'] = '/opt/pw-browsers/chromium'
            try:
                br = p.chromium.launch(**kw)
            except Exception:
                br = p.chromium.launch()
            pg = br.new_page()
            pg.set_content(h)
            mb = pg.eval_on_selector('.folha section.sumario',
                                     'e=>getComputedStyle(e).marginBottom')
            h2 = pg.eval_on_selector('.folha h2',
                                     'e=>getComputedStyle(e).marginBottom')
            pg.set_content(host)
            pg.evaluate("h=>{document.getElementById('o').innerHTML=h}", h)
            host_h2 = pg.eval_on_selector('#hh',
                                          'e=>getComputedStyle(e).fontSize')
            br.close()
    except Exception as e:                      # navegador indisponível
        pytest.skip(f'chromium indisponível: {e}')
    assert mb == '34px' and h2 == '12px'
    assert host_h2 == '26px'


# ═══ BUG 60: "Sim." com zero casos trocados ════════════════════════════
def test_bug60_decisao_zero_trocados_e_escape():
    h = render(_base(decisao={'trocados': 0, 'selecionados': 500,
                              'pct': 0.0}), 'pt')
    dec = _secao(h, 'A mudança afetou a decisão?')
    assert 'Não.' in dec and 'Sim.' not in dec
    x = '<img src=x onerror=alert(1)>'
    h = render(_base(decisao={'trocados': x, 'selecionados': x, 'pct': x}),
               'pt')
    assert '<img src=x onerror=' not in h


# ═══ BUG 61: idioma ════════════════════════════════════════════════════
def test_bug61_laudo_em_ingles_sem_consequencia_em_portugues():
    res = _base(not_evaluated=[
        {'layer': 'train/serving consistency', 'reason': 'sem X_train',
         'consequence': 'não dá para saber se os dados que chegam ao modelo '
                        'mudaram em relação ao treino. Passe `X_train=` para '
                        'verificar'},
        {'layer': 'fairness', 'reason': 'x',
         'consequence': 'duas versões podem errar igual no total e errar mais '
                        'para um grupo específico. Se o seu modelo afeta '
                        'pessoas, meça isso à parte com `fairness_evolucao`'},
        {'layer': 'shape divergence', 'reason': 'x',
         'consequence': 'frase desconhecida que não está no mapa'}],
        skew_status='ausente')
    he = render(res, 'en')
    for w in ('não', 'dá para', 'duas versões', 'meça', 'desconhecida'):
        assert w not in he, w
    # a pergunta de fairness é a mesma nos dois idiomas
    assert 'error change for any group of people' in render(_base(), 'en')
    assert 'O erro mudou para algum grupo de pessoas?' in render(_base(), 'pt')


@pytest.mark.parametrize('lang', ['pt-BR', 'pt_BR', 'PT', 'pt'])
def test_bug61_variantes_de_portugues(lang):
    assert '<html lang="pt"' in render(_base(), lang)


# ═══ BUG 62: NaN, inf e None ═══════════════════════════════════════════
def test_bug62_numeros_nao_finitos_e_ausentes():
    nan, inf = float('nan'), float('inf')
    d = {'feature': 'x', 'frechet': nan, 'shape_v1': 'crescente',
         'shape_v2': 'u', 'efeito_v1': [.1, nan, .3],
         'efeito_v2': [.2, .1, inf], 'bordas': [0, nan],
         'maior_gap': {'bin': 1, 'de': 0.0, 'ate': nan, 'v1': .1,
                       'v2': nan, 'delta': nan}}
    md = [{'feature': 'x', 'peso_v1': None, 'peso_v2': nan, 'delta_pp': None,
           'categoria': 'peso_estavel_forma_mudou', 'forma_mudou': True,
           'shape_v1': 'crescente', 'shape_v2': 'u'}]
    for lang in ('pt', 'en'):
        h = render(_base(verdict='REVISAR', divergences=[d], model_diff=md),
                   lang)
        corpo = re.sub(r'<style.*?</style>', '', h, flags=re.S)
        assert not re.search(r'nan|inf\b', corpo, flags=re.I), lang
    # frechet None numa divergência (dicionário antigo) não derruba
    h = render(_base(verdict='REVISAR', divergences=[
        {'feature': 'x', 'frechet': None}, {'feature': 'y', 'frechet': 0.5}]),
        'pt')
    assert 'distância 0.5' in h
    # frechet cru agora sai formatado
    h = render(_base(verdict='REVISAR', divergences=[
        {'feature': 'y', 'frechet': 0.123456789}]), 'pt')
    assert '0.123456789' not in h and '0.123' in h


# ═══ BUG 63: modelguard.frechet ════════════════════════════════════════
def test_bug63_modelguard_espelha_shapeaudit():
    r = subprocess.run([sys.executable, '-c',
                        'import modelguard.frechet, shapeaudit.frechet'],
                       cwd=str(RAIZ), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]


# ═══ BUG 64: Homepage ══════════════════════════════════════════════════
def test_bug64_homepage_sem_placeholder():
    t = (RAIZ / 'pyproject.toml').read_text(encoding='utf-8')
    assert 'SEU_USUARIO' not in t
    assert 'https://github.com/Rivianee/shapeaudit' in t


# ═══ BUG 65: INDETERMINADO e colapso do modelo ═════════════════════════
@pytest.mark.parametrize('lang', ['pt', 'en'])
def test_bug65_veredito_nao_aprovador_nao_diz_nenhuma_diferenca(lang):
    ind = render(_base(verdict='INDETERMINADO', problem_type=''), lang)
    col = render(_base(verdict='BLOQUEADO', problem_type='MODEL_COLLAPSED'),
                 lang)
    for h in (ind, col):
        assert 'Nenhuma diferença relevante' not in h
        assert 'No relevant difference' not in h
    assert ('mesmo valor' if lang == 'pt' else 'same value') in col
    assert ('Não foi possível medir' if lang == 'pt'
            else 'could not be measured') in _texto(ind)
