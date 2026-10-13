"""
Revisão da 4ª camada (TrainServingValidator) — bugs 43 a 49.

Cada teste de defeito falha na 1.0.3 e passa com a correção. Os testes de
falso alarme rodam splits saudáveis em várias sementes e limitam a taxa de
alarme; o limite e a taxa medida estão no comentário de cada um.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from modelcheck.validation import TrainServingValidator, psi

warnings.filterwarnings('ignore')


def _linha(tr, sv):
    r = TrainServingValidator().check(pd.DataFrame({'f': tr}),
                                      pd.DataFrame({'f': sv}))
    return r['features'][0], r['flagged']


def _taxa(gen, n_tr, n_sv, sementes):
    v = TrainServingValidator(); k = 0
    for s in sementes:
        r = np.random.default_rng(s)
        t = pd.DataFrame({'f': gen(r, n_tr)})
        u = pd.DataFrame({'f': gen(r, n_sv)})
        k += bool(v.check(t, u)['flagged'])
    return k


# ═══ BUG 43: ±inf invisível ═════════════════════════════════════════
def test_bug43_inferencia_toda_inf_alarma():
    r = np.random.default_rng(0)
    row, flagged = _linha(r.normal(0, 1, 2000), np.full(500, np.inf))
    assert flagged == ['f']
    assert 'NONFINITE' in row['alerts']
    assert 'MISSING_SHIFT' in row['alerts']


def test_bug43_trinta_por_cento_inf_alarma():
    r = np.random.default_rng(1)
    sv = r.normal(0, 1, 1000)
    sv[r.random(1000) < .3] = -np.inf
    row, flagged = _linha(r.normal(0, 1, 5000), sv)
    assert flagged == ['f'] and 'NONFINITE' in row['alerts']


# ═══ BUG 44: flag rara indo a zero ═══════════════════════════════════
def test_bug44_flag_rara_indo_a_zero_alarma():
    r = np.random.default_rng(0)
    tr = (r.random(50000) < .008).astype(int)
    row, flagged = _linha(tr, np.zeros(20000, int))
    assert 'RATE_SHIFT' in row['alerts'], row      # 1.0.3: z=12,8 e nada
    # mesma taxa no uso: não alarma
    _, ok = _linha(tr, (r.random(20000) < .008).astype(int))
    assert ok == []


# ═══ BUG 45: binária comum sem teste de taxa ═════════════════════════
@pytest.mark.parametrize('lo,hi', [(0, 1), (1, 2), (-5.0, 7.5)])
def test_bug45_binaria_comum_30_para_50_alarma(lo, hi):
    r = np.random.default_rng(0)
    tr = np.where(r.random(10000) < .3, hi, lo)
    sv = np.where(r.random(2000) < .5, hi, lo)
    row, flagged = _linha(tr, sv)
    assert 'RATE_SHIFT' in row['alerts'], row
    assert isinstance(row['z'], float)


def test_bug45_binaria_comum_nivel_novo_e_sem_complexo():
    r = np.random.default_rng(0)
    tr = (r.random(4000) < .3).astype(float)
    row, _ = _linha(tr, np.full(500, 2.0))
    assert 'NEW_LEVEL' in row['alerts'] and row['z'] is None


# ═══ BUG 46: PSI fundia o nível mais alto em feature discreta ════════
def test_bug46_recodificar_nivel_mais_alto_alarma():
    r = np.random.default_rng(0)
    tr = r.integers(0, 5, 10000)
    sv = r.integers(0, 5, 2000)
    sv = np.where(sv == 4, 3, sv)
    assert psi(tr, sv) > .25                       # 1.0.3: 0,0008
    row, flagged = _linha(tr, sv)
    assert 'PSI_ALERT' in row['alerts']


def test_bug46_discreta_com_muitos_niveis_nao_funde_o_topo():
    # 12 níveis (> n_bins): caminho de quantis; o topo tem massa própria
    r = np.random.default_rng(0)
    tr = r.integers(0, 12, 10000)
    sv = r.integers(0, 12, 2000)
    assert psi(tr, np.where(sv == 11, 10, sv)) > .25


# ═══ BUG 47: categórica só comparava o conjunto de níveis ════════════
@pytest.mark.parametrize('mudanca', ['inverte', 'troca_rotulos'])
def test_bug47_frequencia_categorica_alarma(mudanca):
    r = np.random.default_rng(0)
    tr = r.choice(list('abc'), 10000, p=[.8, .15, .05])
    if mudanca == 'inverte':
        sv = r.choice(list('abc'), 1000, p=[.05, .15, .8])
    else:
        sv = pd.Series(r.choice(list('abc'), 1000, p=[.8, .15, .05])) \
            .map({'a': 'c', 'b': 'b', 'c': 'a'}).to_numpy()
    row, flagged = _linha(tr, sv)
    assert flagged == ['f'] and 'PSI_ALERT' in row['alerts'], row


# ═══ BUG 48: cauda pesada → SCALE_CHANGE falso ═══════════════════════
def test_bug48_lognormal_saudavel_nao_acusa_escala():
    # limite 2/30. Medido: 0/30 (0/200 em sementes 0..199); na 1.0.3 era
    # ~45% (89/200), quase tudo SCALE_CHANGE pela razão de desvios.
    k = _taxa(lambda r, n: r.lognormal(0, 1.5, n), 10000, 200, range(30))
    assert k <= 2, f'{k}/30 splits saudáveis de lognormal alarmaram'


def test_bug48_escala_real_continua_detectada():
    r = np.random.default_rng(0)
    row, _ = _linha(r.lognormal(0, 1.5, 10000), r.lognormal(0, 1.5, 200) * 3)
    assert 'SCALE_CHANGE' in row['alerts']


# ═══ BUG 49: piso de ruído ignorava o tamanho da inferência ══════════
def test_bug49_piso_de_ruido_usa_tamanho_da_inferencia():
    r = np.random.default_rng(0)
    row, _ = _linha(r.normal(0, 1, 10000), r.normal(0, 1, 200))
    # com metades de 5k × 5k o piso saía ~0,002; com 200 linhas o PSI
    # nulo de 10 bins fica na casa de 0,05–0,1
    assert row['psi_piso_ruido'] > .05, row['psi_piso_ruido']


# ═══ falsos alarmes: splits saudáveis ════════════════════════════════
def test_falso_alarme_normal():
    # limite 1/30. Medido: 0/30 (0/200 em sementes 0..199).
    k = _taxa(lambda r, n: r.normal(0, 1, n), 10000, 2000, range(30))
    assert k <= 1, f'{k}/30'


def test_falso_alarme_inferencia_pequena():
    # 10k × 200, a situação do bug 49. Limite 2/40 por feature. Medido:
    # 0/40 (0/200 em sementes 0..199); na 1.0.3 ~1,5% por feature, 17,5%
    # das auditorias de 10 features (hoje 7,5%).
    k = _taxa(lambda r, n: r.normal(0, 1, n), 10000, 200, range(100, 140))
    assert k <= 2, f'{k}/40'


def test_falso_alarme_binaria_rara():
    # garantia do bug 39: 0,2% de positivos, 10k × 300. Limite 2/100.
    # Medido: 0/100 (1.0.3: 1/100; sem a correção de continuidade 2/100).
    v = TrainServingValidator(); k = 0
    for s in range(100):
        t = (np.random.default_rng(s).random(10000) < .002).astype(int)
        u = (np.random.default_rng(s + 999).random(300) < .002).astype(int)
        k += bool(v.check(pd.DataFrame({'f': t}),
                          pd.DataFrame({'f': u}))['flagged'])
    assert k <= 2, f'{k}/100'


def test_falso_alarme_binaria_comum():
    # 30% de positivos, 10k × 500. Limite 2/100. Medido: 0/100
    # (1/300 em sementes 0..299).
    k = _taxa(lambda r, n: (r.random(n) < .3).astype(int), 10000, 500,
              range(100))
    assert k <= 2, f'{k}/100'


def test_falso_alarme_categorica():
    # 80/15/5, 10k × 200. Limite 1/30. Medido: 0/30 (0/200).
    k = _taxa(lambda r, n: r.choice(list('abc'), n, p=[.8, .15, .05]),
              10000, 200, range(30))
    assert k <= 1, f'{k}/30'


# ═══ o que já funcionava continua funcionando ════════════════════════
def _base(r, n):
    return pd.DataFrame({
        'x': r.normal(10, 2, n), 'z': r.normal(0, 1, n),
        'b': (r.random(n) < .3).astype(int), 'k': r.integers(0, 4, n),
        'c': r.choice(list('abc'), n)})


@pytest.mark.parametrize('nome,col,fn', [
    ('constante na media', 'x', lambda d: 10.0),
    ('constante em 0', 'x', lambda d: 0.0),
    ('constante fora da faixa', 'x', lambda d: 1e6),
    ('toda NaN', 'x', lambda d: np.nan),
    ('30% NaN', 'x', lambda d: d.x.mask(np.arange(len(d)) % 10 < 3)),
    ('x1000', 'x', lambda d: d.x * 1000),
    ('sinal invertido', 'x', lambda d: -d.x),
    ('rotulos novos', 'c', lambda d: d.c.str.upper()),
    ('binaria comum constante', 'b', lambda d: 0),
    ('discreta constante', 'k', lambda d: 0),
])
def test_deteccoes_preservadas(nome, col, fn):
    r = np.random.default_rng(0)
    tr, sv = _base(r, 5000), _base(r, 1000)
    sv[col] = fn(sv)
    assert col in TrainServingValidator().check(tr, sv)['flagged'], nome


def test_coluna_embaralhada_nao_alarma():
    r = np.random.default_rng(0)
    tr, sv = _base(r, 5000), _base(r, 1000)
    sv['x'] = r.permutation(sv['x'].to_numpy())
    assert TrainServingValidator().check(tr, sv)['flagged'] == []


@pytest.mark.parametrize('dtype', ['Int64', 'Float64', 'boolean', 'bool'])
def test_dtypes_nulaveis(dtype):
    r = np.random.default_rng(0)
    if dtype in ('boolean', 'bool'):
        tr = pd.array(r.random(5000) < .3, dtype=dtype)
        ok = pd.array(r.random(1000) < .3, dtype=dtype)
        ruim = pd.array(np.zeros(1000, bool), dtype=dtype)
    else:
        tr = pd.array(r.integers(0, 50, 5000), dtype=dtype)
        ok = pd.array(r.integers(0, 50, 1000), dtype=dtype)
        ruim = ok * 1000
    assert _linha(tr, ok)[1] == []
    assert _linha(tr, ruim)[1] == ['f']
    if dtype != 'bool':
        na = ok.copy(); na[:300] = pd.NA
        assert 'MISSING_SHIFT' in _linha(tr, na)[0]['alerts']


def test_category_contra_str_nao_alarma():
    r = np.random.default_rng(0)
    p = [.8, .15, .05]
    tr = pd.Categorical(r.choice(list('abc'), 5000, p=p))
    assert _linha(tr, r.choice(list('abc'), 1000, p=p))[1] == []
