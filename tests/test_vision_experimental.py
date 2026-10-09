"""
Validação com ground truth de `vision.py` (v1.54).

Último módulo do pacote com zero teste. Validado sem torch/tensorflow: um
`VisionWrapper` falso cuja regra de decisão eu controlo, e embeddings
sintéticos com separação conhecida.

Bugs encontrados: 25 (cosseno entre médias contaminava o veredito) e
26 (arrays vazios / 1 amostra devolviam número sem significado).
"""
import warnings, numpy as np, pytest
warnings.filterwarnings('ignore')
from modelcheck.experimental.vision import (EmbeddingDrift, VisionFairness,
                                            VisionWrapper)


class _FakeVision(VisionWrapper):
    """Classificador determinístico com regra de decisão explícita."""

    def __init__(self, regra, n_cls=2):
        self.regra = regra
        self.n_cls = n_cls
        self.framework = 'fake'

    def predict(self, X):
        p = np.zeros((len(X), self.n_cls))
        for i, x in enumerate(X):
            p[i, self.regra(x)] = 1.0
        return p

    def predict_classes(self, X):
        return np.array([self.regra(x) for x in X])

    def embeddings(self, X, layer_name=None):
        return np.asarray(X).reshape(len(X), -1)[:, :16]


@pytest.fixture(scope='module')
def dados():
    rng = np.random.RandomState(0)
    X = rng.rand(120, 8, 8)
    # regra justa: só o conteúdo da imagem decide
    justo = _FakeVision(lambda x: int(x[:4, :4].mean() > 0.5))
    # regra injusta: usa o "grupo" codificado no pixel [0,1]
    injusto = _FakeVision(
        lambda x: int(x[0, 0] > 0.5) if x[0, 1] > 0.5
        else int(x[:4, :4].mean() > 0.5))
    y = np.array([int(x[:4, :4].mean() > 0.5) for x in X])
    grupos = np.array([0 if x[0, 1] <= 0.5 else 1 for x in X])
    return X, y, grupos, justo, injusto


# ═══ EmbeddingDrift — ground truth de separação ═══════════════════════
@pytest.fixture(scope='module')
def embs():
    rng = np.random.RandomState(0)
    return dict(A=rng.randn(200, 16), A2=rng.randn(200, 16),
                B=rng.randn(200, 16) + 3.0, C=rng.randn(200, 16) * 4.0)


def test_embedding_drift_mesma_distribuicao_nao_alerta(embs):
    """
    BUG 25: o veredito era `mmd > t1 OR cosine > t2`, e o cosseno entre as
    MÉDIAS dava 0.824 em amostras da MESMA distribuição — em alta dimensão a
    média tende a zero e o ângulo fica instável. Todo par saudável virava
    falso positivo, apesar de o MMD (0.004) estar correto.
    """
    r = EmbeddingDrift().detect(embs['A'], embs['A2'])
    assert r['drift_detected'] is False, \
        f"mesma distribuição não deve driftar: mmd={r['mmd']}, cos={r['cosine_dist']}"


def test_embedding_drift_identico_nao_alerta(embs):
    r = EmbeddingDrift().detect(embs['A'], embs['A'].copy())
    assert r['drift_detected'] is False
    assert abs(r['mmd']) < 1e-6


def test_embedding_drift_media_deslocada_alerta(embs):
    r = EmbeddingDrift().detect(embs['A'], embs['B'])
    assert r['drift_detected'] is True
    assert r['mmd'] > 0.1


def test_embedding_drift_variancia_maior_alerta(embs):
    r = EmbeddingDrift().detect(embs['A'], embs['C'])
    assert r['drift_detected'] is True


def test_embedding_drift_cosseno_e_informativo_nao_veredito(embs):
    """O cosseno continua reportado, mas com severidade INFO."""
    r = EmbeddingDrift().detect(embs['A'], embs['A2'])
    cos_alerts = [a for a in r['alerts'] if a.get('metric') == 'cosine_dist']
    for a in cos_alerts:
        assert a['severity'] == 'INFO', \
            "cosseno entre médias não deve ser ALERTA — é naturalmente alto"


# ═══ BUG 26: "nada para medir" ════════════════════════════════════════
def test_embedding_drift_arrays_vazios_levantam_erro():
    """Antes: InvalidParameterError do sklearn, sem explicação."""
    with pytest.raises(ValueError, match='insuficiente|2D'):
        EmbeddingDrift().detect(np.zeros((0, 16)), np.zeros((0, 16)))


def test_embedding_drift_uma_amostra_levanta_erro(embs):
    """Antes: devolvia mmd=0.787 — número sem significado estatístico."""
    with pytest.raises(ValueError, match='insuficiente'):
        EmbeddingDrift().detect(embs['A'][:1], embs['B'][:1])


def test_embedding_drift_dimensoes_incompativeis_levantam_erro(embs):
    """Antes: ValueError cru de broadcast, sem dizer o que fazer."""
    rng = np.random.RandomState(1)
    with pytest.raises(ValueError, match='incompat|mesma camada'):
        EmbeddingDrift().detect(embs['A'], rng.randn(50, 8))


def test_embedding_drift_array_1d_levanta_erro(embs):
    with pytest.raises(ValueError, match='2D'):
        EmbeddingDrift().detect(embs['A'][:, 0], embs['B'][:, 0])


# ═══ VisionFairness ═══════════════════════════════════════════════════
def test_vision_fairness_modelo_justo_nao_alerta(dados):
    X, y, grupos, justo, _ = dados
    r = VisionFairness().check(justo, X, y, grupos)
    assert r['bias_detected'] is False
    assert r['accuracy_gap'] < 0.05


def test_vision_fairness_modelo_injusto_alerta(dados):
    """A regra do modelo injusto usa o atributo de grupo para decidir."""
    X, y, grupos, _, injusto = dados
    r = VisionFairness().check(injusto, X, y, grupos)
    assert r['bias_detected'] is True
    assert r['accuracy_gap'] > 0.1


def test_vision_fairness_grupo_unico_nao_quebra(dados):
    X, y, _, justo, _ = dados
    r = VisionFairness().check(justo, X, y, np.zeros(len(X), dtype=int))
    assert r['accuracy_gap'] == 0.0


# ═══ Contrato de alertas ══════════════════════════════════════════════
def test_alertas_vision_tem_type_e_severity(embs):
    """Mesmo contrato exigido em llm.py (bug 24)."""
    r = EmbeddingDrift().detect(embs['A'], embs['B'])
    for a in r['alerts']:
        assert 'type' in a, f"alerta sem type: {a}"
        assert 'severity' in a, f"alerta sem severity: {a}"


def test_vision_continua_fora_do_namespace_principal():
    import modelcheck
    for nome in ('VisionGuard', 'VisionFairness', 'EmbeddingDrift',
                 'SaliencyDiff'):
        assert not hasattr(modelcheck, nome)
