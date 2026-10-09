"""
O tutorial como teste de integração (v1.49).

Motivo: documentação que não é executada apodrece em silêncio. Se a API
mudar, o TUTORIAL.md fica errado e ninguém percebe até um usuário reclamar.

Estes testes reproduzem exatamente o fluxo documentado e exigem as saídas
que o tutorial promete.
"""
import warnings, numpy as np, pandas as pd, pytest
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from modelcheck import Guard, TrainServingValidator
from modelcheck.validation import plain_report


@pytest.fixture(scope='module')
def tutorial():
    """Reproduz o setup do TUTORIAL.md com dados sintéticos equivalentes."""
    rng = np.random.RandomState(42)
    n = 1000
    X = pd.DataFrame({
        'alcohol':          rng.uniform(8, 15, n),
        'sulphates':        rng.uniform(0.3, 1.2, n),
        'volatile_acidity': rng.uniform(0.1, 1.2, n),
        'pH':               rng.uniform(2.9, 3.8, n),
    })
    # Pesos calibrados para reproduzir a premissa do tutorial: 'alcohol'
    # importa, mas não domina — como no Wine Quality real, onde neutralizá-la
    # muda o R² em apenas 0.012. Com 'alcohol' dominante o ΔR² seria 0.69 e o
    # problema apareceria na métrica agregada, o que descaracteriza o caso.
    y = pd.Series(0.12 * X.alcohol + 1.4 * X.sulphates
                  - 1.6 * X.volatile_acidity + 0.9 * X.pH
                  + rng.randn(n) * 0.55)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    mk = lambda Xd: RandomForestRegressor(
        n_estimators=100, max_depth=8, random_state=42, n_jobs=1).fit(Xd, ytr)
    v1 = mk(Xtr)
    Xq = Xtr.copy(); Xq['alcohol'] = Xq['alcohol'].mean()   # pipeline quebrado
    v2 = mk(Xq)
    v1b = RandomForestRegressor(n_estimators=100, max_depth=8,
                                random_state=7, n_jobs=1).fit(Xtr, ytr)
    return dict(v1=v1, v2=v2, v1b=v1b, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                cols=list(X.columns))


# ── §2: o R² quase não cai, mas o modelo está quebrado ───────────────
def test_delta_r2_e_pequeno_apesar_do_pipeline_quebrado(tutorial):
    t = tutorial
    r2_1 = t['v1'].score(t['Xte'], t['yte'])
    r2_2 = t['v2'].score(t['Xte'], t['yte'])
    assert abs(r2_1 - r2_2) < 0.10, \
        ("a premissa do tutorial é que a métrica agregada NÃO revela o "
         f"problema; ΔR²={abs(r2_1-r2_2):.4f} está grande demais")


# ── §3: auditoria em duas linhas ─────────────────────────────────────
def test_audit_bloqueia_e_aponta_a_causa_raiz(tutorial):
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    # O VEREDITO depende da magnitude do dano e varia com o dataset — no
    # Wine real dá BLOQUEADO, aqui dá ATENCAO. O que NÃO pode variar é o
    # diagnóstico: o tipo de anomalia e a causa raiz.
    assert r.verdict != 'APROVADO', \
        f"pipeline quebrado não pode ser aprovado, veio {r.verdict}"
    assert r.problem_type == 'IMPORTANCE_COLLAPSE', \
        f"o tipo deve ser colapso de importância, veio {r.problem_type}"
    assert 'alcohol' in r.root_cause_features, \
        f"a causa raiz é 'alcohol': {r.root_cause_features}"


def test_causa_raiz_vem_primeiro_na_lista(tutorial):
    """
    Achado de UX (v1.49): a lista saía como ['residual sugar', ..., 'alcohol']
    — causa raiz por último. O usuário investigaria na ordem errada.
    """
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    if r.root_cause_features:
        assert r.affected_features[0] in r.root_cause_features, \
            (f"a primeira feature listada deve ser causa raiz; "
             f"lista={r.affected_features}, raiz={r.root_cause_features}")


def test_plain_diz_por_onde_comecar(tutorial):
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    assert 'Comece por' in r.plain, \
        "o relatório deve dizer explicitamente por onde começar"
    assert 'alcohol' in r.plain
    assert 'Recomendação' in r.plain


def test_campos_documentados_existem(tutorial):
    """O tutorial promete estes atributos. Não podem desaparecer."""
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    for attr in ('verdict', 'problem_type', 'root_cause_features',
                 'affected_features', 'plain', 'plain_debug'):
        assert hasattr(r, attr), f"atributo documentado ausente: {attr}"
    assert isinstance(r.root_cause_features, list)
    assert isinstance(r.plain, str) and len(r.plain) > 50


# ── §4: o contraste — retreino saudável ──────────────────────────────
def test_retreino_saudavel_aprova(tutorial):
    t = tutorial
    r = Guard().audit(t['v1'], t['v1b'], t['Xte'], t['yte'])
    assert r.verdict == 'APROVADO', \
        f"só a semente mudou; deveria aprovar, veio {r.verdict}"
    assert r.problem_type == 'NONE'
    assert r.affected_features == []


# ── §5: check_features ───────────────────────────────────────────────
def test_check_features_isola_a_feature_corrompida(tutorial):
    t = tutorial
    c = Guard().check_features(t['v1'], t['v2'], t['Xte'],
                               monitored=['alcohol', 'sulphates'],
                               feature_names=t['cols'])
    assert c['alert'] is True
    assert c['collapsed'] == ['alcohol'], \
        f"deve isolar só 'alcohol': {c['collapsed']}"
    assert 'alcohol' in c['plain']


def test_check_features_silencioso_em_retreino_saudavel(tutorial):
    t = tutorial
    c = Guard().check_features(t['v1'], t['v1b'], t['Xte'],
                               monitored=['alcohol'], feature_names=t['cols'])
    assert c['alert'] is False, "variância de semente não deve alertar"


# ── §6: train/serving skew ───────────────────────────────────────────
def test_validator_detecta_escala_errada_no_serving(tutorial):
    t = tutorial
    Xsv = t['Xte'].copy()
    Xsv['alcohol'] = Xsv['alcohol'] * 1.35      # unidade errada
    r = TrainServingValidator().check(t['Xtr'], Xsv, t['cols'])
    assert 'alcohol' in r['flagged']
    assert any(a in r['alerts'] for a in
               ('MEAN_SHIFT', 'PSI_ALERT', 'OUT_OF_RANGE', 'SCALE_CHANGE'))
    assert 'alcohol' in plain_report(r)


def test_validator_silencioso_com_dados_equivalentes(tutorial):
    t = tutorial
    r = TrainServingValidator().check(t['Xtr'], t['Xtr'], t['cols'])
    assert r['flagged'] == []
    assert r['skew_score'] == 100.0


# ── §7: integrações ──────────────────────────────────────────────────
def test_to_mlflow_e_to_slack_com_as_chaves_documentadas(tutorial):
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    m = r.to_mlflow()
    assert 'modelguard.health_score' in m
    assert all(isinstance(v, float) for v in m.values()), \
        "mlflow.log_metrics aceita só float"
    s = r.to_slack('wine-quality')
    assert 'text' in s and 'blocks' in s
    assert 'wine-quality' in s['text']
    assert r.verdict in s['text']


# ── §9: sequência de versões ─────────────────────────────────────────
def test_audit_versions_exige_cinco_versoes_para_tendencia(tutorial):
    """FP de 34,2% com 3 transições — a API precisa bloquear."""
    t = tutorial
    g = Guard()
    r4 = g.audit_versions([t['v1'], t['v1b'], t['v1'], t['v1b']],
                          t['Xte'], t['yte'], t['cols'])
    assert 'indeterminado' in r4['trend'], \
        f"com 3 transições a tendência deve ser indeterminada: {r4['trend']}"
    r6 = g.audit_versions([t['v1'], t['v1b']] * 3,
                          t['Xte'], t['yte'], t['cols'])
    assert r6['trend'] in ('estável', 'piorando', 'melhorando')


# ── §11: o snippet final do tutorial roda ────────────────────────────
def test_snippet_de_reproducao_roda(tutorial):
    """O bloco de código do §11 precisa funcionar como está escrito."""
    t = tutorial
    r = Guard().audit(t['v1'], t['v2'], t['Xte'], t['yte'])
    linha = f"{r.verdict} | {r.root_cause_features}"
    assert r.verdict != 'APROVADO'
    assert 'alcohol' in linha


# ── Prova de novidade contra Evidently oficial (v1.50) ────────────────
def test_prova_de_novidade_contra_evidently_oficial(tutorial):
    """
    O claim central do projeto: existe uma classe de falha que os detectores
    de data drift não podem ver, por construção.

    Quando v1 e v2 são avaliados no MESMO X, P(X) é idêntico — nenhum
    detector de data drift pode alertar. Medido com Evidently 0.7.21
    oficial: 0 colunas com drift no caso RF vs GB (ΔR²=0.007), contra 3
    divergências de forma detectadas pelo SDD.

    Marcado como skip se evidently não estiver instalado.
    """
    pytest.importorskip('evidently', reason='evidently não instalado')
    from evidently import Report, Dataset, DataDefinition
    from evidently.presets import DataDriftPreset
    from modelcheck import audit

    t = tutorial
    Xe, ye, fs = t['Xte'], t['yte'], t['cols']

    ref = Xe.copy(); cur = Xe.copy()
    ref['prediction'] = t['v1'].predict(Xe)
    cur['prediction'] = t['v2'].predict(Xe)
    dd = DataDefinition(numerical_columns=fs + ['prediction'])
    rep = Report([DataDriftPreset()]).run(
        Dataset.from_pandas(cur, data_definition=dd),
        Dataset.from_pandas(ref, data_definition=dd))
    d = rep.dict()
    n_drift = 0
    for m in d.get('metrics', []):
        v = m.get('value')
        if isinstance(v, dict) and 'count' in v:
            n_drift = max(n_drift, int(v['count']))

    n_sdd = len(audit(t['v1'], t['v2'], Xe, verbose=False)['divergences'])

    # As FEATURES não derivaram: P(X) é o mesmo conjunto nos dois lados.
    assert n_drift <= 1, \
        (f"as features são idênticas nos dois lados; drift de {n_drift} "
         f"colunas indicaria problema no setup do teste")
    assert n_sdd >= 1, \
        "o SDD deve detectar a mudança estrutural que o data drift não vê"
