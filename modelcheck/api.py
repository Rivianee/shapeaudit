"""
api.py — API de alto nível do ShapeAudit (classe Guard)

Fachada única para uso em produção. Esconde a escolha de detectores e
traduz o diagnóstico para linguagem de negócio.

    from shapeaudit import Guard

    g = Guard()
    r = g.audit(model_v1, model_v2, X_eval, y_eval, X_train=X_treino_v2)
    print(r.plain)          # relatório em linguagem natural
    print(r.verdict)        # APROVADO | ATENCAO | REVISAR | BLOQUEADO | INDETERMINADO
    r.to_html("pt")         # laudo; r.to_dict() é o que o laudo recebe
    r.to_mlflow()           # dict achatado para MLflow/W&B
    r.to_slack()            # payload de mensagem para Slack/Teams
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional


_SEV_ORD = {'APROVADO': 0, 'ATENCAO': 1, 'REVISAR': 2,
            'BLOQUEADO': 3, 'INDETERMINADO': 2}

_VERDICT = [
    (85, 'APROVADO',  'Pode seguir para produção.'),
    (70, 'ATENCAO',   'Pode seguir com monitoramento reforçado.'),
    (55, 'REVISAR',   'Revisar antes de promover.'),
    (0,  'BLOQUEADO', 'Não promover sem investigação.'),
]

_PLAIN = {
    'NONE': "Os dois modelos aprenderam as mesmas relações. Nada a fazer.",
    'IMPORTANCE_COLLAPSE':
        "Uma ou mais variáveis deixaram de influenciar o modelo novo. Esse "
        "padrão é típico de falha no pipeline de dados (coluna virou nula, "
        "join perdido, filtro novo) e não de retreinamento normal.",
    'LOCAL_SHAPE':
        "O modelo novo mudou a forma como usa algumas variáveis: onde antes "
        "o efeito crescia, agora pode decrescer. As métricas gerais não "
        "mostram isso, mas decisões individuais mudam.",
    'MIXED':
        "Houve mudança tanto no uso de variáveis específicas quanto na "
        "estrutura geral de decisão. É o padrão mais preocupante: sugere que "
        "os dados de treino mudaram de forma substancial.",
    'GEOMETRIC':
        "A estrutura geral de decisão mudou, mesmo que nenhuma variável "
        "isolada tenha mudado de forma. Pode indicar deriva nos dados de "
        "treino ou, em casos raros, contaminação.",
    'TOPOLOGICAL':
        "Surgiram (ou desapareceram) grupos de comportamento no modelo novo. "
        "Vale checar se apareceu um segmento de dados inesperado.",
    'OOD_ONLY':
        "Os modelos concordam no dia a dia, mas divergem em cenários "
        "extremos. Atenção se o negócio depende de eventos raros.",
    'INTENSITY_ONLY':
        "A direção dos efeitos se manteve, mas a intensidade mudou. "
        "Limiares operacionais podem precisar de recalibração.",
    'EXPECTED_ARCH_DIFF':
        "As diferenças encontradas são as esperadas ao comparar arquiteturas "
        "distintas. Não é anomalia.",
}


MODOS = ("retraining", "exploratory")


def _checar_modo(mode):
    """`mode` com erro de digitação era aceito em silêncio (BUG 87)."""
    if mode is not None and mode not in MODOS:
        raise ValueError(f"mode deve ser um de {MODOS}; veio {mode!r}")


def _normalizar_entrada(X_eval, y_eval, feature_names, X_train):
    """
    Deixa a entrada no formato que o resto do código supõe (v1.0.4).

    BUG 74: com numpy, `Guard.audit` quebrava com AttributeError (sem nomes)
    ou devolvia APROVADO com SHAP e integridade falhando por dentro (com
    nomes). BUG 69: nomes em ordem diferente das colunas trocavam a culpa
    de variável. BUG 75: `y_eval` com tamanho errado ou com NaN passava
    calado.
    """
    if not isinstance(X_eval, pd.DataFrame):
        arr = np.asarray(X_eval)
        if arr.ndim != 2:
            raise ValueError(f"X_eval precisa ser 2D; veio com forma {arr.shape}")
        cols = (list(feature_names) if feature_names is not None
                and len(feature_names) == arr.shape[1]
                else [f"f{i}" for i in range(arr.shape[1])])
        X_eval = pd.DataFrame(arr, columns=cols)
    cols = list(X_eval.columns)
    if feature_names is None:
        feature_names = cols
    else:
        feature_names = list(feature_names)
        faltam = [f for f in feature_names if f not in cols]
        if faltam:
            raise ValueError(
                f"feature_names tem variáveis que não estão em X_eval: "
                f"{faltam[:5]}. Colunas disponíveis: {cols[:8]}")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names tem nomes repetidos")
        # mesma lista em outra ordem: segue a ordem das colunas, que é a
        # ordem da matriz de atribuição
        if set(feature_names) == set(cols):
            feature_names = cols
    if X_train is not None and not isinstance(X_train, pd.DataFrame):
        arr = np.asarray(X_train)
        if arr.ndim != 2 or arr.shape[1] != len(cols):
            raise ValueError(
                f"X_train tem forma {arr.shape}; esperado (n, {len(cols)}) "
                "com as mesmas colunas de X_eval")
        X_train = pd.DataFrame(arr, columns=cols)
    if y_eval is not None:
        y = pd.Series(np.asarray(y_eval).ravel())
        if len(y) != len(X_eval):
            raise ValueError(
                f"y_eval tem {len(y)} valores e X_eval tem {len(X_eval)} linhas")
        n_nan = int(y.isna().sum())
        if n_nan:
            raise ValueError(f"y_eval tem {n_nan} valores ausentes")
        y_eval = y.set_axis(X_eval.index)
    return X_eval, y_eval, feature_names, X_train


class AuditResult:
    """Resultado de auditoria com saídas prontas para consumo."""

    def __init__(self, raw: dict):
        self.raw = raw
        # `score` mantido para compatibilidade, mas é sinal secundário.
        # Ver LIMITATIONS.md §3 — não use para comparar cenários.
        self.score = raw.get('global_score')
        self.debug_score = self.score
        self.problem_type = raw.get('interpretation', {}).get('problem_type', 'NONE')
        self.mode = raw.get('mode', 'retraining')
        # ── VEREDITO (v0.11.0): só sinais medidos; o índice saiu ─────
        # O índice de saúde decidia os casos em que nenhuma camada
        # disparava. Medido: acompanha a sobreposição das janelas de treino
        # (rho=+0.34), não separa corrupção de retreino saudável dentro de
        # cada janela (p=0.23-0.49) e tem AUC 0.53 contra degradação real.
        # Continua calculado em `debug_score`, fora da decisão.
        #
        # Ordem por confiabilidade medida:
        #   integridade (variável saiu)      -> BLOQUEADO
        #   treino/inferência (dados mudaram) -> ATENCAO
        #   clone compensou a perda           -> ATENCAO
        #   forma mudou, mesmos dados         -> REVISAR
        #   nenhum sinal                      -> APROVADO
        ts = self.raw.get('train_serving') or {}
        self.skew_features = list(ts.get('flagged') or [])
        # BUG 41 (v1.0.3): `skew_features` vazio significava duas coisas
        # diferentes — "checou e está tudo certo" e "a checagem não rodou" — e
        # o laudo escrevia "✓ dados de treino e de uso consistentes" nos dois
        # casos. Num instrumento de decisão, afirmar que um check passou
        # quando ele nem rodou é o pior tipo de erro: a pessoa promove o
        # modelo confiando num visto que não existe. Agora o estado é
        # explícito, e o laudo diz "não verificado" quando é o caso.
        #: 'ok' = rodou | 'erro' = rodou e falhou | 'ausente' = não rodou
        if not self.raw.get('train_serving'):
            self.skew_status = 'ausente'      # X_train não foi passado
        elif ts.get('error'):
            self.skew_status = 'erro'         # a checagem levantou exceção
        else:
            self.skew_status = 'ok'
        self.skew_error = str(ts.get('error') or '')
        self.clone_redistribution = self._colapsos_por_clone(self._colapsos_brutos())
        self.relearning = self._relearning(self._colapsos_brutos())
        l1 = self.raw.get('layer1') or {}
        self.shape_divergences = list((l1.get('sdd') or {}).get('divs') or [])

        # BUG 71 (v1.0.4): falha das camadas centrais não pode virar
        # aprovação. A de integridade é a que decide BLOQUEADO; se ela não
        # rodou, nenhuma ausência de alerta significa nada.
        imp = (l1.get('importance') or {})
        self.integrity_failed = bool(imp.get('error'))
        metodo = str((l1.get('sdd') or {}).get('method', ''))
        self.shape_failed = 'ERROR' in metodo.upper()

        if not l1:
            self.verdict, self.verdict_driver = 'INDETERMINADO', 'sem_medicao'
            self.action = 'Não foi possível medir. Verifique compatibilidade dos modelos.'
        elif self.root_cause_features:
            self.verdict = 'BLOQUEADO'
            self.verdict_driver = ('feature_integrity+train_serving'
                                   if self.skew_features else 'feature_integrity')
            self.action = ('Não promover. Uma ou mais variáveis deixaram de '
                           'influenciar o modelo — verifique o pipeline de '
                           'dados antes de qualquer retreino.')
        elif self.raw.get('model_collapsed'):
            self.verdict, self.verdict_driver = 'BLOQUEADO', 'model_collapse'
            self.action = ('Não promover. O modelo passou a prever praticamente '
                           'o mesmo valor para todos os casos.')
        elif self.integrity_failed:
            self.verdict, self.verdict_driver = 'INDETERMINADO', 'integridade_nao_rodou'
            self.action = ('Não foi possível verificar se alguma variável '
                           'deixou de influenciar o modelo. Não promova com '
                           'base nesta auditoria; veja a lacuna declarada.')
        elif [s for s in self.skew_features
              if s not in self.clone_redistribution and s not in self.relearning]:
            # v0.12.2: a coluna trocada por um clone (ou que o modelo deixou
            # de usar) também aparece como mudança de dados. É a mesma
            # observação, não duas — não conta duas vezes.
            self.verdict, self.verdict_driver = 'ATENCAO', 'train_serving'
            self.action = ('Os dados chegam diferentes do que o modelo viu no '
                           'treino. Retreine e revalide os limites '
                           'operacionais antes de promover.')
        elif self.relearning:
            self.verdict, self.verdict_driver = 'REVISAR', 'relearning'
            self.action = ('O modelo novo deixou de usar uma variável que '
                           'continua chegando normalmente. Não é falha de '
                           'pipeline; confirme se a mudança é desejada.')
        elif self.clone_redistribution:
            self.verdict, self.verdict_driver = 'ATENCAO', 'clone_redistribution'
            self.action = ('Uma variável perdeu peso, mas outra quase idêntica '
                           'assumiu a informação. Confirme se a mudança no '
                           'pipeline foi intencional.')
        elif self.shape_divergences and self.mode == 'exploratory':
            # BUG 87 (v1.0.4): o modo exploratório era aceito e não mudava
            # nada no veredito nem no laudo. Comparando arquiteturas de
            # propósito, mudança de forma é esperada: informa, não alerta.
            # Colapso de variável e dados diferentes continuam valendo.
            self.verdict, self.verdict_driver = 'APROVADO', 'forma_esperada'
            self.action = ('Modo exploratório: as curvas mudaram, o que é '
                           'esperado ao comparar arquiteturas diferentes. '
                           'Nenhuma variável deixou de ser usada.')
        elif self.shape_divergences:
            self.verdict, self.verdict_driver = 'REVISAR', 'shape'
            self.action = ('Com os mesmos dados, o modelo novo reage de forma '
                           'diferente a uma ou mais variáveis. Revise antes de '
                           'promover.')
        elif self.shape_failed:
            self.verdict, self.verdict_driver = 'INDETERMINADO', 'forma_nao_rodou'
            self.action = ('A comparação de comportamento não rodou. Nenhuma '
                           'outra camada encontrou problema, mas isso não '
                           'basta para aprovar.')
        else:
            self.verdict, self.verdict_driver = 'APROVADO', 'nenhum_sinal'
            self.action = 'Nenhuma camada principal encontrou mudança.'

    def _relearning(self, raiz=None) -> dict:
        """
        Separa "a coluna sumiu" de "o modelo deixou de usar" (v0.12.0).

        A camada de integridade via só o peso da atribuição. Medido: em
        retreinos sazonais legítimos o modelo novo parou de usar uma
        defasagem e o veredito foi BLOQUEADO — 1 em 5 retreinos saudáveis.

        Se a variável continua variando normalmente nos dados de treino do
        modelo novo, ela está lá e o modelo escolheu não usá-la: isso é
        reaprendizado, não falha de pipeline. Se perdeu variação (virou
        constante, ou quase), é falha de pipeline.

        Devolve {variavel: motivo} apenas para os casos de reaprendizado.
        Sem X_train não há como distinguir, e nada é rebaixado.
        """
        Xt = self.raw.get('X_train_ref')
        Xe = self.raw.get('X_eval_ref')
        raiz = list(raiz or [])
        if Xt is None or Xe is None or not raiz:
            return {}
        # BUG 73 (v1.0.4): o teste era só "desvio no treino > 10% do desvio
        # na avaliação". Sentinela -999 AUMENTA o desvio; join desalinhado
        # que embaralha a coluna o mantém. Os dois viravam "reaprendizado,
        # não é falha de pipeline" — e ainda tiravam a coluna do ATENCAO.
        # Só é reaprendizado se a checagem treino × uso rodou e NÃO acusou a
        # coluna: aí sabemos que ela chega como no treino.
        if getattr(self, 'skew_status', 'ausente') != 'ok':
            return {}
        # Refinamento (BUG 83, v1.0.4): a primeira versão desta correção
        # negava reaprendizado a QUALQUER coluna acusada. Em série temporal
        # toda coluna deriva um pouco, e dois retreinos saudáveis do ENA (o
        # modelo trocou uma defasagem por outra) viraram BLOQUEADO. A
        # distinção certa é o TIPO de alerta: corrupção de pipeline muda
        # escala, média, faixa, nulos ou níveis; deriva natural muda só a
        # forma da distribuição (PSI). Medido: sentinela -999, imputação por
        # 0 e x1000 sempre trazem SCALE_CHANGE/MEAN_SHIFT/OUT_OF_RANGE; a
        # deriva do ENA trouxe só PSI_WARN/PSI_ALERT (PSI 0,16 e 0,26).
        #
        # MEAN_SHIFT e OUT_OF_RANGE ficam FORA: série com tendência (carga de
        # energia, média móvel numa janela 2023-2025 contra o histórico)
        # produz os dois sem nenhuma corrupção, e um retreino saudável da
        # carga virou BLOQUEADO. SCALE_CHANGE apareceu em toda corrupção
        # medida (sentinela 30% e 97%, imputação por 0, x1000).
        SINAL_DE_PIPELINE = {'CONSTANT_IN_TRAINING', 'NONFINITE',
                             'MISSING_SHIFT', 'NEW_LEVEL', 'NEW_LEVELS',
                             'SCALE_CHANGE', 'RATE_SHIFT', 'SCHEMA_MISMATCH'}
        por_col = {x.get('feature'): set(x.get('alerts') or [])
                   for x in (self.raw.get('train_serving') or {}).get('features', [])}
        acusadas = {f for f in (getattr(self, 'skew_features', []) or [])
                    if por_col.get(f, set()) & SINAL_DE_PIPELINE}
        out = {}
        try:
            for f in raiz:
                if f in acusadas:
                    continue
                if f not in Xt.columns or f not in Xe.columns:
                    continue
                vt, ve = float(Xt[f].std()), float(Xe[f].std())
                if ve <= 1e-12:
                    continue
                # "perdeu variação" = desvio no treino < 10% do desvio nos
                # dados avaliados. Coluna constante dá 0.
                if vt > 0.10 * ve:
                    out[f] = (f'a variável continua variando no treino '
                              f'(desvio {vt:.3g} vs {ve:.3g} na avaliação): '
                              f'o modelo novo deixou de usá-la')
        except Exception:
            return {}       # noqa: lacuna-nao-aplicavel — heurística de aviso
        return out

    def _colapsos_por_clone(self, raiz=None) -> dict:
        """
        Para cada variável apontada como colapso, verifica se existe outra
        quase idêntica (|r| >= 0.95) no conjunto de avaliação. Se existir,
        o "colapso" pode ser só o modelo novo preferindo o clone — os dois
        carregam a mesma informação.

        Devolve {variavel_colapsada: clone}. Heurística conservadora: serve
        para NÃO acusar falha de pipeline sem certeza, não para decidir.
        """
        X = self.raw.get('X_eval_ref')
        raiz = list(raiz or [])
        if X is None or not raiz:
            return {}
        out = {}
        try:
            num = X.select_dtypes('number')
            C = num.corr().abs()
            for f in raiz:
                if f not in C.columns:
                    continue
                # BUG 34 (v0.9.1): o clone só serve de explicação se ele
                # próprio NÃO colapsou. Com temp e atemp quebrados juntos, um
                # desculpava o outro e a causa raiz ficava vazia.
                viz = C[f].drop([f] + [g for g in raiz if g in C.columns
                                       and g != f], errors='ignore')
                if len(viz) and viz.max() >= 0.95:
                    out[f] = str(viz.idxmax())
        except Exception:
            return {}     # noqa: lacuna-nao-aplicavel — heurística de aviso
        return out

    def _sinais_de_imputacao(self) -> list:
        """
        Colunas com marca de imputação: valor sentinela negativo extremo, ou
        um único valor repetido em parte relevante da coluna.

        Heurística deliberadamente conservadora — serve para elevar o peso de
        um aviso, não para decidir nada.
        """
        import numpy as np
        X = self.raw.get('X_eval_ref')
        if X is None:
            return []
        achados = []
        try:
            for col in list(getattr(X, 'columns', []))[:60]:
                s = X[col]
                if not np.issubdtype(getattr(s, 'dtype', None), np.number):
                    continue
                v = s.to_numpy(dtype=float)
                v = v[np.isfinite(v)]
                # v1.72: o piso de 20 fazia a heurística SILENCIAR justo
                # quando a amostra é pequena — ou seja, menos aviso quando
                # o resultado é menos confiável. Medido: com n=15 a
                # severidade caía de 'alta' para normal.
                #
                # Com poucos pontos exigimos proporção maior para compensar
                # a variância, em vez de desligar a checagem.
                if len(v) < 8:
                    continue
                piso = 0.05 if len(v) >= 20 else 0.15
                # sentinela: valor muito fora da faixa, repetido
                for sent in (-999, -9999, -1, 0):
                    prop = float((v == sent).mean())
                    if prop >= piso and (sent < 0 or prop >= 0.30):
                        achados.append(str(col))
                        break
        except Exception:
            return []     # noqa: lacuna-nao-aplicavel — heurística de aviso
        return achados

    # ── o que NÃO foi avaliado ─────────────────────────────────────
    @property
    def not_evaluated(self) -> list:
        """
        Camadas que não rodaram nesta auditoria, e por quê (v1.65).

        Um relatório que omite isto parece completo enquanto mede menos do
        que o leitor supõe. "Não avaliamos X" e "X está ok" são afirmações
        diferentes, e só a primeira é verdadeira quando a camada não rodou.

        Cada item traz `layer`, `reason` e `consequence` — o que fica sem
        cobertura por causa da ausência.
        """
        out = []
        ts = self.raw.get('train_serving')
        if ts is None:
            # v1.71: quando há sinal de imputação nos dados (valor sentinela
            # como -999, ou um único valor repetido em mais de 5% da coluna),
            # a ausência de X_train custa caro: trocar a estratégia de
            # imputação entre versões é rotina de pipeline, e sem esta camada
            # passa como APROVADO.
            #
            # Medido: com nulos imputados por -999 numa versão e pela mediana
            # na outra, o R² cai de 0.897 para 0.334 na avaliação cruzada, e
            # o veredito sem X_train era APROVADO.
            suspeitas = self._sinais_de_imputacao()
            if suspeitas:
                out.append({
                    'layer': 'train/serving consistency',
                    'reason': ('X_train não foi fornecido, e há sinal de '
                               f'imputação em {", ".join(suspeitas[:3])}'),
                    'consequence': ('uma troca de estratégia de imputação '
                                    'entre as versões não seria detectada — '
                                    'passe X_train para verificar'),
                    'severity': 'alta',
                })
            else:
                out.append({
                    'layer': 'train/serving consistency',
                    'reason': ('não recebemos os dados de treino do modelo '
                               'novo (parâmetro X_train)'),
                    'consequence': ('não dá para saber se os dados que chegam '
                                    'ao modelo mudaram em relação ao treino. '
                                    'Passe `X_train=` para verificar'),
                })
        elif ts.get('error'):
            out.append({
                'layer': 'train/serving consistency',
                'reason': ts['error'],
                'consequence': ('esta verificação não rodou; o que ela cobre '
                                'ficou sem exame'),
            })

        # redistribuição entre clones (v0.9.1): não vira causa raiz, mas o
        # leitor precisa saber que houve e por que foi descartada.
        for f, clone in (getattr(self, 'clone_redistribution', {}) or {}).items():
            out.append({
                'layer': 'feature integrity',
                'reason': (f'{f} perdeu peso, mas tem clone quase idêntico '
                           f'({clone}, |r|>=0.95)'),
                'consequence': ('tratado como redistribuição, não como falha '
                                'de pipeline; confirme se a mudança foi '
                                'intencional'),
                'severity': 'media',
            })

        # ressalvas de comparabilidade (v1.71)
        for r in (self.raw.get('_ressalvas') or []):
            out.append(dict(r))

        # falhas registradas pela política de exceção (v1.66): toda camada
        # que levantou exceção durante a auditoria vira lacuna declarada.
        for camada in ('layer1', 'layer2', 'layer3'):
            for falha in (self.raw.get(camada) or {}).get('_failures', []):
                out.append(dict(falha))

        l1 = self.raw.get('layer1', {}) or {}
        m = str((l1.get('sdd') or {}).get('method', ''))

        # v1.71: o método de atribuição entra no resultado porque auditorias
        # feitas com métodos diferentes NÃO são comparáveis em escala
        # (LIMITATIONS.md). Um histórico que mistura as duas produziria
        # "mudanças" que são troca de instrumento, não de modelo.
        if 'ALE' in m.upper() and 'SHAP falhou' in m:
            out.append({
                'layer': 'attribution method',
                'reason': f'SHAP indisponível para este modelo; usou ALE',
                'consequence': ('este modelo não aceita o método rápido de '
                                'atribuição, então usamos outro. Os números '
                                'não são comparáveis com auditorias anteriores '
                                'deste mesmo par'),
            })
        if 'ERROR' in m.upper():
            out.append({
                'layer': 'shape divergence',
                'reason': m,
                'consequence': ('não foi possível comparar as curvas: uma '
                                'mudança de comportamento passaria sem ser '
                                'vista'),
            })

        sem_forma = list((l1.get('sdd') or {}).get('sem_forma') or [])
        if sem_forma:
            out.append({
                'layer': 'interaction-only features',
                'reason': ('sem curva de efeito próprio (o efeito vem de '
                           'interação): ' + ', '.join(sem_forma[:5])),
                'consequence': ('a comparação de forma não se aplica a essas '
                                'variáveis; o peso delas continua sendo '
                                'verificado pela camada de integridade'),
            })

        # fairness nunca roda aqui — decisão documentada em fairness.py
        out.append({
            'layer': 'fairness',
            'reason': ('não verificamos se o erro do modelo mudou para algum '
                       'grupo de pessoas'),
            'consequence': ('duas versões podem errar igual no total e errar '
                            'mais para um grupo específico. Se o seu modelo '
                            'afeta pessoas, meça isso à parte com '
                            '`fairness_evolucao`'),
        })
        return out

    # ── contagens ──────────────────────────────────────────────────
    @property
    def counts(self) -> dict:
        i = self.raw.get('interpretation', {})
        return {k: i.get(k, 0) for k in
                ('n_shape_divs', 'n_slope', 'n_hetero',
                 'n_canary', 'n_topo', 'n_ood')}

    @property
    def affected_features(self) -> List[str]:
        """
        Features afetadas, ordenadas por PROBABILIDADE DE SER A CAUSA RAIZ.

        Achado de UX (v1.49): num caso com apenas `alcohol` corrompida, a
        lista saía como ['residual sugar', 'chlorides', 'pH', 'alcohol'] —
        a causa raiz por ÚLTIMO. As outras três eram divergências de forma
        reais (o modelo redistribui peso), mas são CONSEQUÊNCIA. O usuário
        investigaria quatro features na ordem errada.

        Ordem agora:
          1. colapso de importância — a feature saiu do modelo (causa raiz)
          2. divergência de forma — pode ser consequência da redistribuição
        """
        colapsos, formas = [], []
        l1 = self.raw.get('layer1', {})
        imp = l1.get('importance', {}) or {}
        for r in imp.get('features', []):
            if r.get('alert') == 'IMPORTANCE_COLLAPSE':
                colapsos.append(r['feature'])
        for d in l1.get('sdd', {}).get('divs', []):
            if d['feature'] not in colapsos:
                formas.append(d['feature'])
        seen, uniq = set(), []
        for f in colapsos + formas:
            if f not in seen:
                seen.add(f); uniq.append(f)
        return uniq

    @property
    def root_cause_features(self) -> List[str]:
        """
        Colapsos de importância — as prováveis causas raiz — excluindo os
        que são redistribuição para um clone quase idêntico (v0.9.1).
        """
        brutos = self._colapsos_brutos()
        clones = self._colapsos_por_clone(brutos)
        rele = self._relearning(brutos)
        return [f for f in brutos if f not in clones and f not in rele]

    def to_html(self, lang: str = "pt", decisao: Optional[Dict] = None) -> str:
        """
        Laudo HTML completo, pronto para abrir no navegador (v0.14.0).

        `decisao` é opcional: passe {'trocados': n, 'selecionados': m, 'pct': p}
        quando você tiver medido o impacto na sua regra de decisão. Sem isso o
        laudo declara "não avaliado" em vez de omitir a pergunta.
        """
        from modelcheck.laudo import render
        return render(self.to_dict(decisao), lang=lang)

    def to_dict(self, decisao: Optional[Dict] = None, lang: str = "pt") -> Dict:
        """
        Exatamente o dicionário que o laudo recebe (v1.0.4).

        BUG 55: a CLI montava a sua própria versão, sem clones, sem
        reaprendizado e sem `skew_status`, e o laudo da CLI dizia "✓" onde a
        API dizia ATENCAO. Uma construção só, usada pelos dois.
        """
        # `lang` não altera o dicionário (os campos são os mesmos nos dois
        # idiomas); fica na assinatura por compatibilidade com a CLI.
        if isinstance(decisao, str):          # to_dict(lang) por posição
            decisao, lang = None, decisao
        l3 = self.raw.get("layer3") or {}
        ood = (l3.get("ood") or {}).get("ood_features") or []
        return {
            "verdict": self.verdict,
            "verdict_driver": getattr(self, "verdict_driver", ""),
            # BUG 88: rótulos e modo eram aceitos e não apareciam em lugar
            # nenhum que a pessoa visse
            "label_v1": self.raw.get("label_v1", ""),
            "label_v2": self.raw.get("label_v2", ""),
            "mode": self.mode,
            "model_collapsed": bool(self.raw.get("model_collapsed")),
            "ood_features": list(ood),
            "problem_type": getattr(self, "problem_type", "") or "",
            "root_cause_features": self.root_cause_features,
            "divergences": self.shape_divergences,
            "collapses": [],
            "model_diff": self.model_diff,
            "skew_features": self.skew_features,
            "skew_status": getattr(self, "skew_status", "ausente"),
            "skew_error": getattr(self, "skew_error", ""),
            "clone_redistribution": self.clone_redistribution,
            "relearning": list(self.relearning),
            "decisao": decisao or {},
            "n_eval": len(self.raw.get("X_eval_ref")) if self.raw.get("X_eval_ref") is not None else "—",
            "representation": (self.raw.get("layer1", {}) or {}).get("sdd", {}).get("method", "shap"),
            "not_evaluated": self.not_evaluated,
        }

    @property
    def model_diff(self) -> List[Dict[str, Any]]:
        """
        Diff de comportamento entre as duas versões, por variável (v0.14.0).

        Junta duas perguntas que costumam ser feitas separadas:
        "o modelo continua usando os mesmos sinais?" (peso) e "continua
        usando do mesmo jeito?" (forma). A categoria interessante é
        `peso_estavel_forma_mudou`: olhando só importância, ninguém veria.
        """
        infl = (self.raw.get('layer1') or {}).get('influencia') or []
        formas = {d['feature']: d for d in self.shape_divergences}
        perdidas = set(self.root_cause_features)
        out = []
        for r in infl:
            f = r['feature']
            mudou_forma = f in formas
            d = abs(r['delta_pp'])
            if f in perdidas:
                cat = 'saiu_do_modelo'
            elif r['delta_pp'] >= 5:
                cat = 'ganhou_influencia'
            elif r['delta_pp'] <= -5:
                cat = 'perdeu_influencia'
            elif mudou_forma:
                cat = 'peso_estavel_forma_mudou'
            else:
                cat = 'estavel'
            out.append(dict(r, categoria=cat, forma_mudou=mudou_forma,
                            shape_v1=formas.get(f, {}).get('shape_v1'),
                            shape_v2=formas.get(f, {}).get('shape_v2'),
                            frechet=formas.get(f, {}).get('frechet')))
        return out

    def _colapsos_brutos(self) -> List[str]:
        imp = (self.raw.get('layer1', {}) or {}).get('importance', {}) or {}
        return [r['feature'] for r in imp.get('features', [])
                if r.get('alert') == 'IMPORTANCE_COLLAPSE']

    # ── relatório em linguagem natural ─────────────────────────────
    @property
    def plain(self) -> str:
        """
        Relatório em linguagem de negócio.

        IDIOMA (v1.74): este relatório existe apenas em português. Quando
        MODELCHECK_LANG (ou LANG) indica inglês, o texto vem com um aviso
        no topo — preferimos declarar a lacuna a entregar uma tradução
        parcial que misturaria os dois idiomas.

        Para relatório em inglês use `modelcheck.laudo.render(res, "en")`
        ou a CLI com `--format html`, que são bilíngues.

        O ÍNDICE DE SAÚDE NÃO APARECE POR PADRÃO (v1.47). Cinco medições
        independentes mostraram que é o sinal mais fraco do framework:

          - dominado pela sobreposição das janelas de treino (rho=+0.54)
          - não discrimina dentro de cada estrato (p=0.31 a 0.53)
          - INVERTEU a ordem num caso medido: retreino legítimo 43
            (BLOQUEADO) contra pipeline quebrado 57 (REVISAR)
          - satura perto de zero
          - calibração contra degradação real observada deu AUC 0.533,
            indistinguível de aleatório

        O que discrimina é o par (problem_type, affected_features), que é o
        que este relatório mostra. Para depuração use `plain_debug()`.
        """
        feats = self.affected_features
        # O aviso só aparece quando o idioma foi PEDIDO em inglês —
        # explicitamente ou por MODELCHECK_LANG/LANG. Sem nenhum sinal, o
        # padrão deste relatório é português, e avisar seria ruído.
        import os as _os
        _pediu_en = any(
            (_os.environ.get(v) or "").lower().startswith("en")
            for v in ("MODELCHECK_LANG", "LC_ALL", "LANG"))
        L = []
        if _pediu_en:
            L += ["[This text report is only available in Portuguese. For "
                  "English use modelcheck.laudo.render(result, 'en') or the "
                  "CLI with --format html.]", ""]
        L += [
            f"Resultado: {self.verdict}",
            "",
            _PLAIN.get(self.problem_type, _PLAIN['NONE']),
        ]
        raiz = self.root_cause_features
        if raiz:
            nomes = ", ".join(f"'{f}'" for f in raiz[:3])
            _frase = ("estas variáveis deixaram" if len(raiz) > 1
                      else "esta variável deixou")
            L += ["", f"Comece por: {nomes} — {_frase} de influenciar "
                      f"o modelo."]
            outras = [f for f in feats if f not in raiz]
            if outras:
                on = ", ".join(f"'{f}'" for f in outras[:3])
                ex = f" e outras {len(outras)-3}" if len(outras) > 3 else ""
                L += [f"Também mudaram, possivelmente como consequência: "
                      f"{on}{ex}."]
        elif feats:
            nomes = ", ".join(f"'{f}'" for f in feats[:4])
            extra = f" e outras {len(feats)-4}" if len(feats) > 4 else ""
            L += ["", f"Variáveis envolvidas: {nomes}{extra}."]
        c = self.counts
        detalhes = []
        if c['n_shape_divs']: detalhes.append(f"{c['n_shape_divs']} variável(is) mudaram de padrão")
        if c['n_slope']:      detalhes.append(f"{c['n_slope']} mudaram de intensidade")
        if c['n_canary']:     detalhes.append("a estrutura geral de decisão mudou")
        if c['n_ood']:        detalhes.append("há divergência em cenários extremos")
        if detalhes:
            L += ["", "Em números: " + "; ".join(detalhes) + "."]
        L += ["", f"Recomendação: {self.action}"]

        ne = self.not_evaluated
        if ne:
            L += ["", "Não verificado nesta auditoria:"]
            for item in ne:
                L.append(f"  · {item['layer']} — {item['reason']}")
            L.append("  (o veredito acima não cobre estes pontos)")
        if self.mode == 'exploratory':
            L += ["", "(Modo comparação de arquiteturas: diferenças de padrão "
                      "são esperadas e não contam como falha.)"]
        return "\n".join(L)

    def plain_debug(self) -> str:
        """
        Igual a `plain`, mais o índice de saúde e os scores por camada.

        Apenas para depuração. O índice de saúde NÃO deve ser usado para
        comparar cenários — ver LIMITATIONS.md §3 (rho=+0.54 com sobreposição
        de janelas; inverteu a ordem num caso medido; AUC 0.533 na calibração).
        """
        sc = ("não mensurável" if self.score is None
              else f"{self.score:.0f} de 100")
        L = [self.plain, "",
             f"[depuração] índice de saúde: {sc}",
             "[depuração] não compare este número entre cenários "
             "(ver LIMITATIONS.md §3)"]
        for k, nome in (('layer1', 'forma local'), ('layer2', 'geométrica'),
                        ('layer3', 'extremos')):
            v = (self.raw.get(k) or {}).get('layer_score')
            if v is not None:
                L.append(f"[depuração] camada {nome}: {v:.0f}")
        return "\n".join(L)

    # ── integrações ────────────────────────────────────────────────
    def to_mlflow(self) -> dict:
        """Métricas achatadas para mlflow.log_metrics / wandb.log."""
        c = self.counts
        m = {
            'modelguard.health_score': float(self.score) if self.score is not None else float('nan'),
            'modelguard.shape_divergences': float(c['n_shape_divs']),
            'modelguard.slope_alerts':      float(c['n_slope']),
            'modelguard.canary_alerts':     float(c['n_canary']),
            'modelguard.topology_alerts':   float(c['n_topo']),
            'modelguard.ood_alerts':        float(c['n_ood']),
        }
        for k in ('layer1', 'layer2', 'layer3'):
            v = self.raw.get(k, {}).get('layer_score')
            m[f'modelguard.{k}_score'] = float(v) if v is not None else float('nan')
        return m

    def to_tags(self) -> dict:
        """Tags textuais para MLflow/W&B."""
        return {'modelguard.verdict': self.verdict,
                'modelguard.problem_type': self.problem_type,
                'modelguard.mode': self.mode,
                'modelguard.affected_features': ",".join(self.affected_features[:8])}

    def to_slack(self, model_name: str = "modelo") -> dict:
        """Payload de mensagem para Slack (blocks) ou Teams (text)."""
        emoji = {'APROVADO': ':white_check_mark:', 'ATENCAO': ':warning:',
                 'REVISAR': ':mag:', 'BLOQUEADO': ':no_entry:',
                 'INDETERMINADO': ':question:'}.get(self.verdict, ':warning:')
        head = f"{emoji} ShapeAudit — {model_name}: *{self.verdict}*"
        body = _PLAIN.get(self.problem_type, _PLAIN['NONE'])
        feats = self.affected_features
        if feats:
            body += f"\nVariáveis: {', '.join(feats[:4])}"
        return {
            'text': f"{head}\n{body}\n{self.action}",
            'blocks': [
                {'type': 'section',
                 'text': {'type': 'mrkdwn', 'text': head}},
                {'type': 'section',
                 'text': {'type': 'mrkdwn', 'text': body}},
                {'type': 'context',
                 'elements': [{'type': 'mrkdwn', 'text': self.action}]},
            ],
        }

    @property
    def summary(self) -> dict:
        """Saída primária: veredito, tipo do problema e features afetadas."""
        return {
            'verdict':            self.verdict,
            'problem_type':       self.problem_type,
            'affected_features':  self.affected_features,
            'action':             self.action,
            'mode':               self.mode,
        }

    def __repr__(self):
        feats = self.affected_features[:3]
        return (f"<AuditResult {self.verdict} type={self.problem_type} "
                f"features={feats or '—'}>")


class Guard:
    """
    API de alto nível.

        g = Guard()
        r = g.audit(v1, v2, X_eval, y_eval)
        print(r.plain)

    Parâmetros do construtor:
        mode      'retraining' (default) ou 'exploratory'
        threshold limiar legado de espalhamento, usado só por `quick()`
                  (default 0.30). NÃO é o limiar da camada de forma.
        frechet_threshold
                  limiar da distância de Fréchet na camada de forma. None usa
                  o padrão validado (0.33). Passe aqui o resultado de
                  `calibrar_limiar(...)['limiar']` para usar o limiar do seu
                  domínio.
    """

    def __init__(self, mode: str = "retraining", threshold: float = 0.30,
                 frechet_threshold: Optional[float] = None):
        _checar_modo(mode)
        self.mode = mode
        self.threshold = threshold
        # BUG 80 (v1.0.4): não havia como passar o limiar calibrado ao Guard.
        # `threshold` chegava ao orquestrador e morria lá; a camada de forma
        # usava sempre 0,33. O conselho "calibre com calibrar_limiar" não
        # tinha como ser seguido.
        self.frechet_threshold = frechet_threshold

    def check_features(self, model_v1, model_v2,
                       X_eval: pd.DataFrame,
                       monitored: List[str],
                       feature_names: Optional[List[str]] = None) -> dict:
        """
        Verificação de integridade de features específicas.

        Modo recomendado para pipelines com retreino temporal, onde o
        Health Score global é dominado pela sobreposição das janelas de
        treino (ρ=+0.54) e perde poder discriminativo.

        Informe em `monitored` as features cuja integridade é crítica.
        Medido em 108 pares reais: recall 97%, especificidade 100% em
        todos os níveis de sobreposição (0%, 51%, 84%, 100%).

            r = g.check_features(v1, v2, X, monitored=['ENA_MLT','EAR'])
            if r['alert']: print(r['plain'])
        """
        from modelcheck.importance import ImportanceCollapseDetector
        if isinstance(monitored, str):
            monitored = [monitored]      # aceita string única sem virar lista de chars
        if not monitored:
            raise ValueError(
                "check_features: `monitored` está vazio. Informe as features "
                "cuja integridade é crítica, ex: monitored=['ENA_MLT'].")
        # BUG 89 (v1.0.4): nome inexistente ('Alcohol' em vez de 'alcohol')
        # era aceito e o resultado dizia que a variável "continua
        # influenciando o modelo" — um visto sobre algo que nem foi medido.
        cols = list(getattr(X_eval, 'columns', feature_names or []))
        faltam = [m for m in monitored if m not in cols]
        if faltam:
            raise ValueError(
                f"check_features: variáveis monitoradas que não existem em "
                f"X_eval: {faltam}. Colunas disponíveis: {cols[:10]}")
        icd = ImportanceCollapseDetector()
        res = icd.compare(model_v1, model_v2, X_eval,
                          feature_names, monitored=monitored)
        hits = res['monitored_hits']
        if hits:
            det = ", ".join(f"'{h}'" for h in hits)
            plain = (f"Falha de integridade: {det} deixou de influenciar o "
                     f"modelo novo. Verifique o pipeline de dados dessas "
                     f"variáveis (coluna nula, join perdido, filtro novo).")
        else:
            plain = ("As variáveis monitoradas continuam influenciando o "
                     "modelo. Nenhuma falha de integridade detectada.")
        return {'alert': res['monitored_alert'], 'collapsed': hits,
                'monitored': monitored, 'detail': res, 'plain': plain}

    def audit(self, model_v1, model_v2,
              X_eval: pd.DataFrame,
              y_eval=None,
              feature_names: Optional[List[str]] = None,
              label_v1: str = "versão anterior",
              label_v2: str = "versão nova",
              mode: Optional[str] = None,
              X_train: Optional[pd.DataFrame] = None) -> AuditResult:
        """
        Audita duas versões de modelo.

        X_train : conjunto de treino do modelo novo. Quando fornecido, a
            camada de consistência treino/inferência participa do veredito.
            Sem ele, uma inconsistência de escala entre treino e inferência
            NÃO é detectada por esta chamada — v1.63: antes disso a camada
            existia apenas como detector separado e o veredito a ignorava,
            o que fazia um caso com skew real sair como APROVADO.
        """
        _checar_modo(mode)
        X_eval, y_eval, feature_names, X_train = _normalizar_entrada(
            X_eval, y_eval, feature_names, X_train)
        if getattr(self, 'frechet_threshold', None) is not None:
            from modelcheck.orchestrator import _is_tree
            if not (_is_tree(model_v1) and _is_tree(model_v2)):
                import warnings
                warnings.warn(
                    "frechet_threshold só vale para modelos de árvore (camada "
                    "TreeSHAP). Para estes modelos a camada de forma usa ALE e "
                    "o limiar informado é ignorado.", UserWarning, stacklevel=2)

        from modelcheck._validate import checar_comparabilidade
        _ressalvas = checar_comparabilidade(
            model_v1, model_v2, X_eval,
            feature_names or list(X_eval.columns), caller='Guard.audit',
            y_eval=y_eval)

        from modelcheck.orchestrator import ModelGuardOrchestrator
        orc = ModelGuardOrchestrator(threshold=self.threshold,
                                     frechet_threshold=getattr(self, 'frechet_threshold', None))
        raw = orc.diagnose(model_v1, model_v2, X_eval, y_eval,
                           feature_names, label_v1, label_v2,
                           mode=mode or self.mode)
        raw['X_eval_ref'] = X_eval
        raw['X_train_ref'] = X_train
        raw['_ressalvas'] = _ressalvas
        if X_train is not None:
            try:
                from modelcheck.validation import TrainServingValidator
                fn = list(feature_names or X_eval.columns)
                raw['train_serving'] = TrainServingValidator().check(
                    X_train, X_eval, fn)
            except Exception as e:
                raw['train_serving'] = {'error': f'{type(e).__name__}: {e}',
                                        'flagged': [], 'alerts': []}
        return AuditResult(raw)

    def audit_versions(self, models: list, X_eval: pd.DataFrame,
                       y_eval=None,
                       feature_names: Optional[List[str]] = None,
                       labels: Optional[List[str]] = None) -> dict:
        """
        Audita pares consecutivos de uma sequência de versões.

            reports = g.audit_versions([v1, v2, v3, v4], X_eval, y_eval)
            print(reports['summary'])

        Retorna dict com:
            steps     lista de AuditResult por transição
            scores    scores por transição
            trend     'estável' | 'piorando' | 'melhorando'
            worst     índice da pior transição
            summary   texto em linguagem natural
        """
        if len(models) < 2:
            raise ValueError("audit_versions precisa de ao menos 2 modelos")
        if labels is None:
            labels = [f"v{i+1}" for i in range(len(models))]
        steps, scores = [], []
        for i in range(len(models) - 1):
            r = self.audit(models[i], models[i+1], X_eval, y_eval,
                           feature_names, labels[i], labels[i+1])
            steps.append(r)
            scores.append(r.score if r.score is not None else np.nan)
        arr = np.array(scores, dtype=float)
        valid = arr[~np.isnan(arr)]
        # Tendência: exige >= 4 transições e alpha=0.05.
        # Medido em 3.000 séries aleatórias por tamanho:
        #   n=3  -> 34.2% de falso positivo (inaceitável, bloqueado)
        #   n>=4 com alpha=0.05 -> 5.3%-8.0% (calibrado)
        #   n>=6 com alpha=0.05 -> ~5.7% (nominal)
        trend = 'indeterminado (precisa de 4+ transições)'
        trend_p = None
        if len(valid) >= 4:
            from scipy import stats as _st
            rho, p = _st.spearmanr(np.arange(len(valid)), valid)
            trend_p = round(float(p), 4)
            trend = ('piorando'   if rho < 0 and p < 0.05 else
                     'melhorando' if rho > 0 and p < 0.05 else 'estável')
        worst = int(np.nanargmin(arr)) if len(valid) else 0
        bad = [i for i, s in enumerate(scores)
               if not np.isnan(s) and s < 70]
        L = [f"Auditadas {len(steps)} transições de versão "
             f"({labels[0]} → {labels[-1]}).",
             f"Índice médio de saúde: {np.nanmean(arr):.0f} de 100. "
             f"Tendência ao longo das versões: {trend}."]
        if bad:
            nomes = ", ".join(f"{labels[i]}→{labels[i+1]}" for i in bad[:4])
            L.append(f"Transições que merecem atenção: {nomes}.")
            L.append(f"A mais crítica foi {labels[worst]}→{labels[worst+1]} "
                     f"({steps[worst].verdict}): "
                     f"{steps[worst].problem_type}.")
        else:
            L.append("Nenhuma transição apresentou problema relevante.")
        return {'steps': steps, 'scores': scores, 'labels': labels,
                'trend': trend, 'trend_p': trend_p, 'worst': worst,
                'mean_score': float(np.nanmean(arr)) if len(valid) else None,
                'summary': "\n".join(L)}

    def data_drift_context(self, model_v1, model_v2, X_eval,
                           y_eval=None,
                           feature_names: Optional[List[str]] = None) -> dict:
        """
        Contexto de data/prediction drift, para acompanhar o diagnóstico
        de Shape Instability.

        Shape Instability é ORTOGONAL a data drift: quando v1 e v2 são
        avaliados no mesmo X, P(X) é idêntico por construção. Este método
        existe para o usuário ver o quadro completo, não para substituir
        o diagnóstico estrutural.
        """
        from modelcheck.baselines import run_all_baselines
        bl = run_all_baselines(model_v1, model_v2, X_eval, y_eval,
                               feature_names)
        alerts = [v['tool'] for k, v in bl.items()
                  if isinstance(v, dict) and v.get('alert')]
        return {'raw': bl, 'tools_alerting': alerts,
                'any_alert': bool(alerts),
                'plain': ("Nenhum sinal de deriva de dados ou de performance "
                          "pelos métodos tradicionais."
                          if not alerts else
                          "Métodos tradicionais também alertaram: "
                          + ", ".join(alerts) + ".")}

    def quick(self, model_v1, model_v2, X_eval,
              representation: str = "auto") -> dict:
        """Auditoria rápida: apenas camada de forma local (SDD)."""
        from modelcheck.audit import audit as _audit
        return _audit(model_v1, model_v2, X_eval,
                      representation=representation,
                      threshold=self.threshold, verbose=False)
