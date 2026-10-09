"""
_validate.py — Validação de entrada compartilhada (v1.47)

BUG 17: sondando a API pública com entradas inválidas, vários detectores
devolviam resultado NORMAL em vez de erro:
  - `features=[]`  -> resultado como se tudo estivesse bem
  - `X` vazio      -> frechet_distance = nan, score 100
  - `X` com 1 linha-> score 100
  - `X` como numpy -> AttributeError cru
  - modelo None    -> AttributeError cru

Um resultado plausível para entrada inválida é a pior falha possível: o
usuário confia num número que não significa nada.
"""
import numpy as np
import pandas as pd
from typing import List, Optional

MIN_ROWS = 10


def validate_inputs(model_v1, model_v2,
                    X, feature_names: Optional[List[str]] = None,
                    min_rows: int = MIN_ROWS,
                    caller: str = "detector"):
    """
    Normaliza e valida. Devolve (X_df, feature_names).
    Levanta ValueError/TypeError com instrução acionável.
    """
    if model_v1 is None or model_v2 is None:
        qual = "model_v1" if model_v1 is None else "model_v2"
        raise ValueError(
            f"{caller}: {qual} é None. Passe dois modelos treinados com "
            f"método predict().")
    for nome, m in (("model_v1", model_v1), ("model_v2", model_v2)):
        if not hasattr(m, "predict"):
            raise TypeError(
                f"{caller}: {nome} ({type(m).__name__}) não tem método "
                f"predict(). Passe um modelo treinado.")

    if X is None:
        raise ValueError(
            f"{caller}: X é None. Passe um DataFrame com as features de "
            f"avaliação.")
    if not isinstance(X, pd.DataFrame):
        arr = np.asarray(X)
        if arr.ndim != 2:
            raise ValueError(
                f"{caller}: X deve ser 2D, recebido ndim={arr.ndim}.")
        X = pd.DataFrame(arr, columns=[f"f{i}" for i in range(arr.shape[1])])

    if len(X) == 0:
        raise ValueError(
            f"{caller}: X está vazio (0 linhas). Não é possível avaliar.")
    if len(X) < min_rows:
        raise ValueError(
            f"{caller}: X tem apenas {len(X)} linhas (mínimo {min_rows}). "
            f"Resultados com n menor não são interpretáveis.")

    if feature_names is None:
        feature_names = list(X.columns)
    if isinstance(feature_names, str):
        feature_names = [feature_names]
    feature_names = list(feature_names)
    if not feature_names:
        raise ValueError(
            f"{caller}: feature_names está vazio. Passe ao menos uma "
            f"feature, ou None para usar todas as colunas de X.")
    faltando = [f for f in feature_names if f not in X.columns]
    if faltando:
        raise ValueError(
            f"{caller}: features ausentes em X: {faltando[:5]}. "
            f"Colunas disponíveis: {list(X.columns)[:8]}.")
    return X, feature_names


# ══ Condições que invalidam ou enfraquecem a comparação (v1.71) ══════
#
# Encontradas por varredura sistemática depois do caso da imputação. A
# classe é sempre a mesma: a ferramenta produz um veredito num caso onde
# o veredito não significa o que o leitor supõe.
#
#   recusa   a comparação não é válida — nenhum número salva
#   aviso    é válida mas frágil; o resultado sai com a ressalva

N_MINIMO_CONFIAVEL = 60     # abaixo disto a classificação de forma oscila
N_MINIMO_CLASSE = 20        # abaixo disto nada se conclui sobre a classe


def natureza_da_saida(model, X):
    """'classe', 'continuo' ou None se não for possível determinar."""
    import numpy as np
    try:
        p = np.asarray(model.predict(X[:200]), dtype=float)
    except Exception:
        return None
    if p.ndim > 1:
        p = p.ravel()
    p = p[np.isfinite(p)]
    if len(p) == 0:
        return None
    unicos = np.unique(p)
    if len(unicos) <= max(2, len(p) * 0.02) and np.all(unicos == unicos.astype(int)):
        return 'classe'
    return 'continuo'


def checar_comparabilidade(model_v1, model_v2, X, feature_names=None,
                           caller='audit', y_eval=None):
    """
    Levanta ValueError quando a comparação não é válida; devolve a lista de
    ressalvas quando é válida mas frágil.
    """
    import numpy as np
    ressalvas = []

    # ── recusa: ordem de colunas diferente da vista no treino ────────
    # BUG 32 (v1.73): o sklearn recusa `X[colunas_reordenadas]` com
    # "Feature names must be in the same order as they were in fit". Alguma
    # camada engolia esse erro e o veredito saía mesmo assim — apontando a
    # variável ERRADA (corrompemos x0, o laudo dizia x3).
    #
    # Reordenar colunas é acidente comum: um `select` diferente, um join
    # que muda a ordem, um CSV regravado.
    for rotulo, m in (('v1', model_v1), ('v2', model_v2)):
        esperado = getattr(m, 'feature_names_in_', None)
        if esperado is None:
            continue
        recebido = list(X.columns)
        esperado = [str(f) for f in esperado]
        if recebido == esperado:
            continue
        if sorted(recebido) == sorted(esperado):
            raise ValueError(
                f"{caller}: as colunas de X estão em ordem diferente da "
                f"usada no treino do modelo {rotulo}. O modelo casa colunas "
                f"por posição, então isto entrega valores trocados e o "
                f"diagnóstico apontaria a variável errada. Reordene com "
                f"X[list(model.feature_names_in_)].")
        faltando = [f for f in esperado if f not in recebido]
        sobrando = [f for f in recebido if f not in esperado]
        raise ValueError(
            f"{caller}: as colunas de X não correspondem às do treino do "
            f"modelo {rotulo}. Faltando: {faltando[:4] or '—'}. "
            f"Sobrando: {sobrando[:4] or '—'}.")

    # ── recusa: naturezas de saída diferentes ────────────────────────
    n1, n2 = natureza_da_saida(model_v1, X), natureza_da_saida(model_v2, X)
    if n1 and n2 and n1 != n2:
        raise ValueError(
            f"{caller}: os dois modelos produzem saídas de naturezas "
            f"diferentes ({n1} e {n2}). Um regressor e um classificador não "
            f"são comparáveis — as atribuições estão em unidades distintas e "
            f"qualquer veredito seria sobre a diferença de unidade, não sobre "
            f"o modelo. Compare modelos do mesmo tipo.")

    # ── aviso: conjunto de avaliação pequeno ─────────────────────────
    if len(X) < N_MINIMO_CONFIAVEL:
        ressalvas.append({
            'layer': 'evaluation size',
            'reason': f'apenas {len(X)} amostras de avaliação',
            'consequence': (f'abaixo de {N_MINIMO_CONFIAVEL} a classificação '
                            f'de forma não é estável; o veredito pode mudar '
                            f'com outra amostra'),
            'severity': 'alta',
        })

    # ── aviso: classe rara demais para sustentar conclusão ───────────
    # v1.73: um classificador avaliado com 2 exemplos de uma das classes
    # produzia APROVADO sem ressalva. Com 2 amostras nada pode ser
    # concluído sobre o comportamento naquela classe — e o leitor de um
    # APROVADO supõe que houve verificação.
    if y_eval is not None:
        try:
            vc = pd.Series(y_eval).value_counts()
            if 1 < len(vc) <= 20:                # parece classificação
                # o repr de um np.int64 polui a mensagem do usuário
                _r = vc.index[-1]
                rara = _r.item() if hasattr(_r, 'item') else _r
                n_rara = int(vc.iloc[-1])
                if n_rara < N_MINIMO_CLASSE:
                    ressalvas.append({
                        'layer': 'class representation',
                        'reason': (f'a classe {rara!r} tem apenas {n_rara} '
                                   f'exemplos na avaliação'),
                        'consequence': ('nada pode ser concluído sobre o '
                                        'comportamento do modelo nessa '
                                        'classe'),
                        'severity': 'alta',
                    })
        except Exception:
            pass     # noqa: lacuna-nao-aplicavel — heurística de aviso

    # ── aviso: avaliação fora da faixa vista no treino ───────────────
    fora = []
    for m in (model_v1, model_v2):
        faixa = getattr(m, '_modelcheck_train_range', None)
        if faixa:
            for col, (lo, hi) in faixa.items():
                if col in X.columns:
                    v = X[col].to_numpy(dtype=float)
                    v = v[np.isfinite(v)]
                    if len(v) and (v.min() > hi or v.max() < lo):
                        fora.append(col)
    if fora:
        ressalvas.append({
            'layer': 'evaluation range',
            'reason': f'avaliação fora da faixa de treino em {sorted(set(fora))[:3]}',
            'consequence': ('os dois modelos estão extrapolando; a "concordância" '
                            'entre eles não diz nada sobre o comportamento na '
                            'faixa de operação'),
            'severity': 'alta',
        })
    return ressalvas
