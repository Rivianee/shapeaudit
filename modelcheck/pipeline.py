"""
pipeline.py — compor o ModelGuard com ferramentas de deriva (v1.75)

POR QUE ISTO EXISTE
-------------------
Nenhuma equipe usa uma ferramenta só. Usa um pipeline delas, e cada uma
responde uma pergunta diferente:

    Evidently, NannyML, Alibi   "os dados mudaram?"
    ModelGuard                  "o modelo mudou de comportamento?"

As duas perguntas são independentes — é o argumento central do método. Um
modelo pode mudar estruturalmente com os dados idênticos, e os dados podem
mudar sem que o modelo mude.

O que este módulo faz é **combinar os dois sinais numa leitura só**, usando
a mesma matriz de decisão que o `Guard` já usa internamente para a camada
treino/inferência:

    dados mudaram | modelo mudou | leitura
    --------------|--------------|--------------------------------
    não           | sim          | falha de pipeline -> BLOQUEADO
    sim           | não          | mundo mudou -> recalibrar
    sim           | sim          | investigue os dois
    não           | não          | retreino saudável

O QUE ELE NÃO FAZ
-----------------
Não executa a ferramenta externa nem depende dela. Você roda o Evidently (ou
o que usar), extrai quais colunas derivaram, e passa a lista. Manter a
dependência fora é deliberado: `alibi-detect` exige numba, que exige
numpy<=1.26, enquanto o shap exige numpy>=2 — instalar os dois no mesmo
ambiente não funciona hoje, e um `import` opcional esconderia isso.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

# leituras da matriz, por combinação (dados_mudaram, modelo_mudou)
_MATRIZ = {
    (False, True): (
        "PIPELINE_FAILURE",
        "BLOQUEADO",
        "O modelo mudou de comportamento sem que os dados mudassem. É a "
        "assinatura de falha no pipeline: conserte o ETL antes de promover."),
    (True, False): (
        "DATA_SHIFT",
        "ATENCAO",
        "Os dados mudaram e o modelo acompanhou sem quebrar. Retreine e "
        "revalide os limites operacionais — o modelo novo provavelmente "
        "está certo, os limites antigos não."),
    (True, True): (
        "BOTH",
        "BLOQUEADO",
        "Dados e modelo mudaram ao mesmo tempo. Não é possível separar "
        "causa de consequência sem investigar os dois."),
    (False, False): (
        "HEALTHY",
        "APROVADO",
        "Nem os dados nem a estrutura do modelo mudaram. Retreino "
        "saudável."),
}


def combinar(audit_result,
             drifted_columns: Optional[Sequence[str]] = None,
             drift_source: str = "ferramenta de deriva",
             drift_ran: bool = True,
             fairness: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Combina o resultado do ModelGuard com o de uma ferramenta de deriva.

    Parameters
    ----------
    audit_result : o retorno de `Guard().audit(...)`
    drifted_columns : colunas que a ferramenta externa apontou como
        derivadas. Lista vazia significa "rodou e não achou nada".
    drift_source : nome da ferramenta, para constar no relatório
    drift_ran : False quando a ferramenta não foi executada. A diferença
        entre "não achou" e "não rodou" é a mesma que o `not_evaluated`
        carrega — e confundir as duas é o defeito mais frequente deste
        projeto.

    Returns
    -------
    dict com `verdict`, `problem_type`, `reading`, os dois sinais de
    origem, e `not_evaluated` acrescido da lacuna quando a ferramenta não
    rodou.
    """
    # v0.11.0: "modelo mudou" na matriz = uma variável saiu do modelo
    # (integridade). Mudança só de forma não é falha de pipeline: vira
    # SHAPE_CHANGE (REVISAR) quando os dados não mudaram.
    modelo_mudou = bool(getattr(audit_result, "root_cause_features", None))
    # BUG 92 (v1.0.4): no modo exploratório o Guard trata mudança de forma
    # como esperada (APROVADO), mas esta matriz a subia para REVISAR — o
    # mesmo resultado dava vereditos diferentes conforme o caminho.
    forma_mudou = (bool(getattr(audit_result, "shape_divergences", None))
                   and getattr(audit_result, "mode", "") != "exploratory")

    lacunas = list(getattr(audit_result, "not_evaluated", []) or [])
    # v0.9.1: fairness por grupo ao longo do tempo (fairness_evolucao).
    # Sem ela, a lacuna é declarada — o caso do FMMO passa por todas as
    # outras camadas (drift, estrutura, atribuição) sem alarme.
    if fairness is None:
        lacunas.append({"layer": "fairness evolution",
                        "reason": "nenhum atributo protegido informado",
                        "consequence": ("aumento de erro concentrado num grupo "
                                        "não seria detectado")})

    if not drift_ran:
        # Sem o sinal externo, não há matriz: devolvemos o veredito do
        # ModelGuard e declaramos que metade da leitura está faltando.
        lacunas.append({
            "layer": "data drift",
            "reason": f"{drift_source} não foi executada",
            "consequence": ("não é possível distinguir falha de pipeline de "
                            "mudança real nos dados — as duas produzem o "
                            "mesmo veredito aqui"),
            "severity": "alta",
        })
        return {
            "verdict": getattr(audit_result, "verdict", "INDETERMINADO"),
            "problem_type": "MODEL_ONLY",
            "reading": ("Somente o lado do modelo foi avaliado. A leitura "
                        "completa exige também um detector de deriva de "
                        "dados."),
            "model_changed": modelo_mudou,
            "model_root_cause": list(
                getattr(audit_result, "root_cause_features", []) or []),
            "data_drifted": None,
            "drifted_columns": None,
            "drift_source": drift_source,
            "not_evaluated": lacunas,
        }

    cols = list(drifted_columns or [])
    # v0.11.0: a coluna que quebrou (ou foi trocada por um clone) também
    # aparece como "deriva" — é a própria falha, não uma mudança do mundo.
    # Descontamos para não contar a mesma coisa duas vezes.
    clones = dict(getattr(audit_result, "clone_redistribution", {}) or {})
    explicadas = set(getattr(audit_result, "root_cause_features", []) or []) | set(clones)
    dados_mudaram = any(c not in explicadas for c in cols)
    tipo, verdict, leitura = _MATRIZ[(dados_mudaram, modelo_mudou)]
    if tipo == "HEALTHY" and clones:
        tipo, verdict = "CLONE_REDISTRIBUTION", "ATENCAO"
        leitura = ("Uma variável perdeu peso, mas outra quase idêntica assumiu "
                   "a informação. Confirme se a mudança no pipeline foi "
                   "intencional.")
    elif tipo == "HEALTHY" and forma_mudou:
        tipo, verdict = "SHAPE_CHANGE", "REVISAR"
        leitura = ("Com os mesmos dados, o modelo novo reage de forma "
                   "diferente a uma ou mais variáveis. Revise antes de "
                   "promover.")
    # BUG 76 (v1.0.4): a matriz só olhava causa raiz, clones e forma. Um
    # ATENCAO do próprio Guard (treino × uso), um REVISAR por reaprendizado,
    # um BLOQUEADO por colapso do modelo ou um INDETERMINADO viravam
    # APROVADO aqui. A combinação nunca pode ser mais branda que o veredito
    # do Guard: ela acrescenta o sinal externo, não apaga o interno.
    v_guard = getattr(audit_result, "verdict", None)
    if v_guard and _SEV_COMB.get(v_guard, 0) > _SEV_COMB.get(verdict, 0):
        verdict = v_guard
        tipo = tipo if tipo != "HEALTHY" else (
            getattr(audit_result, "verdict_driver", "") or "GUARD").upper()
        leitura = str(getattr(audit_result, "action", "") or leitura)
    grupos_piores = []
    if fairness and fairness.get("status") == "PIOROU":
        grupos_piores = fairness.get("pioras", [])
        if verdict == "APROVADO":
            verdict = "ATENCAO"
        tipo = tipo + "+FAIRNESS_SHIFT"
        g = grupos_piores[0]
        leitura += (f" Além disso, o grupo {g['grupo']} teve {g['metrica'].upper()}"
                    f" de {g['antes']} para {g['depois']}, mais que os demais.")

    return {
        "verdict": verdict,
        "problem_type": tipo,
        "reading": leitura,
        "model_changed": modelo_mudou,
        "model_root_cause": list(
            getattr(audit_result, "root_cause_features", []) or []),
        "data_drifted": dados_mudaram,
        "drifted_columns": cols,
        "drift_source": drift_source,
        "fairness_worsened": grupos_piores,
        "not_evaluated": lacunas,
    }


# ordem de gravidade para combinar vereditos (INDETERMINADO não aprova)
_SEV_COMB = {"APROVADO": 0, "ATENCAO": 1, "REVISAR": 2,
             "INDETERMINADO": 3, "BLOQUEADO": 4}


def exit_code(resultado: Dict[str, Any], fail_on: str = "high") -> int:
    """
    Código de saída para CI, com a mesma convenção da CLI:
    0 limpo, 1 problema encontrado, 2 erro.

    BUG 77 (v1.0.4): INDETERMINADO devolvia 0 (passava no CI) e um `fail_on`
    desconhecido, como 'HIGH', virava 99 e desligava a falha. Agora
    INDETERMINADO é sempre 2 (a auditoria não conseguiu decidir), o nível
    ignora maiúsculas, e valor inválido é erro.
    """
    v = resultado.get("verdict")
    if v == "INDETERMINADO" or v is None:
        return 2
    sev = {"APROVADO": 0, "ATENCAO": 1, "REVISAR": 2, "BLOQUEADO": 3}
    nivel = {"none": 99, "low": 1, "medium": 2, "high": 3}
    chave = str(fail_on).strip().lower()
    if chave not in nivel:
        raise ValueError(f"fail_on deve ser um de {sorted(nivel)}; veio {fail_on!r}")
    if v not in sev:
        return 2
    return 1 if sev[v] >= nivel[chave] else 0


def fairlearn_disparity(y_true, y_pred, sensitive_features,
                        max_difference: float = 0.10,
                        metrics=("selection_rate", "fpr", "fnr")):
    """
    Disparidade entre grupos, via fairlearn, em VÁRIAS métricas.

    v0.9.1 (bug 35): a versão anterior media só a taxa de seleção. No
    cenário do FMMO (Adult, deriva só nas mulheres com rótulo negativo), o
    FPR das mulheres foi de 0.010 para 0.047 — quase 5x — e a disparidade de
    seleção CAIU de 0.166 para 0.132. Mais mulheres aprovadas por engano
    aproximam as taxas de seleção: o dano aparecia como melhora.

    Por isso reportamos seleção, FPR e FNR juntos; o status é o pior deles.
    """
    try:
        import numpy as np
        from fairlearn.metrics import (MetricFrame, selection_rate,
                                       false_positive_rate,
                                       false_negative_rate)
    except ImportError:
        return {"status": "NAO_AVALIAVEL",
                "reason": ("fairlearn não está instalado; "
                           "`pip install modelguard-ml[fairlearn]`"),
                "note": "nada foi verificado sobre disparidade entre grupos"}
    fns = {"selection_rate": selection_rate, "fpr": false_positive_rate,
           "fnr": false_negative_rate}
    try:
        mf = MetricFrame(metrics={k: fns[k] for k in metrics},
                         y_true=np.asarray(y_true), y_pred=np.asarray(y_pred),
                         sensitive_features=np.asarray(sensitive_features))
        bg = mf.by_group
        if len(bg) < 2:
            return {"status": "NAO_AVALIAVEL",
                    "reason": f"apenas {len(bg)} grupo(s) com dados",
                    "note": "comparar exige pelo menos dois grupos"}
        por_metrica = {}
        for k in metrics:
            col = bg[k].dropna()
            dif = float(col.max() - col.min()) if len(col) >= 2 else float("nan")
            por_metrica[k] = {
                "difference": round(dif, 4),
                "by_group": {str(g): round(float(v), 4) for g, v in col.items()},
                "status": "DISPARITY" if dif > max_difference else "OK"}
        pior = max(por_metrica, key=lambda k: por_metrica[k]["difference"]
                   if por_metrica[k]["difference"] == por_metrica[k]["difference"] else -1)
        sr = por_metrica.get("selection_rate", {})
        return {
            "status": "DISPARITY" if any(v["status"] == "DISPARITY"
                                         for v in por_metrica.values()) else "OK",
            "metrics": por_metrica,
            "worst_metric": pior,
            # compatibilidade com v0.9.0
            "difference": sr.get("difference"),
            "by_group": sr.get("by_group"),
            "max_allowed": max_difference,
            "note": ("taxa de seleção sozinha pode esconder aumento de falso "
                     "positivo num grupo; compare as três métricas e a "
                     "evolução entre versões, não só o valor atual"),
        }
    except Exception as e:
        return {"status": "NAO_AVALIAVEL",
                "reason": f"{type(e).__name__}: {e}"[:140],
                "note": "nada foi verificado sobre disparidade entre grupos"}


def fairness_evolucao(y_antes, pred_antes, grupo_antes,
                      y_depois, pred_depois, grupo_depois,
                      max_aumento: float = 0.02, alpha: float = 0.01):
    """
    Compara cada grupo CONSIGO MESMO entre dois momentos.

    No cenário do FMMO o FPR das mulheres foi de 0.010 para 0.047 e toda
    métrica de disparidade num momento só MELHOROU; o dano só aparece na
    evolução do próprio grupo.

    BUG 36 (v0.9.1): a primeira versão usava só um limiar fixo (+0.02). Em
    retreinos SAUDÁVEIS deu alarme em 45% dos casos no COMPAS e 95% no
    German Credit — o FPR de um grupo pequeno (68 pessoas) oscila muito
    mais que 0.02 só por reamostragem. Limiar fixo contra quantidade cuja
    variância depende do tamanho do grupo: classe 2 do catálogo.

    Agora exige, para cada grupo e métrica:
      1. teste pareado (McNemar exato, unilateral) quando as mesmas pessoas
         são avaliadas nos dois momentos; senão, teste de duas proporções;
         corrigido por Bonferroni no número de testes;
      2. que o grupo piore MAIS que o restante (diferença em diferenças) —
         um retreino saudável desloca todos os grupos juntos;
      3. que o excesso passe de `max_aumento`.
    """
    import numpy as np
    from scipy import stats
    ya, pa, ga = map(np.asarray, (y_antes, pred_antes, grupo_antes))
    yd, pd_, gd = map(np.asarray, (y_depois, pred_depois, grupo_depois))
    pareado = (len(ya) == len(yd) and np.array_equal(ya, yd)
               and np.array_equal(ga, gd))
    grupos = [g for g in np.unique(ga) if (gd == g).any()]
    if len(grupos) < 2:
        return {"status": "NAO_AVALIAVEL",
                "reason": "é preciso pelo menos dois grupos nos dois momentos"}
    n_testes = 2 * len(grupos)
    pioras, detalhe = [], []
    for met, alvo in (("fpr", 0), ("fnr", 1)):
        def taxa(y, p, m):
            sel = m & (y == alvo)
            if sel.sum() == 0:
                return None, 0
            err = (p[sel] == 1) if alvo == 0 else (p[sel] == 0)
            return float(err.mean()), int(sel.sum())
        for g in grupos:
            ta, na = taxa(ya, pa, ga == g); td, nd = taxa(yd, pd_, gd == g)
            oa, _ = taxa(ya, pa, ga != g);  od, _ = taxa(yd, pd_, gd != g)
            if None in (ta, td, oa, od) or min(na, nd) < 10:
                continue
            excesso = (td - ta) - (od - oa)
            if pareado:
                sel = (ga == g) & (ya == alvo)
                ea = (pa[sel] == 1) if alvo == 0 else (pa[sel] == 0)
                ed = (pd_[sel] == 1) if alvo == 0 else (pd_[sel] == 0)
                b, cc = int((~ea & ed).sum()), int((ea & ~ed).sum())
                p = stats.binomtest(b, b + cc, .5, alternative="greater").pvalue \
                    if b + cc > 0 else 1.0
            else:
                k = np.array([td * nd, ta * na]); nn = np.array([nd, na])
                pp = k.sum() / nn.sum()
                se = np.sqrt(pp * (1 - pp) * (1 / nd + 1 / na)) or 1e-12
                p = 1 - stats.norm.cdf((td - ta) / se)
            reg = {"grupo": str(g), "metrica": met, "antes": round(ta, 4),
                   "depois": round(td, 4), "excesso_sobre_outros": round(excesso, 4),
                   "p": round(float(p), 5), "n": nd}
            detalhe.append(reg)
            if p < alpha / n_testes and excesso > max_aumento:
                pioras.append(reg)
    return {"status": "PIOROU" if pioras else "OK", "pioras": pioras,
            "detalhe": detalhe, "pareado": pareado,
            "note": ("compara cada grupo consigo mesmo e com os demais; exige "
                     "rótulos nos dois momentos. Grupos com menos de 10 "
                     "casos da classe relevante não são testados.")}


def evidently_drifted_columns(report_dict: Dict[str, Any]) -> List[str]:
    """
    Extrai os nomes das colunas derivadas de um `Report(...).dict()` do
    Evidently.

    Testado com a 0.7.x. O formato do dicionário mudou entre versões
    maiores, então esta função tolera ausência em vez de quebrar: se o
    formato não for reconhecido, devolve lista vazia e o chamador deve
    tratar como "não rodou" em vez de "não achou".
    """
    import re
    achadas: List[str] = []
    try:
        total = None
        por_coluna = []            # (nome, valor, metodo, limiar)
        for m in report_dict.get("metrics", []):
            nome_m = str(m.get("metric_name") or m.get("metric_id") or "")
            val = m.get("value")
            if "DriftedColumnsCount" in nome_m and isinstance(val, dict):
                total = int(val.get("count", 0))
                continue
            if "ValueDrift" not in nome_m or not isinstance(val, (int, float)):
                continue
            col = re.search(r"column=([^,)]+)", nome_m)
            met = re.search(r"method=([^,)]+)", nome_m)
            lim = re.search(r"threshold=([0-9.eE+-]+)", nome_m)
            if col:
                por_coluna.append((
                    col.group(1).strip(),
                    float(val),
                    (met.group(1).strip() if met else ""),
                    (float(lim.group(1)) if lim else None)))

        if total is None:
            return []

        # O Evidently escolhe o teste por coluna, e a direção do limiar
        # depende do teste: em p-valor deriva é ABAIXO do limiar; em
        # distância (Wasserstein, Jensen-Shannon, PSI) é ACIMA. Ordenar por
        # magnitude sem olhar o método apontava a coluna errada — um
        # p-valor de 1.0 (sem deriva) parecia "maior" que 1e-40 (deriva).
        for nome, val, metodo, limiar in por_coluna:
            if limiar is None:
                continue
            p_valor = "p_value" in metodo.lower()
            if (val < limiar) if p_valor else (val > limiar):
                achadas.append(nome)

        if not achadas and total > 0:
            # limiares não expostos: reporta a contagem sem inventar nomes
            achadas = [f"<{total} coluna(s), nomes não expostos>"]
    except Exception:
        return []     # noqa: lacuna-nao-aplicavel — formato não reconhecido
    return sorted(set(achadas))
