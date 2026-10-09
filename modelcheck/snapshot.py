"""
snapshot.py — persistência do estado estrutural de um modelo (v1.62)

O PROBLEMA QUE ISTO RESOLVE
---------------------------
Até aqui a biblioteca comparava dois modelos carregados em memória ao mesmo
tempo. Isso funciona num notebook, mas não em produção: o modelo de janeiro
não está mais carregado em julho, e retreiná-lo só para comparar é caro
quando é possível, e impossível quando os dados de treino mudaram.

Um snapshot guarda o que a auditoria precisa do modelo antigo — os perfis de
atribuição por variável, as importâncias relativas e os metadados — num
arquivo pequeno. O modelo em si não é necessário depois disso.

O QUE UM SNAPSHOT CONTÉM
------------------------
    shapes        forma classificada por variável
    bins          perfil binado (as médias por bin)
    importance    importância relativa por variável
    meta          data, n de avaliação, método de atribuição, versão

O QUE ELE NÃO CONTÉM
--------------------
Nem o modelo, nem os dados. Um snapshot não permite reconstruir nenhum dos
dois — é uma descrição estrutural, não uma cópia. Isso importa quando os
dados são sensíveis: o snapshot pode ser versionado em repositório onde o
dataset não poderia.

LIMITAÇÃO
---------
Snapshots só são comparáveis se calculados sobre o MESMO conjunto de
avaliação e com o MESMO método de atribuição. `load()` verifica os dois e
recusa a comparação quando divergem, porque comparar perfis calculados em
conjuntos diferentes mede a diferença entre os conjuntos, não entre os
modelos.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ══ POLÍTICA DE COMPATIBILIDADE (v1.68) ═══════════════════════════════
#
# Um histórico de auditoria atravessa versões da ferramenta. Um snapshot
# gravado em janeiro precisa abrir em julho — recusá-lo perde o histórico
# exatamente quando ele é necessário.
#
# REGRAS
#   aditivo é seguro   campo novo entra como OPCIONAL; o schema não muda
#   renomear           alias interno por uma versão, depois deprecia
#   remover/quebrar    só com SCHEMA_VERSION nova + migração registrada
#
# `load()` aceita qualquer versão <= SCHEMA_VERSION e normaliza para o
# esquema canônico interno. Versões futuras são recusadas, porque ler um
# formato que ainda não existe exige adivinhar.
SCHEMA_VERSION = 1

#: aceita leitura destas versões; cada uma com sua função de migração
_MIGRACOES = {}          # {versao_origem: func(dict) -> dict na versao+1}

#: campos exigidos pelo esquema canônico interno. Tudo fora daqui é
#: opcional por definição — é o que torna "aditivo é seguro" verificável.
CANONICO = ("schema_version", "label", "created_utc", "features",
            "shapes", "bins", "importance", "meta")

# compatibilidade com o nome antigo, usado até a v1.67
SNAPSHOT_VERSION = SCHEMA_VERSION


# ──────────────────────────────────────────────────────────────────────
def _hash_eval(X: pd.DataFrame, feature_names: List[str]) -> str:
    """
    Identidade do conjunto de avaliação.

    Usamos o nome das colunas, o número de linhas e um resumo estatístico
    por coluna — não os dados. Dois conjuntos com o mesmo resumo são
    tratados como o mesmo; é uma verificação de sanidade, não criptográfica.
    """
    h = hashlib.sha256()
    h.update(",".join(map(str, feature_names)).encode())
    h.update(str(len(X)).encode())
    for f in feature_names:
        col = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
        col = col[np.isfinite(col)]
        if len(col) == 0:
            h.update(b"empty")
            continue
        resumo = (float(np.mean(col)), float(np.std(col)),
                  float(np.min(col)), float(np.max(col)))
        h.update(np.array(resumo).round(6).tobytes())
    return h.hexdigest()[:16]


def _perfis(model, X: pd.DataFrame, feature_names: List[str],
            representation: str = "shap", n_sample: Optional[int] = None):
    """Calcula formas, bins e importâncias — o conteúdo do snapshot."""
    from modelcheck.shape import _shape_profile

    Xe = X if n_sample is None or n_sample >= len(X) else \
        X.sample(n_sample, random_state=0)

    sv = None
    if representation == "shap":
        try:
            import shap
            s = shap.TreeExplainer(model).shap_values(Xe[feature_names])
            s = np.asarray(s, dtype=float)
            if s.ndim == 3:
                s = s[:, :, 1] if s.shape[2] > 1 else s[:, :, 0]
            sv = s
        except Exception:
            sv = None

    if sv is None:                       # caminho model-agnostic
        from modelcheck.ale_fast import ale_curve_fast
        from modelcheck.shape import _shape_profile_small_n
        shapes, bins, imp = {}, {}, {}
        for f in feature_names:
            try:
                grid, eff = ale_curve_fast(model, Xe, f)
                p = _shape_profile_small_n(np.asarray(grid),
                                           np.asarray(eff), min_n=3)
                shapes[f] = p["shape"] if p else None
                bins[f] = [float(v) for v in np.asarray(eff)]
                imp[f] = float(np.abs(np.asarray(eff)).mean())
            except Exception:
                shapes[f], bins[f], imp[f] = None, [], 0.0
        tot = sum(imp.values()) or 1.0
        return (shapes, bins, {k: v / tot for k, v in imp.items()},
                "ale", len(Xe))

    shapes, bins, imp = {}, {}, {}
    for i, f in enumerate(feature_names):
        p = _shape_profile(Xe[f].to_numpy(), sv[:, i])
        shapes[f] = p["shape"] if p else None
        # BUG 31 (v1.72): o perfil de amostra pequena (_shape_profile_small_n)
        # não devolve "means" — só "shape" e "confidence". Assumir a chave
        # quebrava o snapshot com n < 30. Mesma família do bug 16, que era
        # KeyError em 'monotonic'.
        # `or []` seria ambíguo aqui: "means" é um ndarray, e o `or` de
        # numpy levanta ValueError em vez de testar vazio.
        _m = p.get("means") if p else None
        bins[f] = [] if _m is None else [float(x) for x in _m]
        imp[f] = float(np.abs(sv[:, i]).mean())
    tot = sum(imp.values()) or 1.0
    return shapes, bins, {k: v / tot for k, v in imp.items()}, "shap", len(Xe)


# ──────────────────────────────────────────────────────────────────────
def save(model, X: pd.DataFrame, path: str,
         feature_names: Optional[List[str]] = None,
         label: Optional[str] = None,
         representation: str = "shap",
         n_sample: Optional[int] = None) -> Dict[str, Any]:
    """
    Grava o estado estrutural de `model` avaliado em `X`.

    Parameters
    ----------
    model : objeto com `predict`
    X : conjunto de avaliação — o MESMO precisará ser usado na comparação
    path : caminho do arquivo (.json)
    label : nome legível da versão, ex. "v1" ou "2026-01-producao"

    Returns
    -------
    dict com o snapshot gravado.
    """
    from modelcheck._validate import validate_inputs
    X, feature_names = validate_inputs(model, model, X, feature_names,
                                       caller="snapshot.save")

    shapes, bins, imp, metodo, n_eval = _perfis(
        model, X, feature_names, representation, n_sample)

    snap = {
        "schema_version": SCHEMA_VERSION,
        # alias do nome usado até a v1.67; removido quando SCHEMA_VERSION
        # subir para 2.
        "snapshot_version": SCHEMA_VERSION,
        "label": label or os.path.basename(path).rsplit(".", 1)[0],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "features": list(feature_names),
        "shapes": shapes,
        "bins": bins,
        "importance": imp,
        "meta": {
            "representation": metodo,
            "n_eval": int(n_eval),
            "eval_fingerprint": _hash_eval(X, feature_names),
            "model_class": type(model).__name__,
        },
    }
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=2, ensure_ascii=False)
    return snap


def load(path: str) -> Dict[str, Any]:
    """Lê um snapshot, validando a versão do formato."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"snapshot não encontrado: {path}. Use "
            f"`modelcheck snapshot save` para criar um.")
    try:
        with open(path, encoding="utf-8") as fh:
            snap = json.load(fh)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"snapshot corrompido ou inválido: {path} não é JSON válido "
            f"(linha {e.lineno}, coluna {e.colno}). O arquivo pode ter sido "
            f"truncado durante a gravação.") from e
    except UnicodeDecodeError as e:
        raise ValueError(
            f"snapshot corrompido ou inválido: {path} não pôde ser lido como "
            f"texto UTF-8.") from e
    if not isinstance(snap, dict):
        raise ValueError(
            f"snapshot corrompido ou inválido: {path} contém "
            f"{type(snap).__name__}, esperava um objeto JSON.")
    return normalizar(snap, origem=path)


def normalizar(snap: Dict[str, Any], origem: str = "") -> Dict[str, Any]:
    """
    Converte um snapshot de qualquer versão suportada para o esquema
    canônico interno.

    É aqui que o `report` histórico se apoia: snapshots gravados em versões
    diferentes da ferramenta passam por esta função e saem comparáveis. Sem
    isso, comparar janeiro com julho exigiria que o formato nunca mudasse.
    """
    v = snap.get("schema_version", snap.get("snapshot_version"))
    onde = f" ({origem})" if origem else ""
    if v is None:
        raise ValueError(
            f"snapshot sem versão de esquema{onde}. O arquivo pode estar "
            f"corrompido ou não ser um snapshot.")
    if not isinstance(v, int):
        raise ValueError(f"schema_version inválida: {v!r}{onde}")
    if v > SCHEMA_VERSION:
        raise ValueError(
            f"snapshot em esquema v{v}, esta instalação lê até "
            f"v{SCHEMA_VERSION}{onde}. Atualize o modelcheck para lê-lo — "
            f"ler um formato futuro exigiria adivinhar o significado dos "
            f"campos.")

    # aplica migrações em cadeia até chegar na versão corrente
    while v < SCHEMA_VERSION:
        mig = _MIGRACOES.get(v)
        if mig is None:
            raise ValueError(
                f"não há migração registrada de v{v} para v{v + 1}{onde}. "
                f"Isto é um defeito: toda mudança de SCHEMA_VERSION deve "
                f"registrar a migração em _MIGRACOES.")
        try:
            snap = mig(snap)
        except Exception as e:
            # sem estado parcial: um snapshot meio migrado está num formato
            # que não existe em versão nenhuma.
            raise ValueError(
                f"falha ao migrar o esquema de v{v} para v{v + 1}{onde}: "
                f"{type(e).__name__}: {e}. O snapshot não foi alterado."
            ) from e
        v += 1
    snap["schema_version"] = SCHEMA_VERSION

    faltando = [k for k in CANONICO if k not in snap]
    if faltando:
        raise ValueError(
            f"snapshot sem campos canônicos: {faltando}{onde}. "
            f"Regrave com `modelcheck snapshot save`.")
    return snap


def listar(diretorio: str = ".") -> List[Dict[str, Any]]:
    """Lista os snapshots de um diretório, do mais recente para o mais antigo."""
    out = []
    for nome in sorted(os.listdir(diretorio)):
        if not nome.endswith((".json", ".snapshot")):
            continue
        try:
            s = load(os.path.join(diretorio, nome))
        except Exception:
            continue                       # não é um snapshot; ignora
        out.append({
            "file": nome,
            "label": s.get("label"),
            "created": s.get("created_utc"),
            "features": len(s.get("features", [])),
            "n_eval": s.get("meta", {}).get("n_eval"),
            "representation": s.get("meta", {}).get("representation"),
        })
    return sorted(out, key=lambda r: r["created"] or "", reverse=True)


# ──────────────────────────────────────────────────────────────────────
def diff(snapshot_path: str, model, X: pd.DataFrame,
         feature_names: Optional[List[str]] = None,
         monitored: Optional[List[str]] = None,
         tau: float = 0.30,
         min_rel_importance: float = 0.01,
         strict_eval: bool = True) -> Dict[str, Any]:
    """
    Compara um modelo atual contra um snapshot gravado.

    Esta é a operação que o `audit()` não consegue fazer: o modelo antigo não
    precisa existir.

    Parameters
    ----------
    strict_eval : se True (padrão), recusa comparar quando o conjunto de
        avaliação difere do usado no snapshot. Comparar perfis calculados em
        conjuntos diferentes mede a diferença entre os conjuntos.
    """
    from modelcheck._validate import validate_inputs
    snap = load(snapshot_path)
    X, feature_names = validate_inputs(model, model, X, feature_names,
                                       caller="snapshot.diff")

    faltando = [f for f in snap["features"] if f not in feature_names]
    se_o_snapshot_nao_tem = [f for f in feature_names
                             if f not in snap["features"]]
    if faltando or se_o_snapshot_nao_tem:
        raise ValueError(
            f"conjunto de variáveis diferente do snapshot. "
            f"Só no snapshot: {faltando[:5] or '—'}. "
            f"Só no modelo atual: {se_o_snapshot_nao_tem[:5] or '—'}.")

    fp = _hash_eval(X, feature_names)
    mesmo_eval = fp == snap["meta"]["eval_fingerprint"]
    if strict_eval and not mesmo_eval:
        raise ValueError(
            "o conjunto de avaliação difere do usado no snapshot "
            f"(fingerprint {fp} vs {snap['meta']['eval_fingerprint']}). "
            "Perfis calculados em conjuntos diferentes não são comparáveis: "
            "a diferença medida seria entre os conjuntos, não entre os "
            "modelos. Use o mesmo conjunto, ou passe strict_eval=False se "
            "souber o que está fazendo.")

    # v1.72: as guardas de comparabilidade estavam só no Guard.audit. Um
    # diff com 15 amostras não avisava — mesmo padrão do bug 28,
    # implementações paralelas com contratos diferentes.
    from modelcheck._validate import checar_comparabilidade
    ressalvas = checar_comparabilidade(model, model, X, feature_names,
                                       caller="snapshot.diff")

    shapes2, bins2, imp2, metodo2, n2 = _perfis(
        model, X, feature_names, snap["meta"]["representation"], None)

    if metodo2 != snap["meta"]["representation"]:
        raise ValueError(
            f"método de atribuição diferente: snapshot usa "
            f"'{snap['meta']['representation']}', modelo atual só permite "
            f"'{metodo2}'. Os valores não são comparáveis em escala "
            f"(LIMITATIONS.md, seção 5).")

    shapes1, imp1 = snap["shapes"], snap["importance"]
    divergencias, colapsos = [], []
    for f in feature_names:
        a, b = shapes1.get(f), shapes2.get(f)
        i1, i2 = float(imp1.get(f, 0.0)), float(imp2.get(f, 0.0))
        if max(i1, i2) < min_rel_importance:
            continue                       # perfil de variável irrelevante é ruído
        if a and b and a != b:
            spread = abs(i1 - i2) / max(i1, i2, 1e-12)
            if spread >= tau:
                divergencias.append({"feature": f, "shape_before": a,
                                     "shape_after": b,
                                     "spread": round(spread, 3)})
        if i1 >= 0.02 and i2 / max(i1, 1e-12) < 0.40:
            colapsos.append({"feature": f,
                             "importance_before": round(i1, 4),
                             "importance_after": round(i2, 4),
                             "ratio": round(i2 / max(i1, 1e-12), 3)})

    mon = list(monitored or [])
    mon_hits = [c["feature"] for c in colapsos if c["feature"] in mon]

    if colapsos:
        tipo, verdict = "IMPORTANCE_COLLAPSE", "BLOQUEADO"
    elif divergencias:
        tipo, verdict = "SHAPE_DIVERGENCE", "ATENCAO"
    else:
        tipo, verdict = "NONE", "APROVADO"
    if mon and mon_hits:
        verdict = "BLOQUEADO"

    return {
        "verdict": verdict,
        "problem_type": tipo,
        "root_cause_features": [c["feature"] for c in colapsos],
        "divergences": divergencias,
        "collapses": colapsos,
        "monitored": mon or None,
        "monitored_hits": mon_hits,
        "snapshot": {"label": snap.get("label"),
                     "created": snap.get("created_utc"),
                     "file": os.path.basename(snapshot_path)},
        "same_eval_set": mesmo_eval,
        "n_eval": int(n2),
        "representation": metodo2,
        # v1.65: declarar o que esta comparacao NAO cobre. Um relatorio que
        # omite isto parece completo enquanto mede menos do que o leitor
        # supoe.
        "not_evaluated": ressalvas + [
            {"layer": "train/serving consistency",
             "reason": "o snapshot nao guarda a distribuicao de treino",
             "consequence": ("inconsistencia de escala entre treino e "
                             "inferencia nao foi verificada")},
            {"layer": "fairness",
             "reason": "nao integrado por decisao de projeto",
             "consequence": ("peso de variaveis sensiveis nao foi "
                             "verificado")},
        ] + ([] if mesmo_eval else [
            {"layer": "comparabilidade",
             "reason": "conjunto de avaliacao difere do snapshot",
             "consequence": ("parte da diferenca medida vem dos conjuntos, "
                             "nao dos modelos")}]),
    }
