"""
cli.py — interface de linha de comando (v1.62)

Projetada para dois usos:

  1. pessoa no terminal    → saída legível, cores quando há tty
  2. pipeline de CI/CD     → saída JSON, código de saída acionável

CÓDIGOS DE SAÍDA
----------------
    0   nenhum problema no nível exigido por --fail-on (padrão: high)
    1   problema encontrado no nível exigido
    2   erro de execução, ou veredito INDETERMINADO (a auditoria não
        conseguiu decidir)

O código 2 é distinto do 1 de propósito: um pipeline deve tratar "o modelo
tem problema" de forma diferente de "a auditoria não rodou". Confundir os
dois faz uma falha de infraestrutura passar por aprovação.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

EXIT_OK, EXIT_FOUND, EXIT_ERROR = 0, 1, 2

_SEV = {"APROVADO": 0, "ATENCAO": 1, "REVISAR": 2, "BLOQUEADO": 3}
_NIVEL = {"none": 99, "low": 1, "medium": 2, "high": 3}


def _codigo(verdict, fail_on) -> int:
    """
    Uma regra só para o código de saída (BUG 85, v1.0.4).

    A CLI tinha a sua tabela e a API (`pipeline.exit_code`) outra. Na CLI,
    INDETERMINADO valia 2 e, com --fail-on high, uma auditoria que NÃO
    CONSEGUIU decidir saía 0 — igual a aprovado. INDETERMINADO é sempre o
    código 2: a auditoria não rodou o suficiente para responder.
    """
    if verdict not in _SEV:
        return EXIT_ERROR
    return EXIT_FOUND if _SEV[verdict] >= _NIVEL.get(fail_on, 99) else EXIT_OK


# ── apresentação ──────────────────────────────────────────────────────
def _cor(txt: str, cor: str) -> str:
    """Cor só quando a saída é um terminal — pipes recebem texto limpo."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return txt
    c = {"red": "31", "yellow": "33", "green": "32", "blue": "36",
         "grey": "90", "bold": "1"}.get(cor, "0")
    return f"\033[{c}m{txt}\033[0m"


def _icone(verdict: str) -> str:
    return {"APROVADO": _cor("OK", "green"),
            "ATENCAO": _cor("!", "yellow"),
            "REVISAR": _cor("?", "yellow"),
            "BLOQUEADO": _cor("X", "red")}.get(verdict, "-")


def _texto(res: Dict[str, Any]) -> str:
    """Relatório para leitura humana no terminal."""
    v = res.get("verdict", "INDETERMINADO")
    L = [""]
    L.append(f"  {_icone(v)}  {_cor(v, 'bold')}   {res.get('problem_type','')}")

    raiz = res.get("root_cause_features") or []
    if raiz:
        L.append("")
        L.append(f"  {_cor('Comece por:', 'bold')} " +
                 ", ".join(_cor(f, "red") for f in raiz))
        L.append(_cor("  estas variaveis deixaram de influenciar o modelo.",
                      "grey"))

    div = res.get("divergences") or []
    outras = [str(d.get("feature")) for d in div
              if d.get("feature") not in raiz]
    if outras:
        L.append("")
        L.append("  Mudaram de padrao: " + ", ".join(outras[:6]) +
                 (f" (+{len(outras)-6})" if len(outras) > 6 else ""))

    snap = res.get("snapshot")
    if snap:
        L.append("")
        L.append(_cor(f"  comparado com: {snap.get('label')} "
                      f"({str(snap.get('created'))[:10]})", "grey"))
    ne = res.get("not_evaluated") or []
    if ne:
        L.append("")
        L.append(_cor("  nao verificado nesta auditoria:", "yellow"))
        for it in ne:
            L.append(_cor(f"    - {it.get('layer')}: {it.get('reason')}",
                          "grey"))
        L.append(_cor("    (o veredito acima nao cobre estes pontos)", "grey"))

    L.append(_cor(f"  {res.get('n_eval','?')} amostras - "
                  f"atribuicao por {res.get('representation','?')}", "grey"))
    L.append("")
    return "\n".join(L)


def _html(res: Dict[str, Any]) -> str:
    """v1.69: delega ao renderizador de laudo (modelcheck/laudo.py)."""
    from modelcheck.laudo import render
    return render(res)


def _html_legado(res: Dict[str, Any]) -> str:
    """Relatorio autocontido, para anexar num ticket ou enviar por e-mail."""
    v = res.get("verdict", "?")
    cor = {"APROVADO": "#2E7D57", "ATENCAO": "#B8860B",
           "REVISAR": "#B8860B", "BLOQUEADO": "#B03A2E"}.get(v, "#555")
    raiz = res.get("root_cause_features") or []
    linhas = "".join(
        f"<tr><td><code>{d['feature']}</code></td><td>{d.get('shape_before')}</td>"
        f"<td>{d.get('shape_after')}</td><td>{d.get('spread','')}</td></tr>"
        for d in res.get("divergences", []))
    col = "".join(
        f"<tr><td><code>{c['feature']}</code></td>"
        f"<td>{c.get('importance_before')}</td>"
        f"<td>{c.get('importance_after')}</td><td>{c.get('ratio')}</td></tr>"
        for c in res.get("collapses", []))
    bloco_raiz = ""
    if raiz:
        nomes = ", ".join(f"<code>{f}</code>" for f in raiz)
        bloco_raiz = (f'<div class="box"><b>Comece por:</b> {nomes}<br>'
                      f'<span style="color:#666">estas variaveis deixaram de '
                      f'influenciar o modelo.</span></div>')
    t_col = (f'<h2 style="font-size:17px">Colapso de importancia</h2><table>'
             f'<tr><th>variavel</th><th>antes</th><th>depois</th>'
             f'<th>razao</th></tr>{col}</table>') if col else ""
    t_div = (f'<h2 style="font-size:17px">Mudanca de forma</h2><table>'
             f'<tr><th>variavel</th><th>antes</th><th>depois</th>'
             f'<th>espalhamento</th></tr>{linhas}</table>') if linhas else ""
    ne = res.get("not_evaluated") or []
    t_ne = ""
    if ne:
        itens = "".join(f"<li><b>{i['layer']}</b>: {i['reason']}<br>"
                        f"<span style='color:#777'>{i.get('consequence','')}"
                        f"</span></li>" for i in ne)
        t_ne = (f'<div class="warn"><b>Nao verificado nesta auditoria</b>'
                f'<ul>{itens}</ul>'
                f'<span style="color:#777">O resultado acima nao cobre estes '
                f'pontos.</span></div>')

    snap = res.get("snapshot") or {}
    t_snap = (f" - comparado com <b>{snap.get('label')}</b> de "
              f"{str(snap.get('created'))[:10]}") if snap else ""
    return f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>Auditoria de modelo - {v}</title><style>
body{{font:15px/1.6 system-ui,-apple-system,sans-serif;max-width:860px;
margin:40px auto;padding:0 20px;color:#1a1a1a}}
h1{{font-size:26px;margin:0 0 4px}} .v{{color:{cor};font-weight:700}}
table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid #e4e4e4}}
th{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#666;
font-weight:500}}
code{{background:#f4f4f4;padding:1px 5px;border-radius:3px;font-size:13px}}
.meta{{color:#777;font-size:13px;margin-top:26px;padding-top:14px;
border-top:1px solid #e4e4e4}}
.box{{background:#faf9f7;border-left:3px solid {cor};padding:14px 18px;
margin:18px 0}}
.warn{{background:#fffdf5;border-left:3px solid #C9A227;padding:12px 18px;
margin:20px 0;font-size:14px}}
.warn ul{{margin:8px 0 4px;padding-left:20px}} .warn li{{margin:5px 0}}</style></head><body>
<h1>Auditoria de modelo</h1>
<p>Resultado: <span class="v">{v}</span> - {res.get('problem_type','')}</p>
{bloco_raiz}{t_col}{t_div}{t_ne}
<div class="meta">{res.get('n_eval','?')} amostras - atribuicao por
{res.get('representation','?')}{t_snap}
<br>Gerado por modelcheck. Limites conhecidos em LIMITATIONS.md.</div>
</body></html>"""


def _emitir(res: Dict[str, Any], formato: str, saida: Optional[str]) -> None:
    txt = {"json": lambda: json.dumps(res, indent=2, ensure_ascii=False,
                                      default=str),
           "html": lambda: _html(res),
           "text": lambda: _texto(res)}[formato]()
    if saida:
        with open(saida, "w", encoding="utf-8") as fh:
            fh.write(txt)
        if formato != "json":
            print(_cor(f"relatorio salvo em {saida}", "grey"))
    else:
        _stdout_utf8()
        print(txt)


def _stdout_utf8() -> None:
    """
    BUG 57 (v1.0.4): no Windows, com a saída redirecionada, o stdout usa
    cp1252 e o laudo (✓, →, —) derrubava a CLI com UnicodeEncodeError.
    """
    rec = getattr(sys.stdout, "reconfigure", None)
    if rec is None:
        return
    try:
        rec(encoding="utf-8")
    except Exception:
        pass                      # noqa: lacuna-nao-aplicavel — stream sem suporte


# ── carregamento ──────────────────────────────────────────────────────
def _carregar_modelo(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"modelo nao encontrado: {path}")
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)


def _carregar_dados(path: str, sep: Optional[str], target: Optional[str]):
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(f"dados nao encontrados: {path}")
    df = pd.read_csv(path, sep=sep or None, engine="python")
    y = df.pop(target) if target and target in df.columns else None
    return df, y


# ── comandos ──────────────────────────────────────────────────────────
def _cmd_snapshot_save(a) -> int:
    from modelcheck import snapshot
    X, _ = _carregar_dados(a.data, a.sep, a.target)
    m = _carregar_modelo(a.model)
    s = snapshot.save(m, X, a.output, label=a.label,
                      representation=a.representation)
    print(f"{_cor('OK','green')} snapshot gravado: {a.output}")
    print(_cor(f"  {s['label']} - {len(s['features'])} variaveis - "
               f"{s['meta']['n_eval']} amostras - "
               f"{s['meta']['representation']}", "grey"))
    return EXIT_OK


def _cmd_snapshot_list(a) -> int:
    from modelcheck import snapshot
    rows = snapshot.listar(a.dir)
    if not rows:
        print(_cor(f"nenhum snapshot em {a.dir}", "grey"))
        return EXIT_OK
    print(f"\n  {'arquivo':<26}{'rotulo':<20}{'criado':<12}{'vars':>5}{'n':>8}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r['file']:<26}{str(r['label'])[:19]:<20}"
              f"{str(r['created'])[:10]:<12}{r['features']:>5}{r['n_eval']:>8}")
    print()
    return EXIT_OK


def _cmd_diff(a) -> int:
    from modelcheck import snapshot
    X, _ = _carregar_dados(a.data, a.sep, a.target)
    m = _carregar_modelo(a.model)
    mon = [s.strip() for s in a.monitor.split(",")] if a.monitor else None
    res = snapshot.diff(a.against, m, X, monitored=mon,
                        strict_eval=not a.allow_different_eval)
    _emitir(res, a.format, a.output)
    return _codigo(res["verdict"], a.fail_on)


def _cmd_compare(a) -> int:
    """Compara dois modelos carregados - o modo antigo, ainda util."""
    from modelcheck import Guard
    X, y = _carregar_dados(a.data, a.sep, a.target)
    m1, m2 = _carregar_modelo(a.model_a), _carregar_modelo(a.model_b)
    mon = [s.strip() for s in a.monitor.split(",")] if a.monitor else None
    Xt = None
    if getattr(a, "train", None):
        Xt, _ = _carregar_dados(a.train, a.sep, a.target)
        Xt = Xt[[c for c in X.columns if c in Xt.columns]]
    g = Guard(mode=getattr(a, "mode", "retraining"))
    r = g.audit(m1, m2, X, y, list(X.columns), X_train=Xt)
    res = _dict_do_resultado(r, n_eval=len(X))
    res["monitored"] = mon
    # BUG 86 (v1.0.4): --monitor em `compare` era só copiado para a saída,
    # sem efeito nenhum. Agora roda a checagem dirigida (a de maior
    # especificidade) e, se uma variável monitorada colapsou, o resultado
    # vira BLOQUEADO mesmo que a auditoria geral não tenha bloqueado.
    if mon:
        c = g.check_features(m1, m2, X, monitored=mon)
        res["monitored_hits"] = c["collapsed"]
        if c["collapsed"] and res.get("verdict") != "BLOQUEADO":
            res["verdict"] = "BLOQUEADO"
            res["verdict_driver"] = "monitored_feature"
            res["root_cause_features"] = list(dict.fromkeys(
                list(res.get("root_cause_features") or []) + c["collapsed"]))
    _emitir(res, a.format, a.output)
    return _codigo(res["verdict"], a.fail_on)


def _dict_do_resultado(r, n_eval=None) -> Dict[str, Any]:
    """
    BUG 55 (v1.0.4): `compare --format html` montava um dicionário próprio,
    parcial — sem clones, reaprendizado, `skew_status` nem model diff — e
    com `representation='shap'` fixo. Um caso ATENCAO por substituição de
    variável saía com "✓ nenhuma substituição". Agora usamos o mesmo
    dicionário que `AuditResult.to_html` entrega ao laudo.
    """
    from modelcheck.laudo import _idioma
    if hasattr(r, "to_dict"):
        res = dict(r.to_dict(_idioma("auto")))
    else:
        raw = getattr(r, "raw", {}) or {}
        sdd = (raw.get("layer1", {}) or {}).get("sdd", {}) or {}
        res = {
            "verdict": r.verdict,
            "problem_type": getattr(r, "problem_type", "") or "",
            "root_cause_features": r.root_cause_features,
            "divergences": getattr(r, "shape_divergences", None)
            or sdd.get("divs", []),
            "collapses": [],
            "model_diff": getattr(r, "model_diff", []),
            "skew_features": getattr(r, "skew_features", []),
            "skew_status": getattr(r, "skew_status", "ausente"),
            "skew_error": getattr(r, "skew_error", ""),
            "clone_redistribution": getattr(r, "clone_redistribution", {}),
            "relearning": list(getattr(r, "relearning", []) or []),
            "decisao": {},
            "n_eval": (len(raw.get("X_eval_ref"))
                       if raw.get("X_eval_ref") is not None else "—"),
            "representation": sdd.get("method", "shap"),
            "not_evaluated": r.not_evaluated,
        }
    if n_eval is not None and res.get("n_eval") in (None, "—"):
        res["n_eval"] = n_eval
    return res


def _cmd_check(a) -> int:
    """Verificacao de integridade de variaveis - o modo de maior F1."""
    from modelcheck import Guard
    X, _ = _carregar_dados(a.data, a.sep, a.target)
    m1, m2 = _carregar_modelo(a.model_a), _carregar_modelo(a.model_b)
    mon = [s.strip() for s in a.monitor.split(",")]
    c = Guard().check_features(m1, m2, X, monitored=mon,
                               feature_names=list(X.columns))
    res = {"verdict": "BLOQUEADO" if c["alert"] else "APROVADO",
           "problem_type": "IMPORTANCE_COLLAPSE" if c["alert"] else "NONE",
           "root_cause_features": c["collapsed"], "divergences": [],
           "collapses": [], "monitored": mon, "n_eval": len(X),
           "representation": "shap",
           "not_evaluated": [
               {"layer": "shape divergence",
                "reason": "modo `check` verifica apenas integridade",
                "consequence": "mudanca de forma nao foi verificada"},
               {"layer": "train/serving consistency",
                "reason": "nao executada no modo `check`",
                "consequence": "skew de dados nao foi verificado"}]}
    _emitir(res, a.format, a.output)
    return _codigo("BLOQUEADO" if c["alert"] else "APROVADO", a.fail_on)


# ── parser ────────────────────────────────────────────────────────────
def _add_comuns(sp):
    sp.add_argument("--sep", default=None, help="CSV separator")
    sp.add_argument("--target", default=None, help="target column, if present")
    sp.add_argument("--format", choices=["text", "json", "html"],
                    default="text")
    sp.add_argument("-o", "--output", default=None, help="output file")
    # BUG 85: o padrão era "none" — um BLOQUEADO saía com código 0 e o CI
    # deixava passar o modelo quebrado. Padrão agora: "high" (BLOQUEADO
    # falha o job). "none" continua disponível para quem só quer o laudo.
    sp.add_argument("--fail-on", choices=["none", "low", "medium", "high"],
                    default="high",
                    help="exit 1 from this severity upwards (default: high = "
                         "BLOQUEADO). low=ATENCAO+, medium=REVISAR+, none=never. "
                         "INDETERMINADO always exits 2")
    sp.add_argument("--monitor", default=None,
                    help="critical features, comma-separated")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shapeaudit",
        description="Audit what changed between two versions of a model.",
        epilog="exit codes: 0 clean | 1 problem found | 2 execution error")
    sub = p.add_subparsers(dest="cmd", required=True)

    sn = sub.add_parser("snapshot", help="save and list snapshots")
    sn_sub = sn.add_subparsers(dest="sub", required=True)

    s1 = sn_sub.add_parser("save", help="save a model's structural state")
    s1.add_argument("model")
    s1.add_argument("data")
    s1.add_argument("-o", "--output", required=True)
    s1.add_argument("--label", default=None)
    s1.add_argument("--sep", default=None)
    s1.add_argument("--target", default=None)
    s1.add_argument("--representation", choices=["shap", "ale"],
                    default="shap")
    s1.set_defaults(func=_cmd_snapshot_save)

    s2 = sn_sub.add_parser("list", help="list snapshots in a directory")
    s2.add_argument("dir", nargs="?", default=".")
    s2.set_defaults(func=_cmd_snapshot_list)

    d = sub.add_parser("diff",
                       help="compare a model against a saved snapshot")
    d.add_argument("model")
    d.add_argument("data")
    d.add_argument("--against", required=True, help="snapshot file")
    d.add_argument("--allow-different-eval", action="store_true",
                   help="allow a different evaluation set (not "
                        "recommended: measures the gap between sets)")
    _add_comuns(d)
    d.set_defaults(func=_cmd_diff)

    c = sub.add_parser("compare", help="compare two loaded models")
    c.add_argument("model_a")
    c.add_argument("model_b")
    c.add_argument("data")
    # BUG 91 (v1.0.4): sem como passar os dados de treino, a camada treino ×
    # uso nunca rodava pela CLI — o mesmo par dava ATENCAO pela API e
    # APROVADO pela CLI.
    c.add_argument("--train", default=None,
                   help="training data of model_b (CSV, same columns); enables "
                        "the train/serving layer")
    c.add_argument("--mode", choices=["retraining", "exploratory"],
                   default="retraining")
    _add_comuns(c)
    c.set_defaults(func=_cmd_compare)

    k = sub.add_parser("check",
                       help="check integrity of critical features")
    k.add_argument("model_a")
    k.add_argument("model_b")
    k.add_argument("data")
    _add_comuns(k)
    k.set_defaults(func=_cmd_check)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "check" and not getattr(args, "monitor", None):
        print(_cor("error: `check` requires --monitor with the critical features",
                   "red"), file=sys.stderr)
        return EXIT_ERROR
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"{_cor('erro:', 'red')} {e}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print(_cor("\ninterrompido", "grey"), file=sys.stderr)
        return EXIT_ERROR
    except Exception as e:
        print(f"{_cor('erro inesperado:', 'red')} {type(e).__name__}: {e}",
              file=sys.stderr)
        print(_cor("  para o traceback completo: MODELCHECK_DEBUG=1",
                   "grey"), file=sys.stderr)
        if os.environ.get("MODELCHECK_DEBUG"):
            raise
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
