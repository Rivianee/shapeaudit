"""ModelCheck Studio: gera dashboard HTML estático (3 abas) a partir dos resultados."""
import html
import json
from datetime import datetime

_CORES = {"verde": "#22c55e", "amarelo": "#eab308", "vermelho": "#ef4444"}


def _rows(items, cols):
    out = ""
    for it in items:
        out += "<tr>" + "".join(
            f"<td>{html.escape(str(it.get(c, '')))}</td>" for c in cols) + "</tr>"
    return out


def _sev_badge(s):
    c = {"CRITICO": "#ef4444", "CRÍTICO": "#ef4444",
         "ALERTA": "#eab308", "INFO": "#64748b"}.get(str(s).upper(), "#64748b")
    return f'<span class="badge" style="background:{c}">{html.escape(str(s))}</span>'


def build_dashboard(path, score=None, data_scan=None, guard_report=None,
                    diff_result=None, title="ModelCheck Studio"):
    """Escreve um dashboard HTML auto-contido em `path`."""
    ts = datetime.now().strftime("%d/%m/%Y %H:%M")

    # ---- aba dados
    dados = "<p class='muted'>Nenhuma auditoria de dados executada.</p>"
    if data_scan:
        alerts = data_scan.get("alerts", [])
        linhas = "".join(
            f"<tr><td>{_sev_badge(a.get('severity'))}</td>"
            f"<td>{html.escape(str(a.get('detector','')))}</td>"
            f"<td>{html.escape(str(a.get('column','')))}</td>"
            f"<td>{html.escape(str(a.get('message','')))}</td></tr>"
            for a in alerts)
        dados = (f"<p>{data_scan.get('n_rows','?')} linhas · "
                 f"{data_scan.get('n_cols','?')} colunas · "
                 f"{len(alerts)} alertas</p>"
                 "<table><tr><th>Severidade</th><th>Detector</th>"
                 f"<th>Coluna</th><th>Mensagem</th></tr>{linhas}</table>")

    # ---- aba modelo
    modelo = "<p class='muted'>Nenhuma auditoria de modelo executada.</p>"
    if guard_report:
        secoes = []
        div = guard_report.get("divergences", []) or []
        secoes.append(f"<h3>Divergências entre modelos ({len(div)})</h3>")
        if div:
            secoes.append("<pre>" + html.escape(
                json.dumps(div, indent=1, ensure_ascii=False, default=str)[:4000]) + "</pre>")
        contra = guard_report.get("contradictions", []) or []
        secoes.append(f"<h3>Contradições com regras ({len(contra)})</h3>")
        if contra:
            secoes.append("<pre>" + html.escape(
                json.dumps(contra, indent=1, ensure_ascii=False, default=str)[:4000]) + "</pre>")
        leak = guard_report.get("leakage", {}) or {}
        n_leak = sum(1 for v in leak.values() if v)
        secoes.append(f"<h3>Vazamento de dados</h3><p>"
                      f"{'⚠️ suspeita em ' + str(n_leak) + ' modelo(s)' if n_leak else '✅ não detectado'}</p>")
        modelo = "".join(secoes)

    # ---- aba comparação
    comp = "<p class='muted'>Nenhuma comparação executada. Use: modelcheck diff A B dados.csv</p>"
    if diff_result:
        w = diff_result.get("weight_changes", [])
        d = diff_result.get("direction_changes", [])
        s = diff_result.get("shape_changes", [])
        comp = (f"<h3>Mudanças de peso ({len(w)})</h3>"
                "<table><tr><th>feature</th><th>before</th><th>after</th>"
                f"<th>delta_pp</th><th>significant</th></tr>"
                f"{_rows(w, ['feature','before','after','delta_pp','significant'])}</table>"
                f"<h3>Mudanças de direção ({len(d)})</h3>"
                f"<pre>{html.escape(json.dumps(d, indent=1, ensure_ascii=False, default=str)[:3000])}</pre>"
                f"<h3>Mudanças de forma ({len(s)})</h3>"
                f"<pre>{html.escape(json.dumps(s, indent=1, ensure_ascii=False, default=str)[:3000])}</pre>")

    # ---- score
    gauge = ""
    if score:
        cor = _CORES.get(score["cor"], "#64748b")
        bd = " · ".join(f"{k}: {v}" for k, v in score["breakdown"].items())
        gauge = (f'<div class="score" style="border-color:{cor}">'
                 f'<div class="num" style="color:{cor}">{score["score"]}</div>'
                 f'<div class="lbl">{score["status"]}</div>'
                 f'<div class="bd">{html.escape(bd)}</div></div>')

    page = f"""<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}}
header{{padding:20px 32px;background:#1e293b;display:flex;align-items:center;gap:24px}}
h1{{font-size:20px;margin:0}} .muted{{color:#94a3b8}}
.score{{border:3px solid;border-radius:14px;padding:10px 22px;text-align:center;background:#0b1220}}
.score .num{{font-size:40px;font-weight:800}} .score .lbl{{font-weight:600}}
.score .bd{{font-size:12px;color:#94a3b8;margin-top:4px}}
nav{{display:flex;gap:6px;padding:12px 32px;background:#111c31}}
nav button{{background:#1e293b;color:#e2e8f0;border:0;padding:10px 18px;border-radius:8px;cursor:pointer;font-size:14px}}
nav button.on{{background:#3b82f6}}
section{{display:none;padding:24px 32px}} section.on{{display:block}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}}
th,td{{border:1px solid #334155;padding:6px 10px;text-align:left}}
th{{background:#1e293b}} pre{{background:#0b1220;padding:12px;border-radius:8px;overflow:auto;font-size:12px}}
.badge{{color:#fff;padding:2px 8px;border-radius:99px;font-size:11px;font-weight:600}}
footer{{padding:16px 32px;color:#64748b;font-size:12px}}
</style></head><body>
<header><h1>🛡️ {html.escape(title)}</h1>{gauge}
<span class="muted" style="margin-left:auto">gerado em {ts}</span></header>
<nav>
<button class="on" onclick="ab(0,this)">🧹 DataSpec (Dados)</button>
<button onclick="ab(1,this)">🛡️ ModelGuard (Modelo)</button>
<button onclick="ab(2,this)">🔍 ExplainDiff (Comparação)</button>
</nav>
<section class="on">{dados}</section>
<section>{modelo}</section>
<section>{comp}</section>
<footer>ModelCheck Studio — DataSpec + ModelGuard + ExplainDiff</footer>
<script>
function ab(i,b){{document.querySelectorAll('section').forEach((s,j)=>s.classList.toggle('on',i==j));
document.querySelectorAll('nav button').forEach(x=>x.classList.remove('on'));b.classList.add('on')}}
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path
