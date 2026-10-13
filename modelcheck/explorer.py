"""
explorer.py — InteractiveDivergenceExplorer

Gera relatório HTML interativo com Plotly mostrando:
  - Perfis SHAP lado a lado (v1 vs v2)
  - Regiões de discordância (do PairedDivergenceAnalyzer)
  - Alertas de todos os detectores em um painel unificado
  - Health score visual

Uso:
    from modelcheck.explorer import InteractiveDivergenceExplorer
    exp = InteractiveDivergenceExplorer()
    html = exp.generate(model_v1, model_v2, X_eval, features,
                        title="CCEE: Normal vs Crise",
                        output_path="report.html")
"""
import numpy as np
import pandas as pd
from typing import List, Optional
import html as _html
import json


# BUG 56 (v1.0.4): nomes de variável, título, rótulos e regras entravam crus
# no HTML (e os nomes, via innerHTML, no DOM); o JSON dentro de <script> não
# escapava `</`; um rótulo com aspa simples quebrava o JS. Tudo que vem do
# usuário passa por estas duas funções.
def _e(x) -> str:
    return _html.escape(str(x))


def _js(x) -> str:
    """Valor JSON seguro para embutir dentro de <script>."""
    # `<` vira \u003c: cobre `</script>` e `<!--` (json.dumps já escapa
    # U+2028/U+2029 com ensure_ascii)
    return json.dumps(x).replace("<", "\\u003c")


def _shap_profiles(model, X: pd.DataFrame,
                   feature_names: List[str],
                   n_sample: int = 150) -> dict:
    """Calcula perfis SHAP para visualização."""
    import shap
    from modelcheck.shape import _shape_profile
    Xs = X.sample(min(n_sample, len(X)), random_state=42)
    sv = shap.TreeExplainer(model).shap_values(Xs)
    if sv.ndim == 3: sv = sv[:,:,1]
    profiles = {}
    for i, f in enumerate(feature_names):
        xv = Xs[f].values
        sv_f = sv[:, i]
        # Cria bins para visualização
        n_bins = 8
        bins = np.percentile(xv, np.linspace(0, 100, n_bins+1))
        bins = np.unique(bins)
        x_plot, y_mean, y_std = [], [], []
        for j in range(len(bins)-1):
            mask = (xv >= bins[j]) & (xv <= bins[j+1])
            if mask.sum() < 3: continue
            x_plot.append(float((bins[j]+bins[j+1])/2))
            y_mean.append(float(sv_f[mask].mean()))
            y_std.append(float(sv_f[mask].std()))
        profiles[f] = {
            'x': x_plot, 'y_mean': y_mean, 'y_std': y_std,
            'importance': round(float(np.abs(sv_f).mean()), 4)
        }
    return profiles


class InteractiveDivergenceExplorer:
    """Gera relatório HTML interativo de divergências entre modelos."""

    def generate(self, model_v1, model_v2,
                 X_eval: pd.DataFrame,
                 feature_names: Optional[List[str]] = None,
                 title: str = "ModelGuard — Divergence Report",
                 label_v1: str = "Model v1",
                 label_v2: str = "Model v2",
                 output_path: Optional[str] = None,
                 audit_result: Optional[dict] = None,
                 paired_result: Optional[dict] = None) -> str:
        """Gera HTML completo. Retorna string HTML."""

        if feature_names is None:
            feature_names = list(X_eval.columns)

        # Calcula perfis SHAP
        prof1 = _shap_profiles(model_v1, X_eval, feature_names)
        prof2 = _shap_profiles(model_v2, X_eval, feature_names)

        # Cria dados para os gráficos
        charts_data = []
        divergences = audit_result.get('divergences', []) if audit_result else []
        div_feats = {d['feature'] for d in divergences}

        for f in feature_names:
            p1 = prof1.get(f, {}); p2 = prof2.get(f, {})
            if not p1 or not p2: continue
            has_div = f in div_feats
            shape_info = ""
            for d in divergences:
                if d['feature'] == f:
                    shape_info = f"{d.get('shape_v1')} → {d.get('shape_v2')}"
            charts_data.append({
                'feature': f,
                'has_div': has_div,
                'shape_info': shape_info,
                'importance': max(p1.get('importance',0), p2.get('importance',0)),
                'x1': p1.get('x',[]), 'y1': p1.get('y_mean',[]),
                'x2': p2.get('x',[]), 'y2': p2.get('y_mean',[]),
            })

        # Ordena: divergentes primeiro, depois por importância
        charts_data.sort(key=lambda c: (-int(c['has_div']), -c['importance']))

        # Health score
        score = audit_result['health_score'] if audit_result else 100
        status = audit_result['status'] if audit_result else 'HEALTHY'
        score_color = ('#10B981' if score >= 85 else
                       '#F59E0B' if score >= 65 else '#EF4444')

        # Regras do PairedDivergenceAnalyzer
        rules_html = ""
        if paired_result and paired_result.get('rules'):
            rules_html = "<h3>📍 Regiões de Divergência</h3><div class='rules'>"
            for i, r in enumerate(paired_result['rules'][:4], 1):
                pct = int(r.get('pct_divergent_real', 0) * 100)
                div_m = r.get('mean_div_real', 0)
                rules_html += f"""
                <div class='rule'>
                  <div class='rule-num'>[{i}]</div>
                  <div class='rule-body'>
                    <code>{_e(r.get('rule_text', ''))}</code>
                    <div class='rule-stats'>
                      {_e(r.get('n_real', 0))} amostras &nbsp;·&nbsp;
                      {pct}% divergentes &nbsp;·&nbsp;
                      div_média={div_m:.3f}
                    </div>
                  </div>
                </div>"""
            rules_html += "</div>"

        # Gera os gráficos como JSON para Plotly inline
        charts_json = _js(charts_data)
        labels_json = _js([str(label_v1), str(label_v2)])
        title, label_v1, label_v2 = _e(title), _e(label_v1), _e(label_v2)
        score, status = _e(score), _e(status)

        html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#0A0A0F;color:#F0F0FF;font-family:'system-ui',sans-serif;padding:24px}}
h1{{font-size:1.5rem;font-weight:700;margin-bottom:4px;letter-spacing:-0.02em}}
.sub{{color:#A0A0C0;font-size:.875rem;margin-bottom:24px}}
.header{{display:flex;align-items:center;justify-content:space-between;margin-bottom:24px}}
.score-box{{background:#1A1A26;border:2px solid {score_color};border-radius:12px;
           padding:16px 24px;text-align:center;min-width:140px}}
.score-num{{font-size:2.5rem;font-weight:700;color:{score_color};line-height:1;
           font-family:monospace}}
.score-label{{font-size:.75rem;color:{score_color};font-weight:700;
             letter-spacing:.1em;margin-top:4px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px}}
.card{{background:#1A1A26;border:1px solid #2A2A3A;border-radius:10px;padding:16px}}
.card.divergent{{border-color:#EF444466;background:linear-gradient(135deg,#1A1A26,#2A1A1A)}}
.card-title{{font-size:.85rem;font-weight:600;margin-bottom:4px;display:flex;
            align-items:center;gap:8px}}
.badge{{font-size:.65rem;padding:2px 8px;border-radius:4px;font-family:monospace;
        font-weight:700}}
.badge.div{{background:#EF444422;color:#EF4444}}
.badge.ok{{background:#10B98122;color:#10B981}}
.chart-wrap{{height:200px}}
.rules{{display:flex;flex-direction:column;gap:10px;margin-top:12px}}
.rule{{display:flex;gap:12px;background:#111118;border-radius:8px;padding:12px;
       border-left:3px solid #7C3AED}}
.rule-num{{font-family:monospace;font-size:.8rem;color:#7C3AED;font-weight:700;
           white-space:nowrap}}
.rule-body code{{font-family:monospace;font-size:.8rem;color:#C4B5FD;display:block;margin-bottom:4px}}
.rule-stats{{font-size:.75rem;color:#A0A0C0}}
h3{{font-size:1rem;font-weight:600;margin:20px 0 8px;color:#A0A0C0}}
.legend{{display:flex;gap:16px;font-size:.75rem;color:#A0A0C0;margin-bottom:12px}}
.leg-dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:4px}}
footer{{margin-top:32px;font-size:.75rem;color:#606080;text-align:center}}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>🔍 {title}</h1>
    <div class="sub">{label_v1} vs {label_v2} &nbsp;·&nbsp;
    {len(feature_names)} features &nbsp;·&nbsp;
    {len(X_eval)} amostras</div>
  </div>
  <div class="score-box">
    <div class="score-num">{score}</div>
    <div class="score-label">HEALTH SCORE<br>{status}</div>
  </div>
</div>

<div class="legend">
  <span><span class="leg-dot" style="background:#7C3AED"></span>{label_v1}</span>
  <span><span class="leg-dot" style="background:#10B981"></span>{label_v2}</span>
</div>

{rules_html}

<h3>📊 Perfis SHAP por Feature</h3>
<div class="grid" id="grid"></div>

<footer>ShapeAudit · modelcheck · MIT</footer>

<script>
const data = {charts_json};
const labels = {labels_json};
const grid = document.getElementById('grid');

data.forEach((d, idx) => {{
  const div = document.createElement('div');
  div.className = 'card' + (d.has_div ? ' divergent' : '');
  const badgeClass = d.has_div ? 'div' : 'ok';
  const badgeText  = d.has_div ? '⚠ SHAPE_DIV ' + d.shape_info : '✅ stable';
  // só textContent: o nome da variável vem dos dados (BUG 56)
  const tit = document.createElement('div'); tit.className = 'card-title';
  const nome = document.createElement('span'); nome.textContent = d.feature;
  const badge = document.createElement('span');
  badge.className = 'badge ' + badgeClass; badge.textContent = badgeText;
  tit.appendChild(nome); tit.appendChild(badge);
  const wrap = document.createElement('div'); wrap.className = 'chart-wrap';
  wrap.id = 'chart-' + idx;
  div.appendChild(tit); div.appendChild(wrap);
  grid.appendChild(div);

  const traces = [
    {{
      x: d.x1, y: d.y1, name: labels[0],
      mode: 'lines+markers', line: {{color:'#7C3AED',width:2}},
      marker: {{size:5}}, type:'scatter'
    }},
    {{
      x: d.x2, y: d.y2, name: labels[1],
      mode: 'lines+markers', line: {{color:'#10B981',width:2}},
      marker: {{size:5}}, type:'scatter'
    }}
  ];
  Plotly.newPlot('chart-' + idx, traces, {{
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    margin:{{t:4,b:30,l:40,r:4}},
    xaxis:{{color:'#606080',gridcolor:'#2A2A3A',title:d.feature,
             titlefont:{{size:10}}}},
    yaxis:{{color:'#606080',gridcolor:'#2A2A3A',title:'SHAP',
             titlefont:{{size:10}}}},
    legend:{{font:{{color:'#A0A0C0',size:10}},bgcolor:'transparent'}},
    showlegend: idx===0,
    font:{{color:'#A0A0C0'}}
  }}, {{responsive:true, displayModeBar:false}});
}});
</script>
</body></html>"""

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as fh:
                fh.write(html)

        return html
