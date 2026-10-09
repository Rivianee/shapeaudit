"""ModelGuard — LLM Audit (módulo avançado).

Detectores:
  PromptAuditor       — DataSpec para LLMs: prompt injection, viés no input, contexto inválido
  HallucinationMeter  — "Alucinômetro": extrai afirmações, valida contra fonte de verdade, gera score
  ConsistencyAuditor  — Shape Divergence para LLMs: mesma pergunta N vezes, mede divergência
  LLMPersonalityDrift — Drift de personalidade entre versões (v1 → v2)
  LLMAuditGuard       — Ponto de entrada único

Uso mínimo:
    from modelguard.llm_audit import LLMAuditGuard, LLMWrapper
    wrapper = LLMWrapper(fn=meu_llm)
    guard = LLMAuditGuard(wrapper)
    report = guard.run(
        prompts=["Qual o saldo do cliente João?"],
        contexts=["Saldo: R$1.200. Cliente desde 2020."],
    )
    print(guard.narrate(report))
"""
import re
import numpy as np
from typing import List, Optional, Dict, Callable


# ── Utilidades de texto ──────────────────────────────────────────────────────

def _simple_embed(texts: List[str]) -> np.ndarray:
    """Embeddings TF-IDF offline — fallback quando sentence-transformers não está disponível."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=512, sublinear_tf=True)
    embs = vec.fit_transform(texts).toarray().astype("float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
    return embs / norms


def _embed(texts: List[str]) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    except Exception:
        return _simple_embed(texts)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _extract_claims(text: str) -> List[str]:
    """Extrai afirmações factuais simples do texto usando regex + spaCy."""
    claims = []
    # números + contexto
    for m in re.finditer(r'[\w\s]{0,20}\d[\d.,\s%km/hR$€£]{1,15}[\w\s]{0,20}', text):
        s = m.group().strip()
        if len(s) > 5:
            claims.append(s)
    # entidades nomeadas via spaCy
    try:
        import spacy
        nlp = spacy.load("pt_core_news_sm")
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("PER", "ORG", "LOC", "GPE", "DATE", "MONEY"):
                claims.append(ent.text)
    except Exception:
        # fallback: entidades capitalizadas.
        # BUG 19 (v1.51): o padrão exigia DUAS palavras capitalizadas seguidas
        # ({...}+), então nomes de UMA palavra ('Amazonas', 'Itaipu') não eram
        # capturados. Sem claim extraído, `HallucinationMeter.score()` devolvia
        # 0.0 — indistinguível de "nenhuma alucinação". Medido: 2 de 5
        # alucinações escapavam ('Itaipu fica no rio Amazonas' passava com
        # score 0.0).
        _CAP = r'[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ][a-záéíóúàâêôãõç]+'
        for m in re.finditer(rf'\b{_CAP}(?:\s+(?:d[aeo]s?\s+)?{_CAP})*\b', text):
            s = m.group().strip()
            # descarta a primeira palavra da frase quando é só início de período
            if len(s) > 3:
                claims.append(s)
        # substantivos com superlativo/comparativo carregam afirmação factual
        for m in re.finditer(
                r'\b(?:maior|menor|melhor|pior|primeir[ao]|últim[ao]|'
                r'únic[ao]|mais|menos)\b[\w\s]{0,40}', text, flags=re.I):
            s = m.group().strip()
            if len(s) > 6:
                claims.append(s)
    # deduplica mantendo ordem
    seen = set()
    result = []
    for c in claims:
        key = c.lower().strip()
        if key not in seen and len(key) > 3:
            seen.add(key)
            result.append(c)
    return result


# ── 1. PromptAuditor — DataSpec para LLMs ───────────────────────────────────

class PromptAuditor:
    """Audita prompt e contexto ANTES de enviar ao LLM.

    Detecta:
    - Prompt injection (tentativas de manipular o modelo)
    - Viés no input (termos que induzem estereótipos)
    - Contexto irrelevante (RAG fora do tópico)
    - Prompts vazios ou malformados
    """

    INJECTION_PATTERNS = [
        r'ignore\s+(as\s+)?(instru[çc][oõ]es|anteriores|tudo)',
        r'aja\s+como',
        r'pretend\s+you',
        r'forget\s+(everything|all)',
        r'ignore\s+previous',
        r'system\s*:',
        r'você\s+é\s+(agora\s+)?um',
        r'novo\s+sistema\s*:',
        r'<\s*system\s*>',
        r'\[INST\].*\[/INST\]',
        r'ignore\s+all\s+instructions',
    ]

    BIAS_TERMS = {
        'genero': ['enfermeira', 'enfermeiro', 'secretária', 'motorista', 'chefe',
                   'programador', 'programadora', 'médico', 'médica'],
        'racial': ['negro', 'branco', 'asiático', 'indígena'],
        'regional': ['nordestino', 'sulista', 'carioca', 'paulista'],
    }

    def __init__(self, context_relevance_threshold: float = 0.15):
        self.ctx_threshold = context_relevance_threshold

    def audit_prompt(self, prompt: str, context: str = "") -> Dict:
        alerts = []
        details = {}

        # 1. Vazio ou muito curto
        if not prompt or len(prompt.strip()) < 5:
            alerts.append({"severity": "CRITICO", "type": "prompt_vazio",
                           "message": "Prompt vazio ou muito curto."})

        # 2. Prompt injection
        injections = []
        for pat in self.INJECTION_PATTERNS:
            if re.search(pat, prompt, re.IGNORECASE):
                injections.append(pat)
        if injections:
            alerts.append({"severity": "CRITICO", "type": "prompt_injection",
                           "message": f"Possível prompt injection detectado. "
                                      f"Padrões encontrados: {len(injections)}. "
                                      "Não processe esse prompt sem validação humana."})
        details["injection_patterns_found"] = len(injections)

        # 3. Viés no input
        bias_found = {}
        prompt_lower = prompt.lower()
        for categoria, termos in self.BIAS_TERMS.items():
            hits = [t for t in termos if t in prompt_lower]
            if hits:
                bias_found[categoria] = hits
        if bias_found:
            cats = ", ".join(bias_found.keys())
            alerts.append({"severity": "ALERTA", "type": "vies_input",
                           "message": f"Prompt contém termos que podem induzir viés: {cats}. "
                                      "Monitore se a resposta reproduz estereótipos."})
        details["bias_categories"] = bias_found

        # 4. Relevância do contexto (RAG)
        if context and len(context.strip()) > 10:
            embs = _embed([prompt, context])
            sim = _cosine(embs[0], embs[1])
            details["context_relevance"] = round(sim, 4)
            if sim < self.ctx_threshold:
                alerts.append({"severity": "ALERTA", "type": "contexto_irrelevante",
                               "message": f"Contexto (RAG) parece pouco relevante para o prompt "
                                          f"(similaridade={sim:.3f} < {self.ctx_threshold}). "
                                          "A resposta pode ignorar o contexto ou alucinar."})
        return {"alerts": alerts, "details": details,
                "n_alerts": len(alerts), "safe": len(alerts) == 0}


# ── 2. HallucinationMeter — Alucinômetro ────────────────────────────────────

class HallucinationMeter:
    """Detecta alucinação validando afirmações da resposta contra fonte de verdade.

    Estratégia:
    1. Extrai afirmações factuais da resposta (números, entidades, datas)
    2. Para cada afirmação, verifica se está presente no contexto fornecido
    3. Calcula HallucinationScore = afirmações não encontradas / total
    4. Gera alerta narrativo com as afirmações suspeitas

    Funciona mesmo sem API externa — usa busca semântica no contexto.
    """

    def __init__(self, similarity_threshold: float = 0.45):
        self.sim_threshold = similarity_threshold

    def _claim_in_context(self, claim: str, context: str) -> bool:
        """Verifica se uma afirmação está semanticamente presente no contexto."""
        # verificação direta primeiro (mais rápida e mais confiável)
        if claim.lower() in context.lower():
            return True
        # verificação semântica por sentença
        sentences = [s.strip() for s in re.split(r'[.!?\n]', context) if len(s.strip()) > 5]
        if not sentences:
            return False
        try:
            embs = _embed([claim] + sentences)
            sims = [_cosine(embs[0], embs[i + 1]) for i in range(len(sentences))]
            if max(sims) >= self.sim_threshold:
                return True
            # BUG 22 (v1.52): o limiar 0.45 é ABSOLUTO sobre similaridade
            # TF-IDF, que cai quando o claim é curto e a sentença longa.
            # Medido: o claim CORRETO 'Itaipu tem 14 GW' contra o contexto
            # completo dava 0.258 — abaixo do limiar — e era marcado como
            # não verificado. Falso positivo de alucinação num fato literal.
            # Mesma família do bug 13 (limiar absoluto sobre quantidade
            # dependente de escala).
            # Fallback: cobertura lexical do claim pela melhor sentença.
            # Um claim cujos termos significativos estão TODOS presentes
            # numa sentença do contexto está verificado, independente do
            # tamanho relativo dos textos.
            import unicodedata

            def _norm(t):
                t = unicodedata.normalize('NFKD', t.lower())
                t = ''.join(ch for ch in t if not unicodedata.combining(ch))
                return set(re.findall(r'\w+', t))

            _STOP = {'de', 'da', 'do', 'das', 'dos', 'e', 'a', 'o', 'as',
                     'os', 'em', 'no', 'na', 'nos', 'nas', 'um', 'uma',
                     'que', 'com', 'por', 'para', 'tem', 'e', 'foi', 'ser'}
            termos = _norm(claim) - _STOP
            if not termos:
                return False
            for s in sentences:
                cobertura = len(termos & _norm(s)) / len(termos)
                if cobertura >= 0.85:
                    return True
            return False
        except Exception:
            return False

    def score(self, response: str, context: str = "",
              ground_truth: Optional[Dict] = None) -> Dict:
        """
        response     : texto gerado pelo LLM
        context      : contexto fornecido (RAG, documento, banco de dados)
        ground_truth : dict opcional {entidade: valor_correto}
                       ex: {"nome": "João Silva", "saldo": "R$1.200"}
        """
        # BUG 20 (v1.52): sem claim extraído (resposta vazia, resposta curta
        # como "Sim.", ou contexto vazio), o método devolvia
        # hallucination_score=0.0 e label='BAIXO' — "risco baixo de
        # alucinação" quando NADA foi verificado. Indistinguível de uma
        # resposta plenamente fundamentada.
        # Mesmo padrão dos bugs 17, 18 e 19.
        if not (response or "").strip():
            return {'hallucination_score': None, 'n_claims': 0,
                    'label': 'NAO_AVALIAVEL', 'verified': [],
                    'unverified': [],
                    'reason': 'resposta vazia — nada a verificar'}
        if not (context or "").strip():
            return {'hallucination_score': None,
                    'n_claims': len(_extract_claims(response)),
                    'label': 'NAO_AVALIAVEL', 'verified': [],
                    'unverified': [],
                    'reason': ('contexto vazio — sem referência não é '
                               'possível distinguir alucinação de fato '
                               'correto. Forneça o contexto usado pelo '
                               'modelo (RAG, documento, prompt).')}

        claims = _extract_claims(response)
        if not claims:
            # BUG 20 (v1.52): este retorno dava score=0.0 e label='BAIXO' —
            # "risco baixo de alucinação" quando NENHUMA afirmação foi
            # verificada. Uma resposta como "Sim." recebia o mesmo veredito de
            # uma resposta plenamente fundamentada no contexto.
            return {"hallucination_score": None, "n_claims": 0,
                    "label": "NAO_AVALIAVEL",
                    "unverified": [], "verified": [],
                    "alerts": [{"severity": "INFO",
                                "type": "nao_avaliavel",
                                "message": ("Nenhuma afirmação factual "
                                            "extraída. Respostas sem "
                                            "entidades, números ou "
                                            "superlativos não são "
                                            "verificáveis por este método.")}],
                    "message": "Nenhuma afirmação factual extraída da resposta."}

        verified, unverified = [], []
        for claim in claims:
            found = False
            # checa no ground truth primeiro
            if ground_truth:
                for key, val in ground_truth.items():
                    if str(val).lower() in claim.lower() or claim.lower() in str(val).lower():
                        found = True
                        break
            # checa no contexto
            if not found and context:
                found = self._claim_in_context(claim, context)
            if found:
                verified.append(claim)
            else:
                unverified.append(claim)

        score = len(unverified) / len(claims)
        alerts = []
        if score > 0.5:
            alerts.append({"severity": "CRITICO",
                           "message": f"Alta probabilidade de alucinação: {score:.0%} das afirmações "
                                      f"({len(unverified)}/{len(claims)}) não foram encontradas "
                                      f"no contexto. Afirmações suspeitas: {unverified[:3]}"})
        elif score > 0.2:
            alerts.append({"severity": "ALERTA",
                           "message": f"Alucinação moderada: {score:.0%} das afirmações não verificadas. "
                                      f"Revisar: {unverified[:2]}"})

        # BUG 20 (v1.52): com n_claims == 0 o score saía 0.0 e o label
        # 'BAIXO' — "risco baixo de alucinação" quando NENHUMA afirmação foi
        # verificada. Uma resposta como "Sim." recebia o mesmo veredito de uma
        # resposta plenamente fundamentada.
        if not claims:
            return {"hallucination_score": None,
                    "n_claims": 0,
                    "verified": [], "unverified": [],
                    "alerts": [{"severity": "INFO",
                                "type": "nao_avaliavel",
                                "message": ("Nenhuma afirmação factual "
                                            "extraída. Respostas sem "
                                            "entidades, números ou "
                                            "superlativos não são "
                                            "verificáveis por este método.")}],
                    "label": "NAO_AVALIAVEL"}

        return {"hallucination_score": round(score, 4),
                "n_claims": len(claims),
                "verified": verified,
                "unverified": unverified,
                "alerts": alerts,
                "label": "ALTO" if score > 0.5 else "MODERADO" if score > 0.2 else "BAIXO"}

# ── 3. ConsistencyAuditor — Shape Divergence para LLMs ──────────────────────

class ConsistencyAuditor:
    """Mede consistência semântica: mesma pergunta N vezes → respostas devem ser similares.

    Alta divergência = modelo instável = risco de alucinação ou comportamento imprevisível.
    """

    def __init__(self, n_samples: int = 5, threshold: float = 0.25):
        self.n_samples = n_samples
        self.threshold = threshold

    def audit(self, wrapper, prompts: List[str]) -> Dict:
        """
        wrapper : LLMWrapper com .generate(prompt) -> str
        prompts : lista de prompts para testar consistência
        """
        results = []
        for prompt in prompts:
            # gera N respostas com temperatura > 0
            original_temp = getattr(wrapper, 'temperature', 0)
            wrapper.temperature = 0.7
            responses = [wrapper.generate(prompt) for _ in range(self.n_samples)]
            wrapper.temperature = original_temp

            embs = _embed(responses)
            # matriz de similaridade entre todas as respostas
            n = len(embs)
            sims = []
            for i in range(n):
                for j in range(i+1, n):
                    sims.append(_cosine(embs[i], embs[j]))
            mean_sim = float(np.mean(sims)) if sims else 1.0
            inconsistency = 1 - mean_sim

            results.append({
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "mean_similarity": round(mean_sim, 4),
                "inconsistency": round(inconsistency, 4),
                "risk": "ALTO" if inconsistency > self.threshold else "BAIXO",
                "n_samples": self.n_samples,
                "responses_sample": responses[:2]
            })

        mean_inc = float(np.mean([r["inconsistency"] for r in results]))
        alerts = []
        high_risk = [r for r in results if r["risk"] == "ALTO"]
        if high_risk:
            alerts.append({"severity": "ALERTA",
                           "message": f"{len(high_risk)}/{len(results)} prompts com alta "
                                      f"inconsistência (>{self.threshold:.0%}). "
                                      "Modelo responde diferente para a mesma pergunta — "
                                      "comportamento imprevisível em produção."})
        # BUG 21 (v1.52): com lista de prompts vazia, `mean_inc` era
        # np.mean([]) = nan, e o retorno trazia 'mean_inconsistency': nan.
        # Um nan atravessa comparações silenciosamente (nan > 0.25 é False),
        # então o auditor reportava "nenhum risco" sem ter auditado nada.
        if not results:
            return {"results": [], "mean_inconsistency": None,
                    "n_high_risk": 0,
                    "alerts": [{"severity": "CRITICO",
                                "type": "nada_auditado",
                                "message": ("Nenhum prompt auditado. Passe "
                                            "ao menos um prompt em `prompts`.")}],
                    "label": "NAO_AVALIAVEL"}

        return {"results": results, "mean_inconsistency": round(mean_inc, 4),
                "n_high_risk": len(high_risk), "alerts": alerts}


# ── 4. LLMPersonalityDrift — Drift de personalidade entre versões ────────────

class LLMPersonalityDrift:
    """Detecta se o LLM mudou de "personalidade" entre v1 e v2.

    Compara embeddings de respostas a um benchmark de prompts padrão.
    Drift alto = modelo v2 responde de forma estruturalmente diferente do v1.
    """

    def __init__(self, threshold_mmd: float = 0.05):
        self.threshold = threshold_mmd

    def _mmd(self, A: np.ndarray, B: np.ndarray) -> float:
        from sklearn.metrics.pairwise import rbf_kernel
        sigma = float(np.median(np.abs(A - B).sum(axis=1))) + 1e-6
        gamma = 1 / (2 * sigma ** 2)
        return float(rbf_kernel(A, A, gamma).mean()
                     + rbf_kernel(B, B, gamma).mean()
                     - 2 * rbf_kernel(A, B, gamma).mean())

    def compare(self, wrapper_v1, wrapper_v2,
                benchmark_prompts: List[str]) -> Dict:
        """Compara v1 e v2 dado os mesmos prompts de benchmark."""
        resp_v1 = [wrapper_v1.generate(p) for p in benchmark_prompts]
        resp_v2 = [wrapper_v2.generate(p) for p in benchmark_prompts]

        emb_v1 = _embed(resp_v1)
        emb_v2 = _embed(resp_v2)

        # similaridade par a par (mesmo prompt, duas versões)
        pair_sims = [_cosine(emb_v1[i], emb_v2[i]) for i in range(len(benchmark_prompts))]
        mean_sim = float(np.mean(pair_sims))
        mean_div = 1 - mean_sim
        mmd = self._mmd(emb_v1, emb_v2)

        most_changed = sorted(
            [{"prompt": p[:60], "divergence": round(1 - s, 4),
              "response_v1": r1[:80], "response_v2": r2[:80]}
             for p, s, r1, r2 in zip(benchmark_prompts, pair_sims, resp_v1, resp_v2)],
            key=lambda x: x["divergence"], reverse=True)[:3]

        alerts = []
        if mmd > self.threshold:
            alerts.append({"severity": "ALERTA",
                           "message": f"Drift de personalidade detectado entre v1 e v2 "
                                      f"(MMD={mmd:.4f} > {self.threshold}). "
                                      "O modelo mudou de comportamento estruturalmente. "
                                      "Revise o fine-tuning ou os dados de retreinamento."})
        return {"mmd": round(mmd, 4), "mean_divergence": round(mean_div, 4),
                "mean_similarity": round(mean_sim, 4),
                "drift_detected": mmd > self.threshold,
                "most_changed": most_changed, "alerts": alerts}


# ── 5. LLMAuditGuard — Ponto de entrada único ───────────────────────────────

class LLMAuditGuard:
    """Auditoria completa de LLMs em produção.

    Combina:
    - PromptAuditor (antes de gerar)
    - HallucinationMeter (na resposta gerada)
    - ConsistencyAuditor (estabilidade)
    - LLMPersonalityDrift (comparação entre versões)
    """

    def __init__(self, wrapper_v1, wrapper_v2=None):
        self.wrapper = wrapper_v1
        self.wrapper_v2 = wrapper_v2
        self.prompt_auditor = PromptAuditor()
        self.hallucination_meter = HallucinationMeter()
        self.consistency_auditor = ConsistencyAuditor()

    def run(self, prompts: List[str],
            contexts: Optional[List[str]] = None,
            ground_truths: Optional[List[Dict]] = None,
            check_consistency: bool = True,
            check_personality_drift: bool = True) -> Dict:
        """
        prompts        : lista de prompts para auditar
        contexts       : contexto correspondente a cada prompt (RAG/documento)
        ground_truths  : lista de dicts com verdades conhecidas por prompt
        """
        contexts = contexts or [""] * len(prompts)
        ground_truths = ground_truths or [{}] * len(prompts)
        report = {"prompt_audit": [], "hallucination": [],
                  "consistency": {}, "personality_drift": {}, "all_alerts": []}

        for i, (prompt, ctx, gt) in enumerate(zip(prompts, contexts, ground_truths)):
            # 1. Audita o prompt antes de gerar
            pa = self.prompt_auditor.audit_prompt(prompt, ctx)
            report["all_alerts"] += pa["alerts"]

            # 2. Gera a resposta
            response = ""
            if pa["safe"] or not any(a["type"] == "prompt_injection"
                                      for a in pa["alerts"]):
                response = self.wrapper.generate(prompt)
            else:
                response = "[BLOQUEADO: prompt injection detectado]"

            # 3. Mede alucinação
            hm = self.hallucination_meter.score(response, ctx, gt)
            report["all_alerts"] += hm["alerts"]

            report["prompt_audit"].append({
                "prompt": prompt[:60], "context_provided": bool(ctx),
                "prompt_safe": pa["safe"], "response": response[:150],
                "prompt_alerts": pa["alerts"], "details": pa["details"]})
            report["hallucination"].append({
                "prompt": prompt[:60], "response": response[:150],
                "hallucination_score": hm["hallucination_score"],
                "label": hm["label"], "unverified": hm["unverified"],
                "verified": hm["verified"], "alerts": hm["alerts"]})

        # 4. Consistência
        if check_consistency and len(prompts) > 0:
            cons = self.consistency_auditor.audit(self.wrapper, prompts[:3])
            report["consistency"] = cons
            report["all_alerts"] += cons["alerts"]

        # 5. Drift de personalidade (se tiver v2)
        if check_personality_drift and self.wrapper_v2:
            drift = LLMPersonalityDrift()
            pd_result = drift.compare(self.wrapper, self.wrapper_v2, prompts)
            report["personality_drift"] = pd_result
            report["all_alerts"] += pd_result["alerts"]

        # Health score de LLM
        penalties = 0
        for h in report["hallucination"]:
            penalties += int(h["hallucination_score"] * 30)
        for a in report["all_alerts"]:
            if a["severity"] == "CRITICO":
                penalties += 15
            elif a["severity"] == "ALERTA":
                penalties += 5
        report["llm_health_score"] = max(0, 100 - penalties)
        report["n_alerts"] = len(report["all_alerts"])
        return report

    def narrate(self, report: Dict) -> str:
        lines = ["=" * 56, "MODELGUARD — LLM AUDIT", "=" * 56]

        score = report.get("llm_health_score", "?")
        s_color = "SAUDÁVEL" if score >= 80 else "ATENÇÃO" if score >= 60 else "CRÍTICO"
        lines.append(f"\nLLM Health Score: {score}/100 [{s_color}]")

        # Prompt audit
        injections = [p for p in report.get("prompt_audit", [])
                      if not p["prompt_safe"]]
        if injections:
            lines.append(f"\n🔴 Prompt injection detectado em {len(injections)} prompt(s)")

        # Alucinação
        lines.append(f"\n[Alucinação]")
        for h in report.get("hallucination", []):
            icon = "🔴" if h["label"] == "ALTO" else "🟡" if h["label"] == "MODERADO" else "✅"
            lines.append(f"  {icon} '{h['prompt']}' → score={h['hallucination_score']:.0%} [{h['label']}]")
            if h["unverified"]:
                lines.append(f"     Suspeitas: {h['unverified'][:2]}")

        # Consistência
        cons = report.get("consistency", {})
        if cons:
            lines.append(f"\n[Consistência] inconsistência média={cons.get('mean_inconsistency',0):.0%} "
                         f"| {cons.get('n_high_risk',0)} prompt(s) instável(is)")

        # Drift
        drift = report.get("personality_drift", {})
        if drift:
            d = drift.get("drift_detected", False)
            lines.append(f"\n[Drift v1→v2] MMD={drift.get('mmd','?')} → "
                         f"{'DRIFT DETECTADO' if d else 'estável'}")

        lines.append(f"\nTotal alertas: {report.get('n_alerts', 0)}")
        crits = [a for a in report.get("all_alerts", []) if a["severity"] == "CRITICO"]
        if crits:
            lines.append("Alertas críticos:")
            for a in crits[:3]:
                lines.append(f"  🔴 {a['message'][:100]}")
        lines.append("=" * 56)
        return "\n".join(lines)
