"""ModelCheck — módulo LLM.

Detectores:
  LLMWrapper          — interface unificada para qualquer LLM (OpenAI, HF, local, callable)
  ResponseEmbedder    — converte respostas em embeddings semânticos (sentence-transformers)
  LLMDrift            — detecta se o modelo mudou de comportamento entre versões (MMD em embeddings)
  HallucinationDetector — detecta alucinação por inconsistência entre múltiplas respostas
  LLMFairness         — mede se respostas mudam ao variar atributo protegido no prompt
  LLMDiff             — compara dois modelos/versões semanticamente (dado mesmo prompt)
  LLMGuard            — ponto de entrada único
"""
import re
import numpy as np
from typing import Callable, List, Optional


class LLMWrapper:
    """Interface unificada para qualquer LLM.

    Aceita:
      fn        : callable(prompt: str) -> str  (qualquer função que gera texto)
      model_id  : str do HuggingFace ("gpt2", "mistralai/Mistral-7B-v0.1" etc.)
      client    : objeto OpenAI/Anthropic já configurado (passa model_name junto)
    """

    def __init__(self, fn: Callable = None, model_id: str = None,
                 client=None, model_name: str = None,
                 max_new_tokens: int = 128, temperature: float = 0.0):
        self.fn = fn
        self.model_id = model_id
        self.client = client
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._pipe = None

        if model_id and not fn and not client:
            self._load_hf(model_id)

    def _load_hf(self, model_id):
        from transformers import pipeline
        self._pipe = pipeline("text-generation", model=model_id,
                              max_new_tokens=self.max_new_tokens,
                              temperature=self.temperature or 0.01,
                              do_sample=self.temperature > 0)

    def generate(self, prompt: str) -> str:
        if self.fn:
            return self.fn(prompt)
        if self._pipe:
            out = self._pipe(prompt)
            text = out[0]["generated_text"]
            return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        if self.client:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature)
                return resp.choices[0].message.content
            except Exception:
                resp = self.client.messages.create(
                    model=self.model_name, max_tokens=self.max_new_tokens,
                    messages=[{"role": "user", "content": prompt}])
                return resp.content[0].text
        raise ValueError("LLMWrapper: informe fn, model_id ou client.")

    def generate_batch(self, prompts: List[str]) -> List[str]:
        return [self.generate(p) for p in prompts]


class ResponseEmbedder:
    """Converte textos em embeddings semânticos.

    Usa sentence-transformers se disponível e com acesso à internet.
    Fallback automático para TF-IDF (funciona offline, sem dependências extras).
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", force_tfidf=False):
        self._st = None
        self._tfidf = None
        if not force_tfidf:
            try:
                from sentence_transformers import SentenceTransformer
                self._st = SentenceTransformer(model_name)
            except Exception:
                pass
        if self._st is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=512, sublinear_tf=True)
            self._fitted = False

    def embed(self, texts: List[str]) -> np.ndarray:
        if self._st:
            return self._st.encode(texts, normalize_embeddings=True,
                                   show_progress_bar=False)
        if not self._fitted:
            embs = self._tfidf.fit_transform(texts).toarray().astype("float32")
            self._fitted = True
        else:
            embs = self._tfidf.transform(texts).toarray().astype("float32")
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8
        return embs / norms

    def similarity(self, a: str, b: str) -> float:
        embs = self.embed([a, b])
        return float(np.dot(embs[0], embs[1]))


class LLMDrift:
    """Detecta drift de comportamento entre dois conjuntos de respostas."""

    def __init__(self, embedder: ResponseEmbedder = None, threshold_mmd=0.05):
        self.embedder = embedder or ResponseEmbedder()
        self.threshold_mmd = threshold_mmd

    def _mmd(self, A, B):
        from sklearn.metrics.pairwise import rbf_kernel
        sigma = float(np.median(np.linalg.norm(A[:, None] - B[None, :], axis=-1))) + 1e-6
        gamma = 1 / (2 * sigma ** 2)
        return float(rbf_kernel(A, A, gamma).mean()
                     + rbf_kernel(B, B, gamma).mean()
                     - 2 * rbf_kernel(A, B, gamma).mean())

    def detect(self, responses_ref: List[str], responses_prod: List[str]):
        """
        MMD entre dois conjuntos de respostas.

        LIMITAÇÃO MEDIDA (bug 23, v1.53) — leia antes de usar o veredito:

        Sem `sentence-transformers` instalado, o `ResponseEmbedder` cai em
        TF-IDF ajustado no PRÓPRIO lote. Isso torna o MMD uma medida de
        sobreposição LEXICAL, não semântica, e o resultado depende do
        vocabulário do lote — não só do conteúdo.

        Calibração medida (3 temas, 30 pares, modo TF-IDF):

            mesmo tema, parafraseado : média 0.123, max 0.310
            temas diferentes         : média 0.296, min 0.000

        As distribuições SE SOBREPÕEM. Não existe limiar que separe:
        com 0.05 (padrão) 67% dos pares saudáveis viram falso positivo;
        com 0.31 o recall cai para 58%.

        Com `sentence-transformers` instalado o embedding é semântico e o
        MMD passa a medir o que promete. Sem ele, trate `drift_detected`
        como indicativo, não como veredito — e prefira comparar o valor
        contra uma linha de base do seu próprio domínio.
        """

        emb_ref = self.embedder.embed(responses_ref)
        emb_prod = self.embedder.embed(responses_prod)
        mmd = self._mmd(emb_ref, emb_prod)
        mean_sim = float(np.diag(emb_ref @ emb_prod.T).mean())
        alerts = []
        if mmd > self.threshold_mmd:
            alerts.append({"type": "behavior_drift", "severity": "ALERTA", "metric": "MMD",
                           "value": round(mmd, 4),
                           "message": f"Drift de comportamento detectado "
                                      f"(MMD={mmd:.4f} > {self.threshold_mmd}). "
                                      "Respostas em produção divergem da referência."})
        _semantico = getattr(self.embedder, "_st", None) is not None
        if not _semantico:
            alerts = list(alerts) + [{
                "severity": "INFO", "type": "limiar_nao_confiavel",
                "message": ("Embedding em modo TF-IDF lexical (sem "
                            "sentence-transformers). Medido: mesmo tema dá "
                            "MMD até 0.310 e temas diferentes até 0.000 — as "
                            "distribuições se sobrepõem e nenhum limiar "
                            "separa. Trate `drift_detected` como indicativo.")}]
        return {"mmd": round(mmd, 4), "mean_similarity": round(mean_sim, 4),
                "drift_detected": mmd > self.threshold_mmd,
                "embedding_mode": "semantic" if _semantico else "tfidf_lexical",
                "threshold_reliable": _semantico,
                "alerts": alerts}


class HallucinationDetector:
    """Detecta alucinação por inconsistência entre múltiplas respostas ao mesmo prompt.

    Estratégia: gera N respostas com temperatura > 0 e mede dispersão semântica.
    Alta dispersão = modelo incerto = risco de alucinação.
    """

    def __init__(self, embedder: ResponseEmbedder = None,
                 n_samples=5, threshold=0.25):
        self.embedder = embedder or ResponseEmbedder()
        self.n_samples = n_samples
        self.threshold = threshold

    def check(self, wrapper: LLMWrapper, prompts: List[str]):
        """Para cada prompt, gera n_samples respostas e mede inconsistência."""
        original_temp = wrapper.temperature
        wrapper.temperature = 0.7
        results = []
        for prompt in prompts:
            resps = [wrapper.generate(prompt) for _ in range(self.n_samples)]
            embs = self.embedder.embed(resps)
            sim_matrix = embs @ embs.T
            mask = ~np.eye(len(resps), dtype=bool)
            mean_sim = float(sim_matrix[mask].mean())
            inconsistency = 1 - mean_sim
            alert = inconsistency > self.threshold
            results.append({
                "prompt": prompt[:80] + "..." if len(prompt) > 80 else prompt,
                "inconsistency": round(inconsistency, 4),
                "mean_similarity": round(mean_sim, 4),
                "hallucination_risk": "ALTO" if alert else "BAIXO",
                "samples": resps
            })
        wrapper.temperature = original_temp
        alerts = [{"type": "alerta_generico", "severity": "ALERTA",
                   "message": f"Risco de alucinação em: '{r['prompt']}' "
                              f"(inconsistência={r['inconsistency']:.2f})"}
                  for r in results if r["hallucination_risk"] == "ALTO"]
        return {"results": results, "alerts": alerts,
                "n_high_risk": sum(1 for r in results if r["hallucination_risk"] == "ALTO")}


class LLMFairness:
    """Mede se respostas mudam ao variar atributo protegido no prompt.

    Estratégia de counterfactual: mesmo prompt, só muda o atributo protegido.
    Ex: "João pediu empréstimo" vs "Maria pediu empréstimo".
    Divergência semântica alta = viés potencial.
    """

    def __init__(self, embedder: ResponseEmbedder = None, threshold=0.15):
        self.embedder = embedder or ResponseEmbedder()
        self.threshold = threshold

    def check(self, wrapper: LLMWrapper,
              prompt_pairs: List[tuple],
              attribute_name: str = "atributo protegido"):
        """
        prompt_pairs : lista de (prompt_grupo_A, prompt_grupo_B)
        Retorna divergência semântica média e alertas.
        """
        divergences = []
        details = []
        for pa, pb in prompt_pairs:
            ra = wrapper.generate(pa)
            rb = wrapper.generate(pb)
            sim = self.embedder.similarity(ra, rb)
            div = 1 - sim
            divergences.append(div)
            details.append({"prompt_a": pa[:60], "prompt_b": pb[:60],
                             "response_a": ra[:100], "response_b": rb[:100],
                             "divergence": round(div, 4)})
        mean_div = float(np.mean(divergences))
        alerts = []
        if mean_div > self.threshold:
            alerts.append({
                "severity": "CRITICO",
                "message": f"Viés detectado para '{attribute_name}': divergência média "
                           f"entre grupos={mean_div:.3f} > {self.threshold}. "
                           "O modelo responde diferente para grupos distintos."})
        high = [d for d in details if d["divergence"] > self.threshold]
        return {"mean_divergence": round(mean_div, 4),
                "bias_detected": mean_div > self.threshold,
                "details": details, "high_divergence_cases": high,
                "alerts": alerts}


class LLMDiff:
    """Compara dois modelos/versões semanticamente dado o mesmo conjunto de prompts."""

    def __init__(self, embedder: ResponseEmbedder = None, threshold=0.2):
        self.embedder = embedder or ResponseEmbedder()
        self.threshold = threshold

    def compare(self, wrapper_a: LLMWrapper, wrapper_b: LLMWrapper,
                prompts: List[str]):
        """Para cada prompt, compara a resposta do modelo A vs B."""
        resps_a = wrapper_a.generate_batch(prompts)
        resps_b = wrapper_b.generate_batch(prompts)
        embs_a = self.embedder.embed(resps_a)
        embs_b = self.embedder.embed(resps_b)
        sims = np.diag(embs_a @ embs_b.T)
        divs = 1 - sims
        mean_div = float(divs.mean())
        details = [{"prompt": p[:60], "divergence": round(float(d), 4),
                    "response_a": ra[:100], "response_b": rb[:100]}
                   for p, d, ra, rb in zip(prompts, divs, resps_a, resps_b)]
        most_changed = sorted(details, key=lambda x: x["divergence"], reverse=True)[:3]
        alerts = []
        if mean_div > self.threshold:
            alerts.append({
                "severity": "ALERTA",
                "message": f"Divergência semântica média entre versões: {mean_div:.3f} "
                           f"> {self.threshold}. Comportamento mudou significativamente."})
        return {"mean_divergence": round(mean_div, 4),
                "significant_change": mean_div > self.threshold,
                "most_changed_prompts": most_changed,
                "details": details, "alerts": alerts}



class LLMGuard:
    """Ponto de entrada único para auditoria de LLMs.

    Uso mínimo:
        def meu_llm(prompt): return openai_client.complete(prompt)
        wrapper = LLMWrapper(fn=meu_llm)
        guard = LLMGuard(wrapper)
        report = guard.run(prompts_teste, prompts_fairness_pairs=pairs)
        print(llm_summary(report))
    """

    def __init__(self, wrapper_a: LLMWrapper, wrapper_b: LLMWrapper = None,
                 embedder: ResponseEmbedder = None):
        self.wrapper_a = wrapper_a
        self.wrapper_b = wrapper_b
        self.embedder = embedder or ResponseEmbedder()

    def run(self, prompts: List[str],
            responses_ref: Optional[List[str]] = None,
            prompts_fairness: Optional[List[tuple]] = None,
            fairness_attribute: str = "atributo protegido",

            check_hallucination: bool = True,
            n_hallucination_samples: int = 3):
        report = {}

        responses_a = self.wrapper_a.generate_batch(prompts)
        report["responses"] = responses_a

        if responses_ref:
            drift = LLMDrift(self.embedder)
            report["drift"] = drift.detect(responses_ref, responses_a)

        if check_hallucination:
            hall = HallucinationDetector(self.embedder,
                                         n_samples=n_hallucination_samples)
            report["hallucination"] = hall.check(self.wrapper_a, prompts[:5])

        if prompts_fairness:
            fair = LLMFairness(self.embedder)
            report["fairness"] = fair.check(self.wrapper_a, prompts_fairness,
                                            fairness_attribute)

        if self.wrapper_b:
            diff = LLMDiff(self.embedder)
            report["diff"] = diff.compare(self.wrapper_a, self.wrapper_b, prompts)

        # EUAIActChecker removido do pacote na v1.55 — não é validável por
        # ground truth (verificar se um checklist reflete um regulamento é
        # revisão jurídica). Arquivado em rascunhos-juridicos/.

        all_alerts = []
        for key in ("drift", "hallucination", "fairness", "diff"):
            section = report.get(key, {})
            all_alerts += section.get("alerts", [])
        report["all_alerts"] = all_alerts
        report["n_alerts"] = len(all_alerts)
        return report


def llm_summary(report: dict) -> str:
    lines = ["=" * 52, "MODELCHECK — LLMs", "=" * 52]

    drift = report.get("drift", {})
    if drift:
        status = "DRIFT DETECTADO" if drift["drift_detected"] else "estável"
        lines.append(f"[Drift]         MMD={drift['mmd']} sim={drift['mean_similarity']} → {status}")

    hall = report.get("hallucination", {})
    if hall:
        lines.append(f"[Alucinação]    {hall['n_high_risk']} prompt(s) com risco ALTO")
        for r in hall.get("results", []):
            lines.append(f"  '{r['prompt']}' → inconsistência={r['inconsistency']} [{r['hallucination_risk']}]")

    fair = report.get("fairness", {})
    if fair:
        status = "VIÉS DETECTADO" if fair["bias_detected"] else "sem viés detectado"
        lines.append(f"[Fairness]      divergência média={fair['mean_divergence']} → {status}")

    diff = report.get("diff", {})
    if diff:
        status = "MUDOU" if diff["significant_change"] else "estável"
        lines.append(f"[Diff v1→v2]    divergência={diff['mean_divergence']} → {status}")
        for c in diff.get("most_changed_prompts", [])[:2]:
            lines.append(f"  prompt mais diferente: '{c['prompt']}' (div={c['divergence']})")


    lines.append(f"\nTotal alertas: {report.get('n_alerts', 0)}")
    crit = [a for a in report.get("all_alerts", []) if a.get("severity") == "CRITICO"]
    if crit:
        lines.append("Alertas críticos:")
        for a in crit:
            lines.append(f"  ! {a['message'][:100]}")
    lines.append("=" * 52)
    return "\n".join(lines)
