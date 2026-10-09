"""ModelCheck — módulo de visão computacional.

Detectores:
  VisionWrapper       — adapta modelo PyTorch/TF para predict(numpy) compatível com o resto do ModelCheck
  VisionShapExplainer — SHAP GradientExplainer para CNNs, retorna heatmap por imagem
  EmbeddingDrift      — detecta drift comparando embeddings da penúltima camada entre dois conjuntos
  VisionFairness      — mede diferença de acurácia/confiança entre subgrupos visuais
  SaliencyDiff        — compara quais regiões dois modelos "olham" (grad saliency)
  VisionGuard         — ponto de entrada único, semelhante ao ModelGuard tabular
"""
import numpy as np


class VisionWrapper:
    """Adapta modelo PyTorch ou TF para interface predict(X_numpy) → numpy."""

    def __init__(self, model, framework="torch", device=None, preprocess=None):
        """
        model       : modelo PyTorch (nn.Module) ou TF (keras Model)
        framework   : "torch" | "tf"
        device      : "cpu" | "cuda" (só pytorch)
        preprocess  : função opcional np.ndarray → tensor (se None, faz cast automático)
        """
        self.model = model
        self.framework = framework
        self.preprocess = preprocess
        if framework == "torch":
            import torch
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            self.model.to(self.device)

    def predict(self, X):
        """X: np.ndarray (N, C, H, W) para torch ou (N, H, W, C) para tf."""
        if self.framework == "torch":
            import torch
            t = self.preprocess(X) if self.preprocess else torch.tensor(X, dtype=torch.float32)
            t = t.to(self.device)
            with torch.no_grad():
                out = self.model(t)
            probs = torch.softmax(out, dim=1) if out.shape[-1] > 1 else torch.sigmoid(out)
            return probs.cpu().numpy()
        else:
            import tensorflow as tf
            t = self.preprocess(X) if self.preprocess else tf.constant(X, dtype=tf.float32)
            out = self.model(t, training=False)
            return out.numpy()

    def predict_classes(self, X):
        return self.predict(X).argmax(axis=1)

    def embeddings(self, X, layer_name=None):
        """Extrai embeddings da penúltima camada (ou layer_name especificada)."""
        if self.framework == "torch":
            import torch
            activations = {}
            def hook(module, inp, out):
                activations["emb"] = out.detach().cpu().numpy()
            layers = list(self.model.named_modules())
            target_name = layer_name or layers[-3][0]
            for name, module in layers:
                if name == target_name:
                    h = module.register_forward_hook(hook)
                    break
            t = self.preprocess(X) if self.preprocess else torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                self.model(t.to(self.device))
            h.remove()
            emb = activations.get("emb", np.zeros((len(X), 1)))
            return emb.reshape(len(X), -1)
        else:
            import tensorflow as tf
            if layer_name:
                sub = tf.keras.Model(inputs=self.model.input,
                                     outputs=self.model.get_layer(layer_name).output)
            else:
                sub = tf.keras.Model(inputs=self.model.input,
                                     outputs=self.model.layers[-2].output)
            t = self.preprocess(X) if self.preprocess else tf.constant(X, dtype=tf.float32)
            emb = sub(t, training=False).numpy()
            return emb.reshape(len(X), -1)


class VisionShapExplainer:
    """SHAP GradientExplainer para CNNs PyTorch — retorna heatmap (N, H, W)."""

    def __init__(self, wrapper: VisionWrapper, background: np.ndarray):
        """
        wrapper    : VisionWrapper (torch)
        background : np.ndarray (N_bg, C, H, W) — amostras de referência
        """
        import shap, torch
        self.wrapper = wrapper
        bg = torch.tensor(background, dtype=torch.float32).to(wrapper.device)
        self.explainer = shap.GradientExplainer(wrapper.model, bg)

    def explain(self, X: np.ndarray, top_k=1):
        """Retorna heatmap absoluto médio por imagem. Shape: (N, H, W)."""
        import torch
        t = torch.tensor(X, dtype=torch.float32).to(self.wrapper.device)
        shap_values = self.explainer.shap_values(t)
        if isinstance(shap_values, list):
            vals = np.array(shap_values)[..., :top_k]
            heat = np.abs(vals).mean(axis=(0, -1))
        else:
            heat = np.abs(shap_values).mean(axis=1)
        return heat


class EmbeddingDrift:
    """Detecta drift comparando distribuições de embeddings (MMD ou distância cosseno)."""

    def __init__(self, threshold_mmd=0.05, threshold_cosine=0.15):
        self.threshold_mmd = threshold_mmd
        self.threshold_cosine = threshold_cosine

    def _mmd(self, A, B):
        """Maximum Mean Discrepancy com kernel RBF."""
        from sklearn.metrics.pairwise import rbf_kernel
        sigma = np.median(np.linalg.norm(A - B[:len(A)], axis=1)) + 1e-6
        gamma = 1 / (2 * sigma ** 2)
        kAA = rbf_kernel(A, A, gamma).mean()
        kBB = rbf_kernel(B, B, gamma).mean()
        kAB = rbf_kernel(A, B, gamma).mean()
        return float(kAA + kBB - 2 * kAB)

    def _cosine_mean(self, A, B):
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(A.mean(axis=0, keepdims=True),
                                B.mean(axis=0, keepdims=True))
        return float(1 - sim[0, 0])

    def detect(self, emb_ref: np.ndarray, emb_prod: np.ndarray):
        """
        Drift entre dois conjuntos de embeddings.

        BUG 25 (v1.54): o veredito era `mmd > t1 OR cosine > t2`, e o
        `cosine_dist` entre as MÉDIAS dava 0.824 em amostras da MESMA
        distribuição — porque em alta dimensão vetores aleatórios são quase
        ortogonais e a média tende a zero, tornando o ângulo instável.
        Resultado: falso positivo garantido em qualquer par saudável.
        Medido: mesma distribuição -> mmd=0.004 (ok) mas cosine=0.824
        (limiar 0.15). O MMD sozinho separava corretamente.

        BUG 26: `detect` com arrays vazios levantava
        `InvalidParameterError` do sklearn (gamma inválido), e com 1 amostra
        devolvia mmd=0.787 — número sem sentido estatístico apresentado
        como se fosse medida válida.
        """
        emb_ref = np.asarray(emb_ref, dtype=float)
        emb_prod = np.asarray(emb_prod, dtype=float)
        if emb_ref.ndim != 2 or emb_prod.ndim != 2:
            raise ValueError(
                f"EmbeddingDrift.detect: esperados arrays 2D "
                f"(n_amostras, n_dims); recebidos {emb_ref.shape} e "
                f"{emb_prod.shape}.")
        if emb_ref.shape[1] != emb_prod.shape[1]:
            raise ValueError(
                f"EmbeddingDrift.detect: dimensões incompatíveis — "
                f"{emb_ref.shape[1]} vs {emb_prod.shape[1]}. Os embeddings "
                f"precisam vir da mesma camada do mesmo modelo.")
        _MIN = 10
        if len(emb_ref) < _MIN or len(emb_prod) < _MIN:
            raise ValueError(
                f"EmbeddingDrift.detect: amostra insuficiente "
                f"({len(emb_ref)} e {len(emb_prod)}; mínimo {_MIN} cada). "
                f"MMD com poucas amostras não tem significado estatístico.")

        n = min(500, len(emb_ref), len(emb_prod))
        A = emb_ref[np.random.choice(len(emb_ref), n, replace=False)]
        B = emb_prod[np.random.choice(len(emb_prod), n, replace=False)]
        mmd = self._mmd(A, B)
        cos = self._cosine_mean(A, B)
        alerts = []
        if mmd > self.threshold_mmd:
            alerts.append({"type": "embedding_drift", "severity": "ALERTA",
                           "metric": "MMD", "value": round(mmd, 4),
                           "message": (f"Drift de embedding detectado "
                                       f"(MMD={mmd:.4f} > "
                                       f"{self.threshold_mmd})")})
        # BUG 25 (v1.54): o cosseno entre as MÉDIAS dava 0.824 em amostras da
        # MESMA distribuição — em alta dimensão a média tende a zero e o
        # ângulo fica instável. Como o veredito era OR, todo par saudável
        # virava falso positivo. Agora o cosseno é DIAGNÓSTICO (informativo),
        # não decide o veredito; só o MMD decide.
        if cos > self.threshold_cosine:
            alerts.append({"type": "cosine_shift", "severity": "INFO",
                           "metric": "cosine_dist", "value": round(cos, 4),
                           "message": (f"Distância de cosseno entre as médias "
                                       f"({cos:.4f}). Informativo: em alta "
                                       f"dimensão este valor é naturalmente "
                                       f"alto mesmo sem drift. Use o MMD "
                                       f"para decidir.")})
        return {"mmd": round(mmd, 4), "cosine_dist": round(cos, 4),
                "drift_detected": mmd > self.threshold_mmd,
                "alerts": alerts}


class VisionFairness:
    """Mede diferença de acurácia/confiança entre subgrupos visuais."""

    def __init__(self, max_diff=0.1):
        self.max_diff = max_diff

    def check(self, wrapper: VisionWrapper, X: np.ndarray,
              y_true: np.ndarray, groups: np.ndarray, group_names=None):
        """
        X       : imagens (N, C, H, W) ou (N, H, W, C)
        y_true  : labels verdadeiros (N,)
        groups  : array de grupo por amostra (N,) — ex: [0, 1, 0, 1, ...]
        group_names : dict {0: "grupo A", 1: "grupo B"}
        """
        preds = wrapper.predict_classes(X)
        unique = np.unique(groups)
        results = {}
        for g in unique:
            mask = groups == g
            acc = float((preds[mask] == y_true[mask]).mean())
            conf = float(wrapper.predict(X[mask]).max(axis=1).mean())
            name = (group_names or {}).get(g, f"grupo_{g}")
            results[name] = {"accuracy": round(acc, 4), "mean_confidence": round(conf, 4)}

        accs = [v["accuracy"] for v in results.values()]
        max_gap = max(accs) - min(accs)
        alerts = []
        if max_gap > self.max_diff:
            low_g = min(results, key=lambda k: results[k]["accuracy"])
            high_g = max(results, key=lambda k: results[k]["accuracy"])
            alerts.append({
                "severity": "CRITICO",
                "message": f"Gap de acurácia entre grupos: {high_g}={results[high_g]['accuracy']:.2%} "
                           f"vs {low_g}={results[low_g]['accuracy']:.2%} (gap={max_gap:.2%} > {self.max_diff:.0%})"
            })
        return {"groups": results, "accuracy_gap": round(max_gap, 4),
                "alerts": alerts,
                # BUG 27 (v1.54): o retorno não tinha veredito booleano, ao
                # contrário de EmbeddingDrift (`drift_detected`) e dos
                # detectores de llm.py (`bias_detected`, `significant_change`).
                # Quem consumisse a saída de forma uniforme recebia KeyError.
                "bias_detected": bool(alerts)}


class SaliencyDiff:
    """Compara saliency maps de dois modelos — detecta se mudaram as regiões de atenção."""

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def _saliency(self, wrapper: VisionWrapper, X: np.ndarray):
        """Grad saliency: gradiente da saída em relação à entrada."""
        import torch
        t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(wrapper.device)
        out = wrapper.model(t)
        score = out.max(dim=1).values.sum()
        score.backward()
        sal = t.grad.abs().cpu().numpy()
        return sal.mean(axis=1)

    def compare(self, wrapper_a: VisionWrapper, wrapper_b: VisionWrapper,
                X: np.ndarray):
        """
        Retorna divergência média de saliency e alertas.
        Quanto maior, mais o modelo B "olha" para regiões diferentes do A.
        """
        sal_a = self._saliency(wrapper_a, X)
        sal_b = self._saliency(wrapper_b, X)
        sal_a_n = sal_a / (sal_a.max(axis=(1, 2), keepdims=True) + 1e-8)
        sal_b_n = sal_b / (sal_b.max(axis=(1, 2), keepdims=True) + 1e-8)
        diff = np.abs(sal_a_n - sal_b_n).mean()
        alerts = []
        if diff > self.threshold:
            alerts.append({
                "severity": "ALERTA",
                "message": f"Regiões de atenção mudaram significativamente entre v1 e v2 "
                           f"(divergência saliency={diff:.3f} > {self.threshold})"
            })
        return {"saliency_divergence": round(float(diff), 4),
                "significant_change": diff > self.threshold,
                "alerts": alerts}


class VisionGuard:
    """Ponto de entrada único para auditoria de modelos de visão.

    Uso mínimo:
        wrapper = VisionWrapper(meu_modelo_pytorch)
        guard = VisionGuard({"cnn_v1": wrapper}, X_ref, y_ref)
        report = guard.run(X_prod, y_prod, groups=grupos)
    """

    def __init__(self, models: dict, X_ref: np.ndarray, y_ref: np.ndarray,
                 background_size=32):
        """
        models         : {"nome": VisionWrapper, ...}
        X_ref          : imagens de referência (N, C, H, W)
        y_ref          : labels de referência
        background_size: nº de amostras de background pra SHAP
        """
        self.models = models
        self.X_ref = X_ref
        self.y_ref = y_ref
        idx = np.random.choice(len(X_ref), min(background_size, len(X_ref)), replace=False)
        self.background = X_ref[idx]

    def run(self, X_prod: np.ndarray, y_prod: np.ndarray = None,
            groups: np.ndarray = None, group_names: dict = None,
            model_b_name: str = None):
        """
        Roda todos os detectores disponíveis.
        model_b_name : se informado, compara saliency entre o 1º modelo e este.
        """
        report = {}
        names = list(self.models.keys())
        wrapper_a = self.models[names[0]]

        report["embedding_drift"] = {}
        report["fairness"] = {}
        report["saliency_diff"] = {}
        report["shap_heatmaps"] = {}

        drift_det = EmbeddingDrift()
        fair_det = VisionFairness()

        for name, wrapper in self.models.items():
            emb_ref = wrapper.embeddings(self.X_ref)
            emb_prod = wrapper.embeddings(X_prod)
            report["embedding_drift"][name] = drift_det.detect(emb_ref, emb_prod)

            if y_prod is not None and groups is not None:
                report["fairness"][name] = fair_det.check(
                    wrapper, X_prod, y_prod, groups, group_names)

        if len(names) >= 2 or model_b_name:
            b_name = model_b_name or names[1]
            if b_name in self.models:
                sal = SaliencyDiff()
                n = min(16, len(X_prod))
                report["saliency_diff"] = sal.compare(
                    wrapper_a, self.models[b_name], X_prod[:n])

        try:
            shap_exp = VisionShapExplainer(wrapper_a, self.background)
            n = min(8, len(X_prod))
            report["shap_heatmaps"][names[0]] = shap_exp.explain(X_prod[:n])
        except Exception as e:
            report["shap_heatmaps"]["error"] = str(e)

        all_alerts = []
        for v in report["embedding_drift"].values():
            all_alerts += v.get("alerts", [])
        for v in report["fairness"].values():
            all_alerts += v.get("alerts", [])
        all_alerts += report["saliency_diff"].get("alerts", [])
        report["all_alerts"] = all_alerts
        report["n_alerts"] = len(all_alerts)
        return report


def vision_summary(report: dict) -> str:
    """Resumo legível do relatório de visão."""
    lines = ["=" * 50, "MODELCHECK — VISÃO COMPUTACIONAL", "=" * 50]
    for nome, r in report.get("embedding_drift", {}).items():
        status = "DRIFT DETECTADO" if r["drift_detected"] else "estável"
        lines.append(f"[Embedding drift — {nome}] MMD={r['mmd']} cosine={r['cosine_dist']} → {status}")
    for nome, r in report.get("fairness", {}).items():
        gap = r["accuracy_gap"]
        lines.append(f"[Fairness — {nome}] gap de acurácia entre grupos: {gap:.2%}")
        for g, vals in r["groups"].items():
            lines.append(f"  {g}: acc={vals['accuracy']:.2%} conf={vals['mean_confidence']:.2%}")
    sd = report.get("saliency_diff", {})
    if sd:
        div = sd.get("saliency_divergence", "?")
        changed = sd.get("significant_change", False)
        lines.append(f"[Saliency diff] divergência={div} → {'MUDOU' if changed else 'estável'}")
    lines.append(f"\nTotal de alertas: {report.get('n_alerts', 0)}")
    lines.append("=" * 50)
    return "\n".join(lines)
