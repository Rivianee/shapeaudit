"""
Módulos experimentais — sem validação empírica suficiente para produção.

Mantidos para pesquisa e reprodutibilidade dos experimentos documentados.
Não expostos no namespace principal de `modelcheck`.

  calibration.py  nunca validado; substituído por
                  ImportanceCollapseDetector.calibrate()
  topology.py     implementação caseira de homologia; superada por
                  universal.TopologicalDriftGudhi
  zest.py         validado apenas contra ground truth de família de
                  arquitetura, sem ground truth de divergência
  rhale.py        um teste sintético controlado
  gtsf.py         framework agregador; depende dos acima

  llm.py          LLMWrapper, HallucinationDetector, LLMFairness,
                  EUAIActChecker — 429 linhas, ZERO teste. Um checklist de
                  conformidade regulatória sem validação é passivo, não
                  ativo: não deve estar num pacote instalável.
  llm_audit.py    PromptAuditor, HallucinationMeter, ConsistencyAuditor,
                  LLMPersonalityDrift — 487 linhas, ZERO teste.
  vision.py       VisionShapExplainer, EmbeddingDrift, VisionFairness,
                  SaliencyDiff — 330 linhas, ZERO teste.

Os três acima foram movidos na v1.43. Total: 1.246 linhas com claims fortes
(alucinação, fairness, conformidade EU AI Act) e nenhuma validação de ground
truth. Voltam ao pacote principal quando tiverem benchmark medido.
"""
