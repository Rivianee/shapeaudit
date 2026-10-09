# Rodar no Colab — passo a passo

O notebook `ShapeAudit_Colab.ipynb` roda do começo ao fim e termina com o laudo
**aparecendo dentro do próprio notebook**, idêntico ao HTML que você abre no
navegador. Rodei ele inteiro aqui antes de entregar.

## Passo a passo

1. Abra <https://colab.research.google.com> → **Arquivo → Fazer upload de
   notebook** → escolha `ShapeAudit_Colab.ipynb`.
2. **Ambiente de execução → Executar tudo.** Leva cerca de dois minutos.
3. Role até a seção 6: o laudo aparece renderizado ali.

Não precisa de conta paga, GPU nem configuração.

## O que cada seção faz

| # | seção | o que acontece |
|---|---|---|
| 1 | Instalar | `pip install shapeaudit` |
| 2 | Os dados | baixa o churn de telecom (7.043 clientes, dados reais da IBM) |
| 3 | Treinar | duas versões: floresta e gradient boosting |
| 4 | Auditar | `Guard().audit(...)` → veredito |
| 5 | Model diff | tabela de peso e comportamento por variável |
| 6 | **O laudo** | `display(HTML(r.to_html()))` — o laudo dentro do notebook |
| 7 | Salvar | baixa `laudo.html` |
| 8 | Impacto na decisão | mede quantos clientes trocam no top 10% |
| 9 | Calibração | calcula o limiar do seu domínio |
| 10 | **Pipeline quebrado** | uma coluna chega constante: o laudo diz **Não promova** e nomeia a variável |

## O resultado que você vai ver

Rodando agora, com esses dados:

```
versão 1: 0.815
versão 2: 0.808
diferença: 0.0075

VEREDITO: REVISAR   (decidido por: shape)

Contract        20% → 30%   +10.0 pp
MonthlyCharges   8% → 14%    +5.9 pp
tenure          13% → 11%    -1.9 pp   mesma forma, curva diferente

58 de 281 clientes selecionados mudaram (20.6%)
limiar calibrado para este domínio: 0.236
```

**As duas versões diferem 0,7% em acurácia e trocam 20,6% da fila de
retenção.** É o argumento do trabalho inteiro numa execução de dois minutos.

E na última seção, o caso que todo mundo teme:

```
versão boa:      0.815
versão quebrada: 0.807
diferença:       0.0085      ← quase nada

VEREDITO: BLOQUEADO
comece por: ['Contract']
```

Uma coluna chegou constante depois de uma mudança no pipeline. **Um gate por
métrica aprovaria essa versão.** O laudo diz "Não promova", nomeia a variável e
lista o que investigar primeiro.

## Para usar a sua planilha

Na seção 2, suba o arquivo pelo painel da esquerda do Colab e troque:

```python
d = pd.read_excel('meus_dados.xlsx')
ALVO = 'nome_da_coluna_alvo'
```

E na seção 3, troque as duas linhas de treino pelos **seus** modelos já
treinados — o que está em produção e o candidato.

## Fora do Colab

O mesmo fluxo, sem notebook:

```bash
pip install shapeaudit openpyxl
python demo_excel.py                        # usa exemplo_churn.xlsx
python demo_excel.py meus_dados.xlsx alvo   # usa a sua
```
