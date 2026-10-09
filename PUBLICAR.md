# Publicar — passo a passo

Três coisas para publicar, nesta ordem: o repositório, a página da demo e o
pacote. O repositório vem primeiro porque o endereço dele entra no pacote.

---

## 1. O repositório no GitHub

Crie um repositório público chamado `shapeaudit`. Depois, no seu computador:

```bash
cd shapeaudit_FINAL          # a pasta descompactada
git init
git add .
git commit -m "ShapeAudit 1.0.2"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/shapeaudit.git
git push -u origin main
```

**Antes do commit**, apague o que não deve ir junto: ambientes virtuais,
`__pycache__`, `dist/`, `build/` e qualquer arquivo de dados pesado.

Organize a raiz assim — é a ordem em que uma pessoa nova precisa das coisas:

| arquivo | para quê |
|---|---|
| `README.md` | a porta de entrada (é o mesmo texto da página do PyPI) |
| `demo/` | a demonstração visual — vira a página do projeto no passo 2 |
| `COMO_USAR.md` | passo a passo em nove etapas |
| `COLAB.md` | como rodar no Colab |
| `LIMITATIONS.md` | os limites, cada um com o número medido |
| `estudo_de_caso.py`, `demo_excel.py`, `demo_vinho.py` | as demonstrações |
| `exemplo_churn.xlsx` | planilha de exemplo (7.043 clientes) |
| `ShapeAudit_Colab.ipynb` | o notebook |
| `modelcheck/`, `shapeaudit/`, `modelguard/`, `tests/` | o código |

No README do repositório, ponha o link da demo logo no começo — é o que
convence em dez segundos.

---

## 2. A página da demo (GitHub Pages)

A pasta `demo/` é HTML puro, então o GitHub publica direto, sem build:

1. No repositório: **Settings → Pages**.
2. Em *Source*, escolha **Deploy from a branch**.
3. Branch `main`, pasta `/ (root)`. Salve.
4. Em um ou dois minutos a página fica no ar em:
   `https://SEU_USUARIO.github.io/shapeaudit/demo/`

Para um endereço mais curto, mova o conteúdo de `demo/` para uma pasta `docs/`
e escolha `/docs` no passo 3.

**Confira depois de publicar:** abra o link no celular. A página foi feita para
funcionar em tela pequena, e vale ver antes de mandar para alguém.

---

## 3. O pacote no PyPI

### Antes de tudo: trocar o marcador

No `pyproject.toml`:

```toml
Homepage = "https://github.com/SEU_USUARIO/shapeaudit"
```

Esse é o link que aparece na página do PyPI. Um link quebrado é a primeira
coisa que alguém nota.

### Publicar

```bash
cd shapeaudit_FINAL

python -m build                 # gera dist/shapeaudit-1.0.2-*.whl e .tar.gz
python -m twine check dist/*    # os dois precisam dizer PASSED

python -m twine upload --repository testpypi dist/*    # ensaio
python -m twine upload dist/*                          # valendo
```

No ensaio, **abra a página e leia como se fosse a primeira vez**. É o melhor
momento para pegar texto confuso, antes de queimar a versão.

`shapeaudit` é nome novo, então o upload usa o seu token normal do PyPI
(Account settings → API tokens). Não depende da conta do `modelguard-ml`.

### Conferir

```bash
pip install shapeaudit
python -c "import shapeaudit; print(shapeaudit.__version__)"   # 1.0.2
shapeaudit --help
```

---

## As demonstrações, e quando usar cada uma

| demo | quando |
|---|---|
| **`demo/index.html`** | **na apresentação.** Abre offline, tem o momento de revelação e mostra os três laudos sem rodar nada |
| **`ShapeAudit_Colab.ipynb`** | para mandar depois. A pessoa roda do zero, sem instalar nada |
| **`estudo_de_caso.py`** | para gerar os laudos de novo, ou com outros dados |
| **`demo_excel.py`** | para quem quer usar com a própria planilha |
| **`demo_vinho.py`** | plano B sem internet: dados embutidos, 12 segundos |

## Para os 10 minutos

| tempo | o quê |
|---|---|
| 0:00–1:30 | o problema, com o número da fila de retenção |
| 1:30–2:30 | por que ferramentas de deriva não veem (Evidently: zero colunas) |
| 2:30–6:30 | a demo: a tabela, a revelação, e os três laudos |
| 6:30–8:30 | os números de validação (161 retreinos, nenhum bloqueio falso) |
| 8:30–10:00 | limites declarados e fechamento |

No palco, abra a página **já pronta** e deixe o Colab como material de apoio.
Se quiser rodar algo ao vivo, rode **uma** célula só: a da auditoria do caso
quebrado.
