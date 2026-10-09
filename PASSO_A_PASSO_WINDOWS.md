# Publicar o ShapeAudit no PyPI — passo a passo no Windows

Escrito para fazer do começo ao fim no PowerShell, sem supor nada.
Cada passo tem o que digitar e o que você deve ver.

Tempo: uns 20 minutos, a maior parte esperando instalação.

---

## Passo 1 — Baixar e descompactar

1. Nesta conversa, baixe o arquivo **`shapeaudit-repo.zip`**. Ele vai para a
   pasta **Downloads**.
2. Abra o Explorador de Arquivos, vá em Downloads, **clique com o botão
   direito** no arquivo → **Extrair tudo…** → **Extrair**.
3. Vai aparecer uma pasta `shapeaudit-repo`, e dentro dela outra pasta
   `shapeaudit`. **É a de dentro que interessa.**

---

## Passo 2 — Abrir o PowerShell na pasta certa

Entre na pasta `shapeaudit` (a de dentro, onde você vê o arquivo
`pyproject.toml`). Clique na **barra de endereço** do Explorador, apague o que
está escrito, digite `powershell` e aperte **Enter**.

Abre uma janela preta já na pasta certa. Confirme com:

```powershell
dir pyproject.toml
```

**Você deve ver** uma linha com `pyproject.toml` e a data. Se disser
*"não foi possível encontrar"*, você está na pasta errada — volte e entre na
`shapeaudit` de dentro.

---

## Passo 3 — Conferir que o Python responde

```powershell
py --version
```

**Esperado:** algo como `Python 3.11.x`. Se der erro, use `python --version`.
Se nenhum dos dois funcionar, o Python não está no PATH — me avise que
resolvemos isso antes de seguir.

> Daqui para frente, se `py` não funcionar aí, troque `py` por `python` em
> todos os comandos.

---

## Passo 4 — Instalar as duas ferramentas de publicação

```powershell
py -m pip install --upgrade build twine
```

Demora um minuto. **Esperado no final:** `Successfully installed build… twine…`
(ou `Requirement already satisfied`, que também está ok).

---

## Passo 5 — Criar a conta e o token no PyPI

Isso é no navegador, uma vez só.

1. Entre em **pypi.org** → **Register** e crie a conta (ou faça login, se já
   tiver — você já publicou o `modelguard-ml`).
2. Confirme o e-mail.
3. Ative a verificação em duas etapas, se ele pedir. **O PyPI exige.**
4. Vá em **Account settings** → role até **API tokens** → **Add API token**.
   - *Token name:* `shapeaudit`
   - *Scope:* **Entire account** (é a primeira vez; depois dá para restringir)
5. Clique em **Create token**. Aparece um texto longo começando com
   `pypi-AgEIcHl…`

**Copie esse texto agora e guarde num bloco de notas.** Ele aparece **uma única
vez**. Se fechar a página, não dá para ver de novo — só criar outro.

---

## Passo 6 — Colocar o endereço do seu GitHub

Se você ainda não tem o repositório no GitHub, **pule este passo** e volte
depois: dá para publicar sem o link e corrigir numa versão seguinte.

Se já tem, abra o arquivo de configuração:

```powershell
notepad pyproject.toml
```

Procure a linha:

```
Homepage = "https://github.com/SEU_USUARIO/shapeaudit"
```

Troque `SEU_USUARIO` pelo seu nome de usuário do GitHub. Salve (Ctrl+S) e
feche o Bloco de Notas.

---

## Passo 7 — Gerar os arquivos que serão publicados

```powershell
py -m build
```

Demora de 30 segundos a 2 minutos. **Esperado no final:**

```
Successfully built shapeaudit-1.0.2.tar.gz and shapeaudit-1.0.2-py3-none-any.whl
```

Confira que os dois arquivos existem:

```powershell
dir dist
```

**Esperado:** dois arquivos, um `.whl` e um `.tar.gz`.

---

## Passo 8 — Conferir antes de subir

```powershell
py -m twine check dist/*
```

**Esperado:** duas linhas terminando em **PASSED**.

Esse comando também valida se o texto da página (o README) vai aparecer certo
no site. Se der `FAILED`, pare e me mande a mensagem.

---

## Passo 9 — Ensaio no TestPyPI (recomendado)

O TestPyPI é uma cópia do PyPI para treinar. Nada do que você subir lá vai
para o site de verdade.

Primeiro crie uma conta e um token **separados** em **test.pypi.org** (é outro
cadastro, mesmo processo do passo 5). Depois:

```powershell
py -m twine upload --repository testpypi dist/*
```

Ele vai pedir:

- **Enter your username:** digite exatamente `__token__`
  (duas barras baixas, a palavra token, duas barras baixas)
- **Enter your password:** cole o token do TestPyPI

> **Importante:** ao colar a senha, **a tela não mostra nada**. Não é travamento
> — é assim mesmo. Cole com **Ctrl+V** ou clique com o botão direito, e aperte
> Enter.

**Esperado:** uma barra de progresso e no final um link
`https://test.pypi.org/project/shapeaudit/1.0.2/`

**Abra esse link e leia a página como se fosse a primeira vez.** É o momento de
pegar texto confuso, antes de usar a versão de verdade.

---

## Passo 10 — Publicar no PyPI de verdade

```powershell
py -m twine upload dist/*
```

Mesma coisa: usuário `__token__`, senha é o token do **pypi.org** (o do passo 5,
não o do TestPyPI).

**Esperado:** `https://pypi.org/project/shapeaudit/1.0.2/`

Pronto. Está publicado.

---

## Passo 11 — Conferir que qualquer pessoa consegue instalar

Feche o PowerShell, abra outro (em qualquer pasta) e rode:

```powershell
py -m pip install shapeaudit
py -c "import shapeaudit; print(shapeaudit.__version__)"
```

**Esperado:** `1.0.2`

Se aparecer `1.0.2`, acabou: o pacote está no ar e funcionando.

---

## Se algo der errado

| mensagem | o que é | o que fazer |
|---|---|---|
| `File already exists` | essa versão já foi publicada | mude a versão no `pyproject.toml` para `1.0.3` e refaça os passos 7 a 10 |
| `403 Forbidden` / `Invalid credentials` | usuário ou token errado | o usuário é `__token__`, e o token do TestPyPI **não** serve no PyPI |
| `The name is too similar to an existing project` | o PyPI achou o nome parecido demais com outro | me avise que escolhemos outro nome entre os que estão livres |
| `py não é reconhecido` | Python fora do PATH | tente `python`; se também falhar, me avise |
| a senha não aparece ao colar | comportamento normal | cole e aperte Enter assim mesmo |

**Regra geral:** se a mensagem não estiver nesta tabela, copie o texto do erro
inteiro e me mande. Não tente adivinhar — publicar errado queima o número da
versão, e aí só dá para subir outra.

---

## Depois de publicar

Duas coisas valem a pena, sem pressa:

1. **Subir o código no GitHub** e ligar o GitHub Pages apontando para a pasta
   `demo/`. Aí a demonstração vira um link que você manda para qualquer
   pessoa. O passo a passo está no `PUBLICAR.md`, dentro da pasta.
2. **Guardar o token** num lugar seguro. Você vai precisar dele na próxima
   versão.
