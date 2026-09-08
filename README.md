# A Loja que Permanece — Visitante

Aplicação pública da jornada Visitante, com navegação guiada e conteúdo
previamente homologado.

## Executar localmente

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Publicar

Use `app.py` como arquivo principal da aplicação. Todos os arquivos desta pasta
devem permanecer juntos no repositório de publicação.

O pacote não contém o PDF do livro, banco de usuários, índice de busca, Ollama,
nem as modalidades Leitor e Premium.
