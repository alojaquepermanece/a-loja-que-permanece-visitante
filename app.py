import hashlib
import hmac
import html
import json
import math
import pickle
import re
import unicodedata
import urllib.request
import urllib.error
import urllib.parse
import time
from datetime import date, datetime, time as horario, timedelta
from pathlib import Path

import streamlit as st

from nucleo_atendimento import criar_nucleo


BASE = Path(__file__).resolve().parent
INDICE = BASE / "dados" / "indice_livro.pkl"
CAPA_APP = BASE / "capa_app.jpg"

MODELO_CHAT = "llama3.2"

QUANTIDADE_TRECHOS = 4

PESO_SEMANTICO = 0.75
PESO_LEXICAL = 0.25

PAGINAS_IGNORADAS = {374}

STOPWORDS = {
    "a", "ao", "aos", "as", "com", "como", "da", "das",
    "de", "do", "dos", "e", "em", "entre", "na", "nas",
    "no", "nos", "o", "os", "para", "por", "que", "se",
    "um", "uma", "uns", "umas",
    "loja", "lojas",
    "irmao", "irmaos",
    "maconaria",
    "maconico", "maconica",
    "maconicos", "maconicas",
    "obra",
}


def normalizar(texto):
    texto = texto.lower()

    texto = unicodedata.normalize("NFD", texto)

    texto = "".join(
        caractere
        for caractere in texto
        if unicodedata.category(caractere) != "Mn"
    )

    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()


def tokens_relevantes(texto):
    palavras = normalizar(texto).split()

    return [
        palavra
        for palavra in palavras
        if palavra not in STOPWORDS
        and len(palavra) >= 3
    ]


def gerar_embedding(texto, modelo):
    payload = {
        "model": modelo,
        "input": texto,
    }

    requisicao = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        requisicao,
        timeout=180,
    ) as resposta:

        resultado = json.loads(
            resposta.read().decode("utf-8")
        )

    return resultado["embeddings"][0]


def similaridade_cosseno(a, b):
    produto = sum(
        x * y
        for x, y in zip(a, b)
    )

    norma_a = math.sqrt(
        sum(x * x for x in a)
    )

    norma_b = math.sqrt(
        sum(y * y for y in b)
    )

    if norma_a == 0 or norma_b == 0:
        return 0.0

    return produto / (norma_a * norma_b)


def pontuacao_lexical(consulta, texto, capitulo):
    tokens = tokens_relevantes(consulta)

    if not tokens:
        return 0.0

    texto_normalizado = normalizar(texto)
    capitulo_normalizado = normalizar(capitulo)

    palavras_texto = set(texto_normalizado.split())
    palavras_capitulo = set(capitulo_normalizado.split())

    encontrados_texto = sum(
        1 for token in tokens
        if token in palavras_texto
    )

    encontrados_capitulo = sum(
        1 for token in tokens
        if token in palavras_capitulo
    )

    cobertura_texto = encontrados_texto / len(tokens)
    cobertura_capitulo = encontrados_capitulo / len(tokens)

    frase = normalizar(consulta)

    bonus_frase = (
        1.0
        if frase and frase in texto_normalizado
        else 0.0
    )

    return min(
        0.70 * cobertura_texto
        + 0.20 * bonus_frase
        + 0.10 * cobertura_capitulo,
        1.0,
    )


def classificar_cobertura(resultados):
    if not resultados:
        return "FRACA"

    melhor = resultados[0]

    semantica = melhor["semantica"]
    lexical = melhor["lexical"]

    if semantica >= 0.48 and lexical >= 0.45:
        return "DIRETA"

    if (
        semantica >= 0.52
        or (
            semantica >= 0.46
            and lexical >= 0.20
        )
    ):
        return "RELACIONADA"

    return "FRACA"


@st.cache_resource
def carregar_indice():
    with open(INDICE, "rb") as arquivo:
        dados = pickle.load(arquivo)

    trechos = [
        trecho
        for trecho in dados["trechos"]
        if trecho["pagina_pdf"] not in PAGINAS_IGNORADAS
    ]

    return dados["modelo_embedding"], trechos

def eh_referencia_contextual(texto):
    texto_normalizado = normalizar(texto)

    expressoes_contextuais = [
        "o tema",
        "esse tema",
        "este tema",
        "sobre o tema",
        "isso",
        "sobre isso",
        "essa questao",
        "esta questao",
        "essa situacao",
        "esta situacao",
        "esse assunto",
        "este assunto",
        "sobre esse assunto",
        "sobre essa questao",
        "nesse caso",
        "neste caso",
        "quanto a isso",
        "e quanto a isso",
        "onde o livro fala disso",
        "como o livro trata disso",
        "seus limites",
        "os seus limites",
        "suas atribuicoes",
        "as suas atribuicoes",
        "seu papel",
        "o seu papel",
        "sua funcao",
        "a sua funcao",
        "seu sentido",
        "o seu sentido",
        "sua natureza",
        "a sua natureza",
        "sua contribuicao",
        "a sua contribuicao",
    ]

    return any(
        expressao in texto_normalizado
        for expressao in expressoes_contextuais
    )

def eh_aceite_acao_pendente(texto):
    texto_normalizado = normalizar(texto).strip()

    expressoes_aceite = [
        "sim",
        "sim por favor",
        "quero",
        "quero sim",
        "pode ser",
        "pode fazer",
        "faca isso",
        "faca",
        "ajude me nisso",
        "me ajude nisso",
        "pode me ajudar nisso",
        "pode me ajudar",
        "prossiga",
        "vamos",
    ]

    return texto_normalizado in expressoes_aceite

def limpar_consulta_busca(consulta):
    texto = consulta.strip()

    padroes = [
        r"^\s*o\s+livro\s+fala\s+sobre\s+",
        r"^\s*o\s+livro\s+fala\s+de\s+",
        r"^\s*o\s+livro\s+trata\s+de\s+",
        r"^\s*o\s+livro\s+trata\s+sobre\s+",
        r"^\s*o\s+livro\s+aborda\s+",
        r"^\s*o\s+livro\s+ensina\s+",
        r"^\s*o\s+livro\s+explica\s+",
        r"^\s*a\s+obra\s+fala\s+sobre\s+",
        r"^\s*a\s+obra\s+trata\s+de\s+",
        r"^\s*a\s+obra\s+aborda\s+",
        r"^\s*a\s+loja\s+que\s+permanece\s+fala\s+sobre\s+",
        r"^\s*a\s+loja\s+que\s+permanece\s+trata\s+de\s+",
        r"^\s*a\s+loja\s+que\s+permanece\s+aborda\s+",
    ]

    for padrao in padroes:
        texto_novo = re.sub(
            padrao,
            "",
            texto,
            flags=re.IGNORECASE,
        )

        if texto_novo != texto:
            texto = texto_novo
            break

    texto = texto.strip(" ?!.,;:")
    
    texto = re.sub(
        r"^\s*(o|a|os|as)\s+",
        "",
        texto,
        flags=re.IGNORECASE,
    )
    # Se a limpeza produzir algo vazio ou muito curto,
    # mantém a pergunta original.
    if len(texto) < 3:
        return consulta

    return texto
def buscar_no_livro(consulta):
    modelo_embedding, trechos = carregar_indice()

    consulta_busca = limpar_consulta_busca(
        consulta
    )
    consulta_busca = re.sub(
        r"^\s*"
        r"(monte|crie|faça|elabore|prepare|organize)\s+"
        r"(um|uma)?\s*"
        r"(plano|cronograma|checklist|roteiro|estrutura)\s+"
        r"(de|da|do)?\s*",
        "",
        consulta_busca,
        flags=re.IGNORECASE,
    ).strip()
    consulta_busca = re.sub(
        r"\s+com\s+"
        r"(tarefas|etapas|ações|acoes|responsáveis|responsaveis|"
        r"prazos|periodicidades)"
        r"(?:\s*(?:,|e)\s*"
        r"(tarefas|etapas|ações|acoes|responsáveis|responsaveis|"
        r"prazos|periodicidades))*"
        r"\s*$",
        "",
        consulta_busca,
        flags=re.IGNORECASE,
    ).strip()
    consulta_busca = re.sub(
        r"[.!?]\s*"
        r"(monte|crie|faça|elabore|prepare|organize)\b.*$",
        "",
        consulta_busca,
        flags=re.IGNORECASE,
    ).strip()
    if re.search(
        r"\b(trocar|troca|mudança|transição)\s+de\s+gestão\b",
        consulta_busca,
        flags=re.IGNORECASE,
    ):
        consulta_busca = "transição de gestão"
    embedding_consulta = gerar_embedding(
        consulta_busca,
        modelo_embedding,
    )

    resultados = []

    for trecho in trechos:
        semantica = similaridade_cosseno(
            embedding_consulta,
            trecho["embedding"],
        )

        lexical = pontuacao_lexical(
            consulta_busca,
            trecho["texto"],
            trecho["capitulo"],
        )

        combinada = (
            PESO_SEMANTICO * semantica
            + PESO_LEXICAL * lexical
        )

        resultados.append(
            {
                "pontuacao": combinada,
                "semantica": semantica,
                "lexical": lexical,
                "pagina_pdf": trecho["pagina_pdf"],
                "capitulo": trecho["capitulo"],
                "texto": trecho["texto"],
            }
        )

    resultados.sort(
        key=lambda item: item["pontuacao"],
        reverse=True,
    )

    melhores = resultados[:QUANTIDADE_TRECHOS]
    cobertura = classificar_cobertura(resultados)

    return cobertura, melhores, consulta_busca

def carregar_mapa_global_obra():
    with open(
        "mapa_global_obra.json",
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)
def carregar_dados_comerciais():
    with open(
        "dados_comerciais.json",
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)
def localizar_dados_operacionais(consulta):
    if not consulta:
        return []

    consulta_normalizada = normalizar(
        consulta
    )

    categorias_encontradas = []

    marcadores = {
        "localidades_com_vendas": [
            "quantos livros ja foram vendidos",
            "quantos livros foram vendidos",
            "onde o livro ja foi vendido",
            "onde o livro foi vendido",
            "em quais estados o livro foi vendido",
            "estados onde o livro foi vendido",
            "estados com vendas",
            "onde ja houve vendas",
            "onde houve vendas",
            "alcance das vendas",
        ],
    }

    for categoria, termos in marcadores.items():
        for termo in termos:
            if normalizar(termo) in consulta_normalizada:
                categorias_encontradas.append(
                    categoria
                )
                break

    return categorias_encontradas
def montar_resposta_dados_operacionais(categorias):
    dados = carregar_dados_operacionais()

    vendas = dados.get(
        "vendas",
        {},
    )

    linhas = []

    if "localidades_com_vendas" in categorias:
        localidades = vendas.get(
            "estados_com_vendas",
            [],
        )

        if localidades:
            linhas.append(
                "**O livro já foi vendido nas seguintes localidades:**"
            )

            for localidade in localidades:
                linhas.append(
                    f"- {localidade}"
                )

    return "\n".join(linhas).strip()
def carregar_dados_operacionais():
    with open(
        "dados_operacionais.json",
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)
def montar_resposta_dados_comerciais(
    categorias,
):
    if not categorias:
        return ""

    dados = carregar_dados_comerciais()
    linhas = []

    if "preco" in categorias:
        preco = dados.get(
            "preco",
            {},
        )

        valor = preco.get(
            "valor"
        )

        moeda = preco.get(
            "moeda",
            "BRL",
        )

        observacao = preco.get(
            "observacao",
            "",
        )

        if valor is not None:
            if moeda == "BRL":
                valor_formatado = (
                    f"R$ {valor:,.2f}"
                    .replace(",", "X")
                    .replace(".", ",")
                    .replace("X", ".")
                )
            else:
                valor_formatado = (
                    f"{valor:.2f} {moeda}"
                )

            linhas.append(
                f"**Preço do livro:** {valor_formatado}"
            )

            if observacao:
                linhas.append(
                    observacao
                )
        else:
            linhas.append(
                "O preço não está disponível "
                "na base comercial atual."
            )

    if "disponibilidade" in categorias:
        disponibilidade = dados.get(
            "disponibilidade"
        )

        if linhas:
            linhas.append("")

        if disponibilidade:
            linhas.append(
                f"**Disponibilidade:** {disponibilidade}"
            )
        else:
            linhas.append(
                "A disponibilidade não está informada "
                "na base comercial atual."
            )

    if "formas_de_pagamento" in categorias:
        formas = dados.get(
            "formas_de_pagamento",
            [],
        )

        if linhas:
            linhas.append("")

        if formas:
            linhas.append(
                "**Formas de pagamento:**"
            )

            for forma in formas:
                linhas.append(
                    f"- {forma}"
                )
        else:
            linhas.append(
                "As formas de pagamento não estão "
                "informadas na base comercial atual."
            )

    if "frete" in categorias:
        frete = dados.get(
            "frete"
        )

        if linhas:
            linhas.append("")

        if isinstance(frete, dict):
            economico = frete.get(
                "economico",
                {},
            )

            linhas.append(
                "**Frete:**"
            )

            valor_1 = economico.get(
                "1_exemplar"
            )

            valor_2 = economico.get(
                "2_exemplares"
            )

            if valor_1 is not None:
                linhas.append(
                    f"- 1 exemplar: R$ {valor_1:.2f}".replace(
                        ".",
                        ",",
                    )
                )

            if valor_2 is not None:
                linhas.append(
                    f"- 2 exemplares: R$ {valor_2:.2f}".replace(
                        ".",
                        ",",
                    )
                )

            demais = frete.get(
                "demais_situacoes",
                "",
            )

            if demais:
                linhas.append(
                    f"- {demais}"
                )
        else:
            linhas.append(
                "As condições de frete não estão "
                "informadas na base comercial atual."
            )

    if "retirada" in categorias:
        retirada = dados.get(
            "retirada"
        )

        if linhas:
            linhas.append("")

        if retirada:
            linhas.append(
                f"**Retirada:** {retirada}"
            )
        else:
            linhas.append(
                "As condições de retirada não estão "
                "informadas na base comercial atual."
            )

    if "descontos_por_quantidade" in categorias:
        descontos = dados.get(
            "descontos_por_quantidade"
        )

        if linhas:
            linhas.append("")

        if descontos:
            linhas.append(
                f"**Compras em quantidade:** {descontos}"
            )
        else:
            linhas.append(
                "Não há informação cadastrada sobre "
                "condições para compras em quantidade."
            )

    if "canais_de_aquisicao" in categorias:
        canais = dados.get(
            "canais_de_aquisicao",
            {},
        )

        if linhas:
            linhas.append("")

        if canais:
            linhas.append(
                "**Onde adquirir:**"
            )

            site = canais.get(
                "site",
                "",
            )

            venda_presencial = canais.get(
                "venda_presencial",
                "",
            )

            if site:
                linhas.append(
                    f"- Site: {site}"
                )

            if venda_presencial:
                linhas.append(
                    f"- {venda_presencial}"
                )
        else:
            linhas.append(
                "Os canais de aquisição não estão "
                "informados na base comercial atual."
            )

    if "canais_de_contato" in categorias:
        contatos = dados.get(
            "canais_de_contato",
            {},
        )

        if linhas:
            linhas.append("")

        if contatos:
            linhas.append(
                "**Canais de contato:**"
            )

            email = contatos.get(
                "email",
                "",
            )

            instagram = contatos.get(
                "instagram",
                "",
            )

            whatsapp = contatos.get(
                "whatsapp",
                "",
            )

            if email:
                linhas.append(
                    f"- E-mail: {email}"
                )

            if instagram:
                linhas.append(
                    f"- Instagram: {instagram}"
                )

            if whatsapp:
                linhas.append(
                    f"- WhatsApp: {whatsapp}"
                )
        else:
            linhas.append(
                "Os canais de contato não estão "
                "informados na base comercial atual."
            )

    return "\n".join(linhas).strip()
def localizar_dados_comerciais(consulta):
    if not consulta:
        return []

    consulta_normalizada = normalizar(
        consulta
    )

    pergunta_sobre_valor_compra = any(
        marcador in consulta_normalizada
        for marcador in [
            "por que eu deveria comprar",
            "por que deveria comprar",
            "por que comprar",
            "vale a pena comprar",
            "motivos para comprar",
            "motivos para adquirir",
        ]
    )

    categorias_encontradas = []

    marcadores = {
        "preco": [
            "preco do livro",
            "preco da obra",
            "qual o preco",
            "qual e o preco",
            "quanto custa",
            "valor do livro",
            "valor da obra",
            "quanto e o livro",
        ],
        "disponibilidade": [
            "esta disponivel",
            "esta em estoque",
            "tem em estoque",
            "pronta entrega",
            "disponibilidade do livro",
            "disponibilidade da obra",
        ],
        "formas_de_pagamento": [
            "formas de pagamento",
            "forma de pagamento",
            "como posso pagar",
            "como pagar",
            "aceita pix",
            "aceita cartao",
            "aceita boleto",
            "pagamento",
        ],
        "frete": [
            "frete",
            "valor do frete",
            "quanto custa o envio",
            "quanto custa para enviar",
            "envio do livro",
            "envio da obra",
        ],
        "retirada": [
            "retirada",
            "retirar pessoalmente",
            "retirar em maos",
            "buscar pessoalmente",
        ],
        "descontos_por_quantidade": [
            "desconto",
            "desconto por quantidade",
            "varios exemplares",
            "mais de um exemplar",
            "compra em quantidade",
            "comprar varios",
        ],
        "canais_de_aquisicao": [
            "onde comprar",
            "como comprar",
            "como adquirir",
            "onde adquirir",
            "quero comprar",
            "quero adquirir",
            "comprar o livro",
            "comprar a obra",
        ],
        "canais_de_contato": [
            "como entrar em contato",
            "como entro em contato",
            "contato",
            "email",
            "e-mail",
            "instagram",
            "whatsapp",
            "falar com o projeto",
            "falar com voces",
        ],
    }

    for categoria, termos in marcadores.items():
        if (
            categoria == "canais_de_aquisicao"
            and pergunta_sobre_valor_compra
        ):
            continue
        for termo in termos:
            if normalizar(termo) in consulta_normalizada:
                categorias_encontradas.append(
                    categoria
                )
                break

    return categorias_encontradas
def localizar_secoes_globais_obra(consulta):
    if not consulta:
        return []

    consulta_normalizada = normalizar(consulta)

    secoes_encontradas = []

    marcadores = {
        "identidade_obra": [
            "autoria",
            "quem escreveu",
            "quem e o autor do livro",
            "quem e o autor da obra",
            "qual e o autor do livro",
            "qual e o autor da obra",
            "autor do livro",
            "autor da obra",
            "titulo",
            "titulo completo",
            "titulo completo da obra",
            "titulo completo do livro",
            "titulo do livro",
            "titulo da obra",
            "subtitulo",
            "isbn",
            "edicao",
            "ano de publicacao",
            "quantas paginas",
            "numero de paginas",
            "quantidade de paginas",
            "paginas tem o livro",
            "paginas tem a obra",
            "publicacao",
            "foi publicado",
            "livro foi publicado",
            "obra foi publicada",
            "onde foi publicado",
            "quando foi publicado",
            "onde o livro foi publicado",
            "quando o livro foi publicado",
            "dados da obra",
            "dados do livro",
            "dados bibliograficos",
            "dados bibliograficos da obra",
            "dados bibliograficos do livro",
            "copyright",
            "direitos autorais",
            "direitos autorais do livro",
            "direitos autorais da obra",
        ],
        "autor_perfil": [
            "quem e mauro arantes",
            "quem e o mauro arantes",
            "quem e esse autor",
            "quem e o autor",
            "sobre o autor",
            "perfil do autor",
            "biografia do autor",
            "trajetoria do autor",
            "trajetoria maconica do autor",
            "experiencia do autor",
            "experiencia profissional do autor",
            "profissao do autor",
            "qual e a profissao do autor",
            "qual a profissao do autor",
            "formacao do autor",
            "qual e a formacao do autor",
            "qual a formacao do autor",
            "mauro arantes e macom",
            "mauro arantes e mestre instalado",
            "autor e mestre instalado",
            "ele e mestre instalado",
            "cargos exercidos pelo autor",
            "cargos que o autor exerceu",
            "quais cargos o autor exerceu",
            "autor ja foi veneravel mestre",
            "ele ja foi veneravel mestre",
            "quantas vezes foi veneravel mestre",
            "experiencia para escrever este livro",
            "experiencia para escrever o livro",
             "por que o autor escreveu este livro",
            "relacao do autor com a obra",
            "em quais lojas",
            "em que lojas",
            "quais lojas o autor",
            "nome das lojas",
            "numero das lojas",
            "data de iniciacao",
            "quando foi iniciado",
            "quando se iniciou",
            "data de elevacao",
            "quando foi elevado",
            "data de exaltacao",
            "quando foi exaltado",
            "data de instalacao",
            "quando foi instalado",
        ],
        "estrutura_obra": [
            "estrutura da obra",
            "estrutura do livro",
            "como a obra esta estruturada",
            "como o livro esta estruturado",
            "como a obra esta organizada",
            "como o livro esta organizado",
            "partes",
            "capitulos",
            "cargos e funcoes abordados",
            "cargos e funcoes sao abordados",
            "quais cargos e funcoes",
            "quais cargos sao abordados",
            "quais cargos aparecem",
            "cargos abordados no livro",
            "cargos abordados na obra",
            "funcoes abordadas no livro",
            "funcoes abordadas na obra",
            "que funcoes maconicas",
            "quais funcoes maconicas",
            "sumario",
            "interludio",
            "introducao do livro",
            "introducao da obra",
            "o que ha na introducao",
            "o que existe na introducao",
            "sobre o que trata a introducao",
            "o que a introducao apresenta",
            "conteudo da introducao",
            "estrutura da introducao",
            "encerramento do livro",
            "encerramento da obra",
            "o que ha no encerramento",
            "o que existe no encerramento",
            "estrutura do encerramento",
            "secoes finais do livro",
            "secoes finais da obra",
        ],
        "escopo_e_proposito": [
            "o livro ensina procedimentos ritualisticos",
            "a obra ensina procedimentos ritualisticos",
            "o livro ensina procedimentos liturgicos",
            "a obra ensina procedimentos liturgicos",
            "o livro ensina rituais",
            "a obra ensina rituais",
            "o livro descreve rituais",
            "a obra descreve rituais",
            "procedimentos ritualisticos no livro",
            "procedimentos liturgicos no livro",
            "conteudo ritualistico da obra",
            "conteudo ritualistico do livro",
            "conteudo liturgico da obra",
            "conteudo liturgico do livro",
            "questoes ritualisticas",
            "sigilo ritual",
            "conteudo reservado",
            "revela sinais",
            "revela palavras",
            "preciso ter experiencia administrativa",
            "precisa ter experiencia administrativa",
            "e necessario ter experiencia administrativa",
            "preciso ter experiencia previa",
            "precisa ter experiencia previa",
            "experiencia previa para ler",
            "experiencia previa para aproveitar",
            "preciso ter formacao academica",
            "precisa ter formacao academica",
            "formacao academica para ler",
            "formacao previa",
            "pre requisitos do leitor",
            "pre requisitos para o leitor",
            "pre requisitos para ler",
            "prerequisitos do leitor",
            "prerequisitos para ler",
            "por que este livro foi escrito",
            "por que o livro foi escrito",
            "por que esta obra foi escrita",
            "por que a obra foi escrita",
            "o que motivou a criacao deste livro",
            "o que motivou a criacao do livro",
            "o que motivou a criacao desta obra",
            "o que motivou a criacao da obra",
            "origem e motivacao",
            "origem da obra",
            "origem do livro",
            "motivacao da obra",
            "motivacao do livro",
            "como surgiu este livro",
            "como surgiu o livro",
            "proposito da obra",
            "proposito do livro",
            "objetivo da obra",
            "objetivo do livro",
            "publico alvo",
            "publico da obra",
            "publico do livro",
            "para quem e o livro",
            "para quem e a obra",
            "para quem se destina",
            "para quem o livro se destina",
            "para quem a obra se destina",
            "a quem o livro se destina",
            "a quem a obra se destina",
            "escopo da obra",
            "escopo do livro",
            "este livro pretende esgotar",
            "o livro pretende esgotar",
            "esta obra pretende esgotar",
            "a obra pretende esgotar",
            "pretende esgotar o tema",
            "esgotar o tema da governanca maconica",
            "limite de pretensao",
            "limites da proposta",
            "limites da obra",
            "limites do livro",
            "obra pretende ser definitiva",
            "livro pretende ser definitivo",
            "como utilizar o livro",
            "como utilizar este livro",
            "como utilizar esta obra",
            "como devo utilizar o livro",
            "como devo utilizar este livro",
            "como devo utilizar a obra",
            "como devo utilizar esta obra",
            "como usar o livro",
            "como usar este livro",
            "como usar a obra",
            "como usar esta obra",
            "como ler este livro",
            "como ler o livro",
            "melhor forma de ler o livro",
            "melhor forma de utilizar o livro",
            "preciso ler o livro inteiro",
            "preciso ler este livro inteiro",
            "posso consultar apenas um capitulo",
            "posso consultar por cargo",
            "posso consultar por tema",
            "delimitacao ritualistica",
            "o que encontrarei neste livro",
            "por que eu deveria comprar o livro",
            "por que deveria comprar o livro",
            "o que encontrarei no livro",
            "o que vou encontrar neste livro",
            "o que vou encontrar no livro",
            "o que encontro neste livro",
            "o que encontro no livro",
            "encontrar neste livro",
            "encontrar no livro",
            "encontrar nesta obra",
            "encontrar na obra",
            "conteudo do livro",
            "ajudar minha loja",
            "ajudar a minha loja",
            "ajudar uma loja",
            "ajudar a loja",
            "ajudar minha oficina",
            "ajudar uma oficina",
            "como esse livro pode ajudar",
            "como este livro pode ajudar",
            "como o livro pode ajudar",
            "como essa obra pode ajudar",
            "como esta obra pode ajudar",
            "beneficios para a loja",
            "beneficios para uma loja",
            "beneficios para a oficina",
            "loja com poucos membros",
            "loja com poucos irmaos",
            "oficina com poucos membros",
            "oficina com poucos irmaos",
            "loja com quadro reduzido",
            "oficina com quadro reduzido",
            "aplicado em uma loja com poucos membros",
            "aplicado em uma oficina com poucos membros",
            "funciona em loja pequena",
            "funciona em uma loja pequena",
            "funciona em oficina pequena",
            "funciona em uma oficina pequena",
            "adaptacao a realidade das lojas",
            "adaptar a realidade da loja",
            "adaptar a realidade da oficina",
            "acumulo de funcoes",
            "acumular funcoes",
            "conteudo da obra",
            "o que este livro apresenta",
            "o que o livro apresenta",
            "o que esta obra apresenta",
            "o que a obra apresenta",
            "sobre o que e o livro",
            "sobre o que e esta obra",
            "o que o livro oferece",
            "o que a obra oferece",
            "por que eu deveria comprar este livro",
            "por que deveria comprar este livro",
            "por que comprar este livro",
            "por que comprar o livro",
            "por que comprar esta obra",
            "vale a pena comprar este livro",
            "vale a pena comprar o livro",
            "motivos para comprar o livro",
            "motivos para adquirir o livro",
        ],
    }

    for secao, termos in marcadores.items():
        for termo in termos:
            if normalizar(termo) in consulta_normalizada:
                secoes_encontradas.append(secao)
                break

    return secoes_encontradas

def montar_resposta_secoes_globais_obra(
    secoes,
    pergunta,
):
    if not secoes:
        return ""

    mapa_global = carregar_mapa_global_obra()
    linhas = []

    pergunta_normalizada = normalizar(
        pergunta
    )

    if "identidade_obra" in secoes:
        identidade = mapa_global.get(
            "identidade_obra",
            {},
        )

        autor = identidade.get(
            "autor",
            "",
        )

        titulo = identidade.get(
            "titulo",
            "",
        )

        subtitulo = identidade.get(
            "subtitulo",
            "",
        )

        titulo_completo = identidade.get(
            "titulo_completo",
            "",
        )

        local_publicacao = identidade.get(
            "local_publicacao",
            "",
        )

        ano_publicacao = identidade.get(
            "ano_publicacao",
            "",
        )

        edicao = identidade.get(
            "edicao",
            "",
        )

        isbn = identidade.get(
            "isbn",
            "",
        )

        copyright_obra = identidade.get(
            "copyright",
            "",
        )
        numero_de_paginas = identidade.get(
            "numero_de_paginas",
            "",
        )
        pergunta_sobre_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quem escreveu",
                "quem e o autor",
                "qual e o autor",
                "autor do livro",
                "autor da obra",
                "autoria",
            ]
        )

        pergunta_sobre_subtitulo = any(
            marcador in pergunta_normalizada
            for marcador in [
                "qual e o subtitulo",
                "qual o subtitulo",
                "subtitulo do livro",
                "subtitulo da obra",
            ]
        )

        pergunta_sobre_titulo_completo = any(
            marcador in pergunta_normalizada
            for marcador in [
                "qual e o titulo completo",
                "qual o titulo completo",
                "titulo completo do livro",
                "titulo completo da obra",
                "titulo completo",
            ]
        )

        pergunta_sobre_titulo = (
            not pergunta_sobre_subtitulo
            and not pergunta_sobre_titulo_completo
            and any(
                marcador in pergunta_normalizada
                for marcador in [
                    "qual e o titulo",
                    "qual o titulo",
                    "titulo do livro",
                    "titulo da obra",
                ]
            )
        )

        pergunta_sobre_publicacao = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quando foi publicado",
                "onde foi publicado",
                "quando o livro foi publicado",
                "onde o livro foi publicado",
                "livro foi publicado",
                "obra foi publicada",
                "foi publicado",
                "ano de publicacao",
                "local de publicacao",
                "dados de publicacao",
            ]
        )

        pergunta_sobre_edicao = any(
            marcador in pergunta_normalizada
            for marcador in [
                "qual e a edicao",
                "edicao do livro",
                "edicao da obra",
            ]
        )

        pergunta_sobre_isbn = any(
            marcador in pergunta_normalizada
            for marcador in [
                "isbn",
            ]
        )

        pergunta_sobre_copyright = any(
            marcador in pergunta_normalizada
            for marcador in [
                "copyright",
                "direitos autorais",
                "direitos autorais do livro",
                "direitos autorais da obra",
            ]
        )

        pergunta_sobre_paginas = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quantas paginas",
                "numero de paginas",
                "quantidade de paginas",
                "paginas tem o livro",
                "paginas tem a obra",
            ]
        )

        identidade_especifica = any(
            [
                pergunta_sobre_autor,
                pergunta_sobre_titulo,
                pergunta_sobre_titulo_completo,
                pergunta_sobre_subtitulo,
                pergunta_sobre_publicacao,
                pergunta_sobre_edicao,
                pergunta_sobre_isbn,
                pergunta_sobre_copyright,
                pergunta_sobre_paginas,
            ]
        )

        if identidade_especifica:
            if pergunta_sobre_autor and autor:
                linhas.append(
                    f"O autor da obra é **{autor}**."
                )

            if pergunta_sobre_titulo and titulo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"O título da obra é **{titulo}**."
                )

            if (
                pergunta_sobre_titulo_completo
                and titulo_completo
            ):
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"O título completo é **{titulo_completo}**."
                )

            if pergunta_sobre_subtitulo and subtitulo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"O subtítulo da obra é **{subtitulo}**."
                )

            if (
                pergunta_sobre_publicacao
                and (local_publicacao or ano_publicacao)
            ):
                if linhas:
                    linhas.append("")

                if local_publicacao and ano_publicacao:
                    linhas.append(
                        f"A obra foi publicada em "
                        f"**{local_publicacao}, {ano_publicacao}**."
                    )
                elif local_publicacao:
                    linhas.append(
                        f"O local de publicação é "
                        f"**{local_publicacao}**."
                    )
                elif ano_publicacao:
                    linhas.append(
                        f"O ano de publicação é "
                        f"**{ano_publicacao}**."
                    )

            if pergunta_sobre_edicao and edicao:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"A edição indicada na obra é **{edicao}**."
                )

            if pergunta_sobre_isbn and isbn:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"O ISBN da obra é **{isbn}**."
                )

            if pergunta_sobre_copyright and copyright_obra:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Copyright:** {copyright_obra}"
                )
            if pergunta_sobre_paginas and numero_de_paginas:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"A obra possui **{numero_de_paginas} páginas**."
                )
        else:
            if autor:
                linhas.append(
                    f"O autor da obra é **{autor}**."
                )

            if titulo_completo:
                linhas.append("")
                linhas.append(
                    f"O título completo é **{titulo_completo}**."
                )

            if local_publicacao or ano_publicacao:
                linhas.append("")

                if local_publicacao and ano_publicacao:
                    linhas.append(
                        f"Publicação: **{local_publicacao}, "
                        f"{ano_publicacao}**."
                    )
                elif local_publicacao:
                    linhas.append(
                        f"Local de publicação: "
                        f"**{local_publicacao}**."
                    )
                elif ano_publicacao:
                    linhas.append(
                        f"Ano de publicação: "
                        f"**{ano_publicacao}**."
                    )

            if edicao:
                linhas.append("")
                linhas.append(
                    f"Edição: **{edicao}**."
                )

            if isbn:
                linhas.append("")
                linhas.append(
                    f"ISBN: **{isbn}**."
                )

            if copyright_obra:
                linhas.append("")
                linhas.append(
                    f"Copyright: **{copyright_obra}**."
                )

    if "autor_perfil" in secoes:
        perfil_autor = mapa_global.get(
            "autor_perfil",
            {},
        )

        nome_autor = perfil_autor.get(
            "nome",
            "",
        )

        profissao_autor = perfil_autor.get(
            "profissao",
            "",
        )

        formacao_autor = perfil_autor.get(
            "formacao",
            "",
        )

        apresentacao_autor = perfil_autor.get(
            "apresentacao_resumida",
            "",
        )

        trajetoria_autor = perfil_autor.get(
            "trajetoria_maconica_publica",
            {},
        )

        macom_desde = trajetoria_autor.get(
            "macom_desde",
            "",
        )

        mestre_instalado = trajetoria_autor.get(
            "mestre_instalado",
            False,
        )

        cargos_autor = trajetoria_autor.get(
            "cargos_exercidos",
            [],
        )

        veneravel_mestre = trajetoria_autor.get(
            "veneravel_mestre",
            {},
        )

        experiencia_governanca = perfil_autor.get(
            "experiencia_em_governanca",
            "",
        )

        atuacao_oficinas = perfil_autor.get(
            "atuacao_com_oficinas",
            "",
        )

        relacao_profissao_maconaria = perfil_autor.get(
            "relacao_entre_profissao_e_maconaria",
            "",
        )

        relacao_com_obra = perfil_autor.get(
            "relacao_com_a_obra",
            "",
        )

        pergunta_sobre_profissao_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "profissao do autor",
                "qual e a profissao do autor",
                "qual a profissao do autor",
                "experiencia profissional do autor",
            ]
        )

        pergunta_sobre_formacao_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "formacao do autor",
                "qual e a formacao do autor",
                "qual a formacao do autor",
            ]
        )

        pergunta_sobre_mestre_instalado = any(
            marcador in pergunta_normalizada
            for marcador in [
                "mauro arantes e mestre instalado",
                "autor e mestre instalado",
                "ele e mestre instalado",
            ]
        )

        pergunta_sobre_cargos_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "cargos exercidos pelo autor",
                "cargos que o autor exerceu",
                "quais cargos o autor exerceu",
            ]
        )

        pergunta_sobre_veneravel_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "autor ja foi veneravel mestre",
                "ele ja foi veneravel mestre",
                "quantas vezes foi veneravel mestre",
            ]
        )

        pergunta_sobre_trajetoria_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "trajetoria do autor",
                "trajetoria maconica do autor",
                "mauro arantes e macom",
            ]
        )

        pergunta_sobre_experiencia_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "experiencia do autor",
                "experiencia para escrever este livro",
                "experiencia para escrever o livro",
            ]
        )

        pergunta_sobre_relacao_obra = any(
            marcador in pergunta_normalizada
            for marcador in [
                "por que o autor escreveu este livro",
                "relacao do autor com a obra",
            ]
        )

        pergunta_sobre_dados_privados_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "em quais lojas",
                "em que lojas",
                "quais lojas",
                "qual loja",
                "nome das lojas",
                "numero das lojas",
                "data de iniciacao",
                "quando foi iniciado",
                "quando se iniciou",
                "data de elevacao",
                "quando foi elevado",
                "data de exaltacao",
                "quando foi exaltado",
                "data de instalacao",
                "quando foi instalado",
            ]
        )

        pergunta_sobre_perfil_autor = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quem e mauro arantes",
                "quem e o mauro arantes",
                "quem e esse autor",
                "quem e o autor",
                "sobre o autor",
                "perfil do autor",
                "biografia do autor",
            ]
        )

        perfil_especifico = any(
            [
                pergunta_sobre_profissao_autor,
                pergunta_sobre_formacao_autor,
                pergunta_sobre_mestre_instalado,
                pergunta_sobre_cargos_autor,
                pergunta_sobre_veneravel_autor,
                pergunta_sobre_trajetoria_autor,
                pergunta_sobre_experiencia_autor,
                pergunta_sobre_relacao_obra,
                pergunta_sobre_dados_privados_autor,
            ]
        )

        if pergunta_sobre_dados_privados_autor:
            if linhas:
                linhas.append("")

            linhas.append(
                "Essas informações não fazem parte do perfil público do autor. "
                "Posso informar que Mauro Arantes é maçom desde 2016, "
                "é Mestre Instalado e exerceu a função de Venerável Mestre "
                "por duas vezes, em Lojas distintas."
            )

        if pergunta_sobre_profissao_autor and profissao_autor:
            if linhas:
                linhas.append("")

            linhas.append(
                f"**Profissão:** {profissao_autor}."
            )

        if pergunta_sobre_formacao_autor and formacao_autor:
            if linhas:
                linhas.append("")

            linhas.append(
                f"**Formação:** {formacao_autor}."
            )

        if pergunta_sobre_mestre_instalado:
            if linhas:
                linhas.append("")

            if mestre_instalado:
                linhas.append(
                    f"Sim. **{nome_autor} é Mestre Instalado**."
                )

        if pergunta_sobre_cargos_autor and cargos_autor:
            if linhas:
                linhas.append("")

            linhas.append(
                "**Cargos maçônicos já exercidos pelo autor:**"
            )

            for cargo in cargos_autor:
                linhas.append(
                    f"- {cargo}"
                )

        if pergunta_sobre_veneravel_autor:
            if linhas:
                linhas.append("")

            observacao_veneravel = veneravel_mestre.get(
                "observacao",
                "",
            )

            if observacao_veneravel:
                linhas.append(
                    f"**Venerável Mestre:** {observacao_veneravel}"
                )

        if pergunta_sobre_trajetoria_autor:
            if linhas:
                linhas.append("")

            if macom_desde:
                linhas.append(
                    f"{nome_autor} é maçom desde **{macom_desde}**."
                )

            if mestre_instalado:
                linhas.append(
                    "É **Mestre Instalado**."
                )

            observacao_veneravel = veneravel_mestre.get(
                "observacao",
                "",
            )

            if observacao_veneravel:
                linhas.append(
                    observacao_veneravel
                )

            if cargos_autor:
                linhas.append("")
                linhas.append(
                    "**Cargos já exercidos:**"
                )

                for cargo in cargos_autor:
                    linhas.append(
                        f"- {cargo}"
                    )

        if pergunta_sobre_experiencia_autor:
            if linhas:
                linhas.append("")

            if experiencia_governanca:
                linhas.append(
                    experiencia_governanca
                )

            if atuacao_oficinas:
                linhas.append("")
                linhas.append(
                    atuacao_oficinas
                )

            if relacao_profissao_maconaria:
                linhas.append("")
                linhas.append(
                    relacao_profissao_maconaria
                )

        if pergunta_sobre_relacao_obra and relacao_com_obra:
            if linhas:
                linhas.append("")

            linhas.append(
                relacao_com_obra
            )

        if (
            pergunta_sobre_perfil_autor
            and not perfil_especifico
            and apresentacao_autor
        ):
            if linhas:
                linhas.append("")

            linhas.append(
                apresentacao_autor
            )

    if "estrutura_obra" in secoes:
        estrutura = mapa_global.get(
            "estrutura_obra",
            {},
        )

        numero_partes = estrutura.get(
            "numero_de_partes",
            "",
        )

        numero_capitulos = estrutura.get(
            "numero_de_capitulos",
            "",
        )

        introducao = estrutura.get(
            "introducao",
            {},
        )

        cargos_funcoes = estrutura.get(
            "cargos_e_funcoes_abordados",
            [],
        )

        encerramento = estrutura.get(
            "encerramento",
            [],
        )

        pergunta_sobre_encerramento = any(
            marcador in pergunta_normalizada
            for marcador in [
                "encerramento do livro",
                "encerramento da obra",
                "o que ha no encerramento",
                "o que existe no encerramento",
                "estrutura do encerramento",
                "secoes finais do livro",
                "secoes finais da obra",
            ]
        )

        pergunta_sobre_cargos_funcoes = any(
            marcador in pergunta_normalizada
            for marcador in [
                "cargos e funcoes abordados",
                "cargos e funcoes sao abordados",
                "quais cargos e funcoes",
                "quais cargos sao abordados",
                "quais cargos aparecem",
                "cargos abordados no livro",
                "cargos abordados na obra",
                "funcoes abordadas no livro",
                "funcoes abordadas na obra",
                "que funcoes maconicas",
                "quais funcoes maconicas",
            ]
        )

        pergunta_sobre_introducao = any(
            marcador in pergunta_normalizada
            for marcador in [
                "introducao do livro",
                "introducao da obra",
                "o que ha na introducao",
                "o que existe na introducao",
                "sobre o que trata a introducao",
                "o que a introducao apresenta",
                "conteudo da introducao",
                "estrutura da introducao",
            ]
        )

        pergunta_sobre_quantidade_capitulos = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quantos capitulos",
                "numero de capitulos",
                "quantidade de capitulos",
            ]
        )

        pergunta_sobre_quantidade_partes = any(
            marcador in pergunta_normalizada
            for marcador in [
                "quantas partes",
                "numero de partes",
                "quantidade de partes",
            ]
        )

        pergunta_sobre_porta_bandeira_estandarte = (
            (
                "porta bandeira" in pergunta_normalizada
                and "porta estandarte" in pergunta_normalizada
            )
            and any(
                marcador in pergunta_normalizada
                for marcador in [
                    "capitulos separados",
                    "capitulo separado",
                    "mesmo capitulo",
                    "tratados conjuntamente",
                    "tratados juntos",
                ]
            )
        )

        pergunta_sobre_interludio = (
            "interludio" in pergunta_normalizada
        )

        if pergunta_sobre_porta_bandeira_estandarte:
            observacoes = estrutura.get(
                "observacoes_estruturais",
                [],
            )

            observacao_porta = next(
                (
                    observacao
                    for observacao in observacoes
                    if (
                        "Porta-Bandeira" in observacao
                        and "Porta-Estandarte" in observacao
                    )
                ),
                "",
            )

            if observacao_porta:
                linhas.append(
                    f"**Não.** {observacao_porta}"
                )

            return "\n".join(linhas).strip()

        if pergunta_sobre_introducao:
            titulo_introducao = introducao.get(
                "titulo",
                "INTRODUÇÃO",
            )

            pagina_inicio_introducao = introducao.get(
                "pagina_inicio_impressa",
                "",
            )

            pagina_fim_introducao = introducao.get(
                "pagina_fim_impressa",
                "",
            )

            secoes_introducao = introducao.get(
                "secoes_abordadas",
                [],
            )

            if linhas:
                linhas.append("")

            linhas.append(
                f"**{titulo_introducao}**"
            )

            if (
                pagina_inicio_introducao
                and pagina_fim_introducao
            ):
                linhas.append("")
                linhas.append(
                    f"**Páginas impressas:** "
                    f"{pagina_inicio_introducao} a "
                    f"{pagina_fim_introducao}"
                )

            if secoes_introducao:
                linhas.append("")
                linhas.append(
                    "**Seções abordadas:**"
                )

                for item in secoes_introducao:
                    linhas.append(
                        f"- {item}"
                    )

            return "\n".join(linhas).strip()

        if pergunta_sobre_encerramento:
            if linhas:
                linhas.append("")

            linhas.append(
                "**Encerramento da obra:**"
            )

            for secao in encerramento:
                titulo_secao = secao.get(
                    "titulo",
                    "",
                )

                pagina_secao = secao.get(
                    "pagina_inicio_impressa",
                    "",
                )

                if titulo_secao and pagina_secao:
                    linhas.append(
                        f"- **{titulo_secao}** — "
                        f"página impressa {pagina_secao}"
                    )

                elif titulo_secao:
                    linhas.append(
                        f"- **{titulo_secao}**"
                    )

            return "\n".join(linhas).strip()

        if pergunta_sobre_cargos_funcoes:
            if linhas:
                linhas.append("")

            linhas.append(
                "**Cargos e funções abordados na obra:**"
            )

            for item in cargos_funcoes:
                linhas.append(
                    f"- {item}"
                )

            secao_especial = estrutura.get(
                "secao_especial",
                {},
            )

            titulo_secao_especial = secao_especial.get(
                "titulo",
                "",
            )

            natureza_secao_especial = secao_especial.get(
                "natureza",
                "",
            )

            if titulo_secao_especial:
                linhas.append("")
                linhas.append(
                    "Além desses cargos e funções, a obra contém "
                    f"a seção especial **{titulo_secao_especial}**."
                )

            if natureza_secao_especial:
                linhas.append(
                    f"**Natureza:** {natureza_secao_especial}"
                )

            return "\n".join(linhas).strip()

        if pergunta_sobre_interludio:
            secao_especial = estrutura.get(
                "secao_especial",
                {},
            )

            titulo_interludio = secao_especial.get(
                "titulo",
                "",
            )

            pagina_interludio = secao_especial.get(
                "pagina_inicio_impressa",
                "",
            )

            posicao_interludio = secao_especial.get(
                "posicao_editorial",
                "",
            )

            natureza_interludio = secao_especial.get(
                "natureza",
                "",
            )

            if linhas:
                linhas.append("")

            if titulo_interludio:
                linhas.append(
                    f"**{titulo_interludio}**"
                )

            if pagina_interludio:
                linhas.append("")
                linhas.append(
                    f"**Página inicial impressa:** {pagina_interludio}"
                )

            if posicao_interludio:
                linhas.append("")
                linhas.append(
                    f"**Posição editorial:** {posicao_interludio}"
                )

            if natureza_interludio:
                linhas.append("")
                linhas.append(
                    f"**Natureza:** {natureza_interludio}"
                )

            return "\n".join(linhas).strip()

        linhas.append("")
        linhas.append(
            "**Estrutura da obra:**"
        )
        if pergunta_sobre_quantidade_capitulos:
            linhas.append(
                f"A obra possui **{numero_capitulos} capítulos**, "
                f"distribuídos em **{numero_partes} partes**."
            )

            linhas.append("")
            linhas.append(
                "Além dos capítulos, há um Interlúdio dedicado ao "
                "Conselho de Mestres Instalados, que não constitui "
                "um décimo oitavo capítulo."
            )

            return "\n".join(linhas).strip()

        if pergunta_sobre_quantidade_partes:
            linhas.append(
                f"A obra está organizada em **{numero_partes} partes**, "
                f"que reúnem **{numero_capitulos} capítulos**."
            )

            return "\n".join(linhas).strip()
        if numero_partes and numero_capitulos:
            linhas.append(
                f"A obra está organizada em "
                f"**{numero_partes} partes e "
                f"{numero_capitulos} capítulos**."
            )

        partes = estrutura.get(
            "partes",
            [],
        )

        for parte in partes:
            numero = parte.get(
                "numero",
                "",
            )

            titulo = parte.get(
                "titulo",
                "",
            )

            if titulo:
                titulo_normalizado = normalizar(
                    titulo
                )

                if titulo_normalizado.startswith("parte "):
                    linhas.append(
                        f"- {titulo}"
                    )
                elif numero:
                    linhas.append(
                        f"- Parte {numero}: {titulo}"
                    )
                else:
                    linhas.append(
                        f"- {titulo}"
                    )


        observacoes = estrutura.get(
            "observacoes_estruturais",
            [],
        )

        if observacoes:
            linhas.append("")
            linhas.append(
                "**Observações estruturais importantes:**"
            )

            for observacao in observacoes:
                linhas.append(
                    f"- {observacao}"
                )

    if "escopo_e_proposito" in secoes:
        escopo = mapa_global.get(
            "escopo_e_proposito",
            {},
        )
        origem_motivacao = escopo.get(
            "origem_e_motivacao",
            "",
        )
        pre_requisitos = escopo.get(
            "pre_requisitos_do_leitor",
            "",
        )
        objetivo = escopo.get(
            "objetivo_principal",
            "",
        )

        proposito = escopo.get(
            "proposito_de_continuidade",
            "",
        )

        publico = escopo.get(
            "publico_alvo",
            [],
        )

        conteudo_metodo = escopo.get(
            "conteudo_e_metodo",
            "",
        )
        adaptacao_lojas = escopo.get(
            "adaptacao_a_realidade_das_lojas",
            "",
        )

        forma_de_uso = escopo.get(
            "forma_de_uso",
            [],
        )

        limite_de_pretensao = escopo.get(
            "limite_de_pretensao",
            "",
        )

        pergunta_sobre_origem = any(
            marcador in pergunta_normalizada
            for marcador in [
                "por que este livro foi escrito",
                "por que o livro foi escrito",
                "por que esta obra foi escrita",
                "por que a obra foi escrita",
                "o que motivou a criacao deste livro",
                "o que motivou a criacao do livro",
                "o que motivou a criacao desta obra",
                "o que motivou a criacao da obra",
                "origem e motivacao",
                "origem da obra",
                "origem do livro",
                "motivacao da obra",
                "motivacao do livro",
                "como surgiu este livro",
                "como surgiu o livro",
            ]
        )

        pergunta_sobre_pre_requisitos = any(
            marcador in pergunta_normalizada
            for marcador in [
                "preciso ter experiencia administrativa",
                "precisa ter experiencia administrativa",
                "e necessario ter experiencia administrativa",
                "preciso ter experiencia previa",
                "precisa ter experiencia previa",
                "experiencia previa para ler",
                "experiencia previa para aproveitar",
                "preciso ter formacao academica",
                "precisa ter formacao academica",
                "formacao academica para ler",
                "formacao previa",
                "pre requisitos do leitor",
                "pre requisitos para o leitor",
                "pre requisitos para ler",
                "prerequisitos do leitor",
                "prerequisitos para ler",
            ]
        )

        pergunta_sobre_forma_de_uso = any(
            marcador in pergunta_normalizada
            for marcador in [
                "como utilizar o livro",
                "como utilizar este livro",
                "como utilizar esta obra",
                "como devo utilizar o livro",
                "como devo utilizar este livro",
                "como devo utilizar a obra",
                "como devo utilizar esta obra",
                "como usar o livro",
                "como usar este livro",
                "como usar a obra",
                "como usar esta obra",
                "como ler este livro",
                "como ler o livro",
                "melhor forma de ler o livro",
                "melhor forma de utilizar o livro",
                "preciso ler o livro inteiro",
                "preciso ler este livro inteiro",
                "posso consultar apenas um capitulo",
                "posso consultar por cargo",
                "posso consultar por tema",
            ]
        )

        pergunta_sobre_limite_de_pretensao = any(
            marcador in pergunta_normalizada
            for marcador in [
                "este livro pretende esgotar",
                "o livro pretende esgotar",
                "esta obra pretende esgotar",
                "a obra pretende esgotar",
                "pretende esgotar o tema",
                "esgotar o tema da governanca maconica",
                "limite de pretensao",
                "limites da proposta",
                "limites da obra",
                "limites do livro",
                "obra pretende ser definitiva",
                "livro pretende ser definitivo",
            ]
        )

        pergunta_sobre_delimitacao_ritualistica = any(
            marcador in pergunta_normalizada
            for marcador in [
                "delimitacao ritualistica",
                "o livro ensina procedimentos ritualisticos",
                "a obra ensina procedimentos ritualisticos",
                "o livro ensina procedimentos liturgicos",
                "a obra ensina procedimentos liturgicos",
                "o livro ensina rituais",
                "a obra ensina rituais",
                "o livro descreve rituais",
                "a obra descreve rituais",
                "procedimentos ritualisticos no livro",
                "procedimentos liturgicos no livro",
                "conteudo ritualistico da obra",
                "conteudo ritualistico do livro",
                "conteudo liturgico da obra",
                "conteudo liturgico do livro",
                "questoes ritualisticas",
                "sigilo ritual",
                "conteudo reservado",
                "revela sinais",
                "revela palavras",
            ]
        )

        delimitacao_ritualistica = escopo.get(
            "delimitacao_ritualistica",
            {},
        )

        pergunta_sobre_objetivo = any(
            marcador in pergunta_normalizada
            for marcador in [
                "objetivo da obra",
                "objetivo do livro",
                "qual e o objetivo",
            ]
        )

        pergunta_sobre_proposito = any(
            marcador in pergunta_normalizada
            for marcador in [
                "proposito da obra",
                "proposito do livro",
                "qual e o proposito",
                "proposito de continuidade",
            ]
        )

        pergunta_sobre_publico = any(
            marcador in pergunta_normalizada
            for marcador in [
                "publico alvo",
                "publico da obra",
                "publico do livro",
                "para quem e o livro",
                "para quem e a obra",
                "para quem se destina",
                "para quem o livro se destina",
                "para quem a obra se destina",
                "a quem o livro se destina",
                "a quem a obra se destina",
            ]
        )

        pergunta_sobre_abordagem = any(
            marcador in pergunta_normalizada
            for marcador in [
                "abordagem da obra",
                "abordagem do livro",
                "como a obra aborda",
                "como o livro aborda",
                "conteudo e metodo",
            ]
        )

        pergunta_sobre_conteudo = any(
            marcador in pergunta_normalizada
            for marcador in [
                "conteudo do livro",
                "conteudo da obra",
                "o que encontrarei neste livro",
                "o que encontrarei no livro",
                "o que vou encontrar neste livro",
                "o que vou encontrar no livro",
                "encontrar neste livro",
                "encontrar no livro",
                "encontrar nesta obra",
                "encontrar na obra",
                "o que este livro apresenta",
                "o que o livro apresenta",
                "o que esta obra apresenta",
                "o que a obra apresenta",
                "sobre o que e o livro",
                "sobre o que e esta obra",
                "o que o livro oferece",
                "o que a obra oferece",
            ]
        )

        pergunta_sobre_ajuda_loja = any(
            marcador in pergunta_normalizada
            for marcador in [
                "ajudar minha loja",
                "ajudar a minha loja",
                "ajudar uma loja",
                "ajudar a loja",
                "ajudar minha oficina",
                "ajudar uma oficina",
                "como esse livro pode ajudar",
                "como este livro pode ajudar",
                "como o livro pode ajudar",
                "como essa obra pode ajudar",
                "como esta obra pode ajudar",
                "beneficios para a loja",
                "beneficios para uma loja",
                "beneficios para a oficina",
            ]
        )

        pergunta_sobre_adaptacao_lojas = any(
            marcador in pergunta_normalizada
            for marcador in [
                "loja com poucos membros",
                "loja com poucos irmaos",
                "oficina com poucos membros",
                "oficina com poucos irmaos",
                "loja com quadro reduzido",
                "oficina com quadro reduzido",
                "aplicado em uma loja com poucos membros",
                "aplicado em uma oficina com poucos membros",
                "funciona em loja pequena",
                "funciona em uma loja pequena",
                "funciona em oficina pequena",
                "funciona em uma oficina pequena",
                "adaptacao a realidade das lojas",
                "adaptar a realidade da loja",
                "adaptar a realidade da oficina",
                "acumulo de funcoes",
                "acumular funcoes",
            ]
        )

        pergunta_sobre_valor_compra = any(
            marcador in pergunta_normalizada
            for marcador in [
                "por que eu deveria comprar este livro",
                "por que eu deveria comprar o livro",
                "por que deveria comprar o livro",
                "por que deveria comprar este livro",
                "por que comprar este livro",
                "por que comprar o livro",
                "por que comprar esta obra",
                "vale a pena comprar este livro",
                "vale a pena comprar o livro",
                "motivos para comprar o livro",
                "motivos para adquirir o livro",
            ]
        )


        escopo_especifico = any(
            [
                pergunta_sobre_origem,
                pergunta_sobre_pre_requisitos,
                pergunta_sobre_forma_de_uso,
                pergunta_sobre_limite_de_pretensao,
                pergunta_sobre_delimitacao_ritualistica,
                pergunta_sobre_objetivo,
                pergunta_sobre_proposito,
                pergunta_sobre_publico,
                pergunta_sobre_abordagem,
                pergunta_sobre_conteudo,
                pergunta_sobre_ajuda_loja,
                pergunta_sobre_adaptacao_lojas,
                pergunta_sobre_valor_compra,
            ]
        )

        if escopo_especifico:
            if pergunta_sobre_origem and origem_motivacao:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Origem e motivação da obra:** {origem_motivacao}"
                )

            if (
                pergunta_sobre_pre_requisitos
                and pre_requisitos
            ):
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Pré-requisitos para o leitor:** {pre_requisitos}"
                )

            if pergunta_sobre_forma_de_uso and forma_de_uso:
                if linhas:
                    linhas.append("")

                linhas.append(
                    "**Como utilizar a obra:**"
                )

                for item in forma_de_uso:
                    linhas.append(
                        f"- {item}"
                    )

            if (
                pergunta_sobre_limite_de_pretensao
                and limite_de_pretensao
            ):
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Limite da proposta da obra:** {limite_de_pretensao}"
                )

            if (
                pergunta_sobre_delimitacao_ritualistica
                and delimitacao_ritualistica
            ):
                if linhas:
                    linhas.append("")

                principio = delimitacao_ritualistica.get(
                    "principio",
                    "",
                )

                fundamentos = delimitacao_ritualistica.get(
                    "fundamentos",
                    [],
                )

                conteudos_reservados = delimitacao_ritualistica.get(
                    "conteudos_reservados_as_fontes_oficiais",
                    [],
                )

                regra_ritual = delimitacao_ritualistica.get(
                    "regra_de_resposta",
                    "",
                )

                linhas.append(
                    "**Delimitação ritualística da obra:**"
                )

                if principio:
                    linhas.append(
                        principio
                    )

                if fundamentos:
                    linhas.append("")
                    linhas.append(
                        "**Fundamentos dessa delimitação:**"
                    )

                    for item in fundamentos:
                        linhas.append(
                            f"- {item}"
                        )

                if conteudos_reservados:
                    linhas.append("")
                    linhas.append(
                        "**Conteúdos reservados às fontes oficiais:**"
                    )

                    for item in conteudos_reservados:
                        linhas.append(
                            f"- {item}"
                        )

                if regra_ritual:
                    linhas.append("")
                    linhas.append(
                        f"**Regra adotada pela obra:** {regra_ritual}"
                    )

            if pergunta_sobre_objetivo and objetivo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Objetivo da obra:** {objetivo}"
                )

            if pergunta_sobre_proposito and proposito:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Propósito de continuidade:** {proposito}"
                )

            if pergunta_sobre_abordagem and conteudo_metodo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Abordagem:** {conteudo_metodo}"
                )

            if pergunta_sobre_conteudo and conteudo_metodo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**O que você encontrará na obra:** {conteudo_metodo}"
                )

            if pergunta_sobre_ajuda_loja:
                if linhas:
                    linhas.append("")

                linhas.append(
                    "**Como a obra pode contribuir para uma Loja:**"
                )

                if objetivo:
                    linhas.append(
                        f"{objetivo}"
                    )

                if adaptacao_lojas:
                    linhas.append("")
                    linhas.append(
                        f"{adaptacao_lojas}"
                    )

            if (
                pergunta_sobre_adaptacao_lojas
                and adaptacao_lojas
            ):
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Adaptação à realidade das Lojas:** {adaptacao_lojas}"
                )

            if pergunta_sobre_valor_compra:
                if linhas:
                    linhas.append("")

                linhas.append(
                    "**Por que a obra pode ser útil para você:**"
                )

                if objetivo:
                    linhas.append(
                        f"{objetivo}"
                    )

                if conteudo_metodo:
                    linhas.append("")
                    linhas.append(
                        f"{conteudo_metodo}"
                    )

                if adaptacao_lojas:
                    linhas.append("")
                    linhas.append(
                        f"{adaptacao_lojas}"
                    )

            if pergunta_sobre_publico and publico:
                if linhas:
                    linhas.append("")

                linhas.append(
                    "**Público a que se destina:**"
                )

                for item in publico:
                    linhas.append(
                        f"- {item}"
                    )

        else:
            if objetivo:
                if linhas:
                    linhas.append("")

                linhas.append(
                    f"**Objetivo da obra:** {objetivo}"
                )

            if proposito:
                linhas.append("")
                linhas.append(
                    f"**Propósito de continuidade:** {proposito}"
                )

            if conteudo_metodo:
                linhas.append("")
                linhas.append(
                    f"**Abordagem:** {conteudo_metodo}"
                )

            if publico:
                linhas.append("")
                linhas.append(
                    "**Público a que se destina:**"
                )

                for item in publico:
                    linhas.append(
                        f"- {item}"
                    )

    return "\n".join(linhas).strip()
def carregar_mapa_conceitual():
    with open(
        "mapa_conceitual.json",
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)

def localizar_mapa_conceitual(resultados):
    if not resultados:
        return None

    mapa = carregar_mapa_conceitual()

    capitulo_principal = resultados[0].get(
        "capitulo",
        "",
    )

    for capitulo in mapa.get("capitulos", []):
        if capitulo.get("capitulo") == capitulo_principal:
            return capitulo

    return None
def localizar_tema_conceitual(
    mapa,
    consulta,
):
    if not mapa or not consulta:
        return None

    consulta_normalizada = normalizar(
        consulta
    )

    for tema in mapa.get(
        "temas_conceituais",
        [],
    ):
        candidatos = [
            tema.get("tema", "")
        ]

        candidatos.extend(
            tema.get("aliases", [])
        )

        for candidato in candidatos:
            candidato_normalizado = normalizar(
                candidato
            )

            if (
                candidato_normalizado
                and candidato_normalizado
                in consulta_normalizada
            ):
                return tema

    return None

def localizar_tema_conceitual_global(
    consulta,
):
    if not consulta:
        return None, None

    mapa_completo = carregar_mapa_conceitual()

    for capitulo in mapa_completo.get(
        "capitulos",
        [],
    ):
        tema = localizar_tema_conceitual(
            capitulo,
            consulta,
        )

        if tema:
            return capitulo, tema

    return None, None

def localizar_tema_global_obra(
    consulta,
):
    if not consulta:
        return None

    mapa_global = carregar_mapa_global_obra()

    consulta_normalizada = normalizar(
        consulta
    )

    for tema in mapa_global.get(
        "temas_globais",
        [],
    ):
        candidatos = [
            tema.get("tema", "")
        ]

        candidatos.extend(
            tema.get("aliases", [])
        )

        for candidato in candidatos:
            candidato_normalizado = normalizar(
                candidato
            )

            if (
                candidato_normalizado
                and candidato_normalizado
                in consulta_normalizada
            ):
                return tema

    return None

def montar_resposta_tema_global_verificado(
    tema,
    pergunta,
):
    if not tema:
        return ""

    pergunta_normalizada = normalizar(
        pergunta
    )

    nome_tema = tema.get(
        "tema",
        "esse tema",
    )

    natureza = tema.get(
        "natureza",
        "",
    )

    sentido = tema.get(
        "sentido",
        "",
    )

    papel_governanca = tema.get(
        "papel_na_governanca",
        "",
    )

    contribuicao = tema.get(
        "contribuicao_continuidade",
        "",
    )

    limites = tema.get(
        "limites_de_atuacao",
        [],
    )

    pergunta_sobre_natureza = any(
        marcador in pergunta_normalizada
        for marcador in [
            "o que e",
            "qual e a natureza",
            "qual e a sua natureza",
            "qual a sua natureza",
            "sua natureza",
            "a sua natureza",
            "natureza do",
            "natureza da",
        ]
    )

    pergunta_sobre_sentido = any(
        marcador in pergunta_normalizada
        for marcador in [
            "qual e o sentido",
            "sentido institucional",
            "o que representa",
            "o que representam",
        ]
    )

    pergunta_sobre_papel = any(
        marcador in pergunta_normalizada
        for marcador in [
            "qual e o papel",
            "qual o papel",
            "qual e o seu papel",
            "qual o seu papel",
            "seu papel",
            "o seu papel",
            "papel do",
            "papel da",
            "qual e a funcao",
            "qual a funcao",
            "funcao do",
            "funcao da",
            "papel na governanca",
        ]
    )

    pergunta_sobre_contribuicao = any(
        marcador in pergunta_normalizada
        for marcador in [
            "como contribui",
            "como contribuem",
            "contribuicao para",
            "contribuicao do",
            "contribuicao da",
            "continuidade institucional",
        ]
    )

    pergunta_sobre_limites = any(
        marcador in pergunta_normalizada
        for marcador in [
            "quais sao os limites",
            "quais os limites",
            "quais sao os seus limites",
            "quais os seus limites",
            "seus limites",
            "os seus limites",
            "limites de atuacao",
            "limites do",
            "limites da",
            "o que nao pode",
            "o que nao podem",
        ]
    )

    aspecto_especifico = any(
        [
            pergunta_sobre_natureza,
            pergunta_sobre_sentido,
            pergunta_sobre_papel,
            pergunta_sobre_contribuicao,
            pergunta_sobre_limites,
        ]
    )

    linhas = []

    if aspecto_especifico:
        if pergunta_sobre_natureza and natureza:
            linhas.append(
                f"**Natureza:** {natureza}"
            )

        if pergunta_sobre_sentido and sentido:
            if linhas:
                linhas.append("")

            linhas.append(
                f"**Sentido institucional:** {sentido}"
            )

        if pergunta_sobre_papel and papel_governanca:
            if linhas:
                linhas.append("")

            linhas.append(
                f"**Papel na governança:** {papel_governanca}"
            )

        if pergunta_sobre_contribuicao and contribuicao:
            if linhas:
                linhas.append("")

            linhas.append(
                f"**Contribuição para a continuidade:** {contribuicao}"
            )

        if pergunta_sobre_limites and limites:
            if linhas:
                linhas.append("")

            linhas.append(
                "**Limites importantes estabelecidos pela obra:**"
            )

            for limite in limites:
                linhas.append(
                    f"- {limite}"
                )

    else:
        linhas.append(
            f"Na obra, **{nome_tema}** é tratado como um tema "
            "institucional específico."
        )

        if natureza:
            linhas.append("")
            linhas.append(
                f"**Natureza:** {natureza}"
            )

        if sentido:
            linhas.append("")
            linhas.append(
                f"**Sentido institucional:** {sentido}"
            )

        if papel_governanca:
            linhas.append("")
            linhas.append(
                f"**Papel na governança:** {papel_governanca}"
            )

        if contribuicao:
            linhas.append("")
            linhas.append(
                f"**Contribuição para a continuidade:** {contribuicao}"
            )

        if limites:
            linhas.append("")
            linhas.append(
                "**Limites importantes estabelecidos pela obra:**"
            )

            for limite in limites:
                linhas.append(
                    f"- {limite}"
                )

    return "\n".join(linhas)

def montar_resposta_conceitual_verificada(
    mapa,
    tema,
):
    if not mapa or not tema:
        return ""

    linhas = []

    nome_tema = tema.get(
        "tema",
        "",
    )

    denominacao = tema.get(
        "denominacao_na_obra",
        nome_tema,
    )

    posicao = tema.get(
        "posicao_editorial",
        "",
    )

    sentido = tema.get(
        "sentido",
        "",
    )

    linhas.append(
        f"Na obra, **{denominacao}** ocupa a seguinte posição conceitual: "
        f"{posicao}."
    )

    if sentido:
        linhas.append("")
        linhas.append(
            f"Seu sentido é: {sentido}"
        )

    desenvolvimentos = tema.get(
        "desenvolvimentos_relacionados",
        [],
    )

    if desenvolvimentos:
        linhas.append("")
        linhas.append(
            "A obra desenvolve esse tema também por meio "
            "de elementos editoriais relacionados, preservando "
            "a classificação própria de cada um:"
        )

        for desenvolvimento in desenvolvimentos:
            nome = desenvolvimento.get(
                "nome",
                "",
            )

            classificacao = desenvolvimento.get(
                "classificacao",
                "",
            )

            linhas.append(
                f"- **{nome}** — {classificacao}"
            )

    linhas.append("")
    linhas.append(
        "Esses elementos não alteram a posição conceitual "
        "do tema na estrutura da obra."
    )

    return "\n".join(linhas)

def montar_resposta_explicativa_verificada(
    mapa,
    tema,
):
    if not mapa or not tema:
        return ""

    denominacao = tema.get(
        "denominacao_na_obra",
        tema.get("tema", "esse tema"),
    )

    capitulo = mapa.get(
        "capitulo",
        "",
    )

    sentido = tema.get(
        "sentido",
        "",
    )

    linhas = []

    linhas.append(
        f"A obra trata de **{denominacao}** como um tema "
        "relacionado à governança e à continuidade institucional da Loja."
    )

    if capitulo:
        linhas.append("")
        linhas.append(
            f"**Onde encontrar:** {capitulo}."
        )

    if sentido:
        linhas.append("")
        linhas.append(
            f"**Princípio central:** {sentido}"
        )

    linhas.append("")
    linhas.append(
        "Essa é a orientação geral apresentada pela obra. "
        "A elaboração de procedimentos, planos ou instrumentos "
        "de implementação depende de um nível de acesso voltado "
        "à aplicação."
    )

    return "\n".join(linhas)

def montar_resposta_interpretativa_verificada(
    mapa,
    tema,
):
    if not mapa or not tema:
        return ""

    linhas = []

    denominacao = tema.get(
        "denominacao_na_obra",
        tema.get("tema", "esse tema"),
    )

    capitulo = mapa.get(
        "capitulo",
        "",
    )

    posicao = tema.get(
        "posicao_editorial",
        "",
    )

    sentido = tema.get(
        "sentido",
        "",
    )

    linhas.append(
        f"Para interpretar **{denominacao}** à luz da obra, "
        "o primeiro ponto é compreender o lugar que esse tema "
        "ocupa na estrutura do livro."
    )

    if capitulo:
        linhas.append("")
        linhas.append(
            f"**Onde consultar:** {capitulo}."
        )

    if posicao:
        linhas.append("")
        linhas.append(
            f"**Posição conceitual:** {posicao}."
        )

    if sentido:
        linhas.append("")
        linhas.append(
            f"**Critério central de interpretação:** {sentido}"
        )

    desenvolvimentos = tema.get(
        "desenvolvimentos_relacionados",
        [],
    )

    if desenvolvimentos:
        linhas.append("")
        linhas.append(
            "Para aprofundar a análise, observe também como a obra "
            "distingue os elementos relacionados ao tema:"
        )

        for desenvolvimento in desenvolvimentos:
            nome = desenvolvimento.get(
                "nome",
                "",
            )

            classificacao = desenvolvimento.get(
                "classificacao",
                "",
            )

            linhas.append(
                f"- **{nome}** — {classificacao}"
            )

    linhas.append("")
    linhas.append(
        "Esses critérios ajudam a compreender e estruturar a decisão "
        "à luz da obra, sem substituir a análise do caso concreto nem "
        "transformar esta interpretação em um plano pronto de execução."
    )

    return "\n".join(linhas)

def montar_contexto_conceitual(mapa):
    if not mapa:
        return ""

    linhas = []

    linhas.append(
        f"CARGO: {mapa.get('cargo', '')}"
    )
    linhas.append(
        f"CAPÍTULO: {mapa.get('capitulo', '')}"
    )
    linhas.append("")

    missao = mapa.get("missao_essencial")

    if missao:
        linhas.append(
            f"MISSÃO ESSENCIAL: {missao}"
        )
        linhas.append("")

    pilares = mapa.get("pilares_criticos", {})

    if pilares:
        quantidade = pilares.get("quantidade")

        linhas.append(
            f"PILARES CRÍTICOS: {quantidade}"
        )

        for item in pilares.get("itens", []):
            linhas.append(
                "- "
                + item.get("nome", "")
                + " — "
                + item.get("subtitulo", "")
            )

            sentido = item.get("sentido")

            if sentido:
                linhas.append(
                    f"  Sentido: {sentido}"
                )

        linhas.append("")
    temas_conceituais = mapa.get(
        "temas_conceituais",
        [],
    )

    if temas_conceituais:
        linhas.append(
            "TEMAS CONCEITUAIS ESPECÍFICOS:"
        )

        for tema in temas_conceituais:
            linhas.append(
                f"TEMA: {tema.get('tema', '')}"
            )
            linhas.append(
                "DENOMINAÇÃO NA OBRA: "
                + tema.get(
                    "denominacao_na_obra",
                    "",
                )
            )
            linhas.append(
                "POSIÇÃO EDITORIAL: "
                + tema.get(
                    "posicao_editorial",
                    "",
                )
            )
            linhas.append(
                "SENTIDO: "
                + tema.get(
                    "sentido",
                    "",
                )
            )

            desenvolvimentos = tema.get(
                "desenvolvimentos_relacionados",
                [],
            )

            if desenvolvimentos:
                linhas.append(
                    "DESENVOLVIMENTOS RELACIONADOS:"
                )

                for desenvolvimento in desenvolvimentos:
                    linhas.append(
                        "- "
                        + desenvolvimento.get(
                            "nome",
                            "",
                        )
                        + " — "
                        + desenvolvimento.get(
                            "classificacao",
                            "",
                        )
                    )

            regras_tema = tema.get(
                "regras_de_fidelidade",
                [],
            )

            if regras_tema:
                linhas.append(
                    "REGRAS DE FIDELIDADE DO TEMA:"
                )

                for regra in regras_tema:
                    linhas.append(
                        f"- {regra}"
                    )

            linhas.append("")
    estrutura = mapa.get(
        "estrutura_editorial",
        [],
    )

    if estrutura:
        linhas.append(
            "ESTRUTURA EDITORIAL DO CAPÍTULO:"
        )

        for secao in estrutura:
            linhas.append(
                f"- {secao}"
            )

        linhas.append("")

    regras = mapa.get(
        "regras_de_fidelidade",
        [],
    )

    if regras:
        linhas.append(
            "REGRAS DE FIDELIDADE:"
        )

        for regra in regras:
            linhas.append(
                f"- {regra}"
            )

    return "\n".join(linhas)

def carregar_base_operacional():
    with open(
        "base_operacional.json",
        "r",
        encoding="utf-8",
    ) as arquivo:
        return json.load(arquivo)

def localizar_tema_operacional(consulta):
    base = carregar_base_operacional()
    consulta_normalizada = consulta.lower().strip()

    for tema in base.get("temas", []):
        nomes_possiveis = [
            tema.get("tema", ""),
            *tema.get("aliases", []),
        ]

        for nome in nomes_possiveis:
            if nome.lower() in consulta_normalizada:
                return tema

    return None

def montar_contexto_operacional(tema):
    if not tema:
        return ""

    linhas = [
        f"TEMA: {tema.get('tema', '')}",
        "",
    ]

    for protocolo in tema.get("protocolos", []):
        linhas.append(
            f"PROTOCOLO: {protocolo.get('nome', '')}"
        )
        linhas.append(
            f"CAPÍTULO: {protocolo.get('capitulo', '')}"
        )
        linhas.append(
            "RESPONSÁVEL PRINCIPAL: "
            + protocolo.get(
                "responsavel_principal",
                "",
            )
        )

        for acao in protocolo.get("acoes", []):
            texto = (
                f"{acao.get('ordem', '')}. "
                f"{acao.get('acao', '')}"
            )

            prazo = acao.get("prazo")

            if prazo:
                texto += f" | Prazo: {prazo}"

            linhas.append(texto)

        linhas.append("")

    return "\n".join(linhas)

def montar_resposta_operacional_verificada(
    tema,
    pergunta,
):
    linhas = []

    nome_tema = tema.get(
        "tema",
        "aplicação operacional",
    )

    linhas.append(
        f"**Aplicação baseada na obra — {nome_tema.title()}**"
    )
    linhas.append("")

    pergunta_minuscula = pergunta.lower()

    if (
        tema.get("tema") == "transição de gestão"
        and "próximo mês" in pergunta_minuscula
    ):
        linhas.append(
            "**Adaptação ao cenário informado:** "
            "a obra recomenda iniciar determinadas providências "
            "cerca de 60 dias antes do término do mandato. "
            "Como a transição ocorrerá no próximo mês, essas "
            "providências devem ser iniciadas imediatamente. "
            "Os demais prazos não serão artificialmente "
            "subdivididos."
        )
        linhas.append("")

    for protocolo in tema.get("protocolos", []):
        linhas.append(
            f"### {protocolo.get('nome', '')}"
        )

        linhas.append(
            "**Responsável principal:** "
            + protocolo.get(
                "responsavel_principal",
                "Não indicado",
            )
        )

        linhas.append("")

        for acao in protocolo.get("acoes", []):
            ordem = acao.get("ordem", "")
            texto_acao = acao.get("acao", "")
            prazo = acao.get("prazo")

            linhas.append(
                f"**{ordem}. {texto_acao}**"
            )

            if prazo:
                linhas.append(
                    f"Prazo indicado pela obra: {prazo}."
                )
            else:
                linhas.append(
                    "Prazo específico: não indicado pela obra."
                )

            linhas.append("")

    linhas.append(
        "*Esta estrutura utiliza somente ações e prazos "
        "registrados na base operacional verificada da obra.*"
    )

    return "\n".join(linhas)

def montar_fontes(resultados):
    blocos = []

    for numero, resultado in enumerate(resultados, start=1):
        blocos.append(
            f"""
Capítulo: {resultado['capitulo']}
Página PDF: {resultado['pagina_pdf']}
Trecho:
{resultado['texto']}
"""
        )

    return "\n".join(blocos)
def perguntar_ao_llama(
    pergunta,
    cobertura,
    resultados,
    historico,
    perfil_acesso="Visitante",
    consulta_busca=None,
):

    regras_perfil = {
        "Visitante": """
PERFIL DE ACESSO: VISITANTE

O usuário pode descobrir se e onde a obra trata de um tema
e compreender seus princípios gerais.

Pode receber:
- explicações conceituais breves;
- indicação dos temas ou capítulos relacionados;
- uma pequena ação inicial, quando útil.

Não pode receber:
- protocolos completos;
- checklists completos;
- rotinas estruturadas;
- modelos prontos;
- conjuntos de indicadores;
- reconstrução extensa do método da obra;
- planos personalizados de implementação.

Regra central:
explique o problema e o princípio, mas preserve o método
detalhado para a obra e para níveis superiores de acesso.
""",

        "Leitor": """
PERFIL DE ACESSO: LEITOR

O usuário é reconhecido como possuidor da obra.

Pode:
- localizar onde a obra trata de um assunto;
- compreender conceitos e critérios;
- relacionar conteúdos de capítulos diferentes;
- receber orientação de leitura;
- interpretar uma situação concreta à luz da obra;
- receber critérios para pensar e decidir.

Não deve receber um projeto completo de implementação,
um plano personalizado detalhado ou uma consultoria
operacional pronta para execução.

Regra central:
ajude o leitor a compreender e navegar pela obra,
sem substituir a aplicação personalizada.
""",

        "Premium": """
PERFIL DE ACESSO: PREMIUM

O usuário pode aplicar os princípios da obra à sua realidade.

Pode receber:
- diagnóstico de situações concretas;
- planos personalizados;
- sequências de ações;
- cronogramas;
- rotinas e checklists adaptados;
- estruturas de transição;
- análise de indicadores;
- acompanhamento de projetos.

Mesmo no Premium:
- não reproduza extensamente o texto da obra;
- não invente normas;
- não substitua regulamentos da Potência;
- não invente procedimentos ritualísticos;
- não atribua ao livro conteúdo que ele não apresenta.
REGRAS PARA APLICAÇÃO PERSONALIZADA:

- Cada tarefa proposta deve estar sustentada por uma ação,
rotina, protocolo ou boa prática efetivamente presente nas
fontes recuperadas.

- Não transforme indicadores de desempenho ou indicadores
de alerta em tarefas de execução. Indicadores servem para
avaliar ou sinalizar condições, salvo se a própria fonte
determinar uma ação específica associada a eles.

- Preserve os responsáveis institucionais indicados pela
obra. Não crie cargos genéricos como "gestor", "equipe" ou
"funcionários" quando a fonte identifica um Oficial específico.

- Preserve os prazos recomendados pela obra quando existirem.

- Se o prazo informado pelo usuário for menor que o prazo
recomendado pela obra, explique essa diferença e apresente
a adaptação como uma adequação ao cenário concreto.

- Nesse caso, não invente subdivisões temporais como
"2 semanas antes", "1 semana antes" ou "no dia seguinte"
sem fundamento nas fontes. Use "iniciar imediatamente"
quando necessário e mantenha os demais marcos que a obra
efetivamente sustenta.

- Diferencie claramente aquilo que é orientação da obra
daquilo que é adaptação feita para a realidade apresentada
pelo usuário.

- Não acrescente tarefas apenas para tornar o plano mais
completo. Um plano menor e totalmente sustentado pelas
fontes é preferível a um plano abrangente com elementos
inventados.
""",
    }

    instrucao_perfil = regras_perfil.get(
        perfil_acesso,
        regras_perfil["Visitante"],
    )

    tema_global_obra = None

    if consulta_busca:
        tema_global_obra = localizar_tema_global_obra(
            consulta_busca
        )
    secoes_globais_obra = localizar_secoes_globais_obra(
    pergunta
)
    mapa_conceitual = localizar_mapa_conceitual(
        resultados
    )

    tema_conceitual = None

    if consulta_busca:
        (
            mapa_global,
            tema_global,
        ) = localizar_tema_conceitual_global(
            consulta_busca
        )

        if tema_global:
            mapa_conceitual = mapa_global
            tema_conceitual = tema_global

    if (
        not tema_conceitual
        and mapa_conceitual
        and consulta_busca
    ):
        tema_conceitual = localizar_tema_conceitual(
            mapa_conceitual,
            consulta_busca,
        )

    contexto_conceitual = ""

    if mapa_conceitual:
        contexto_conceitual = montar_contexto_conceitual(
            mapa_conceitual
        )
    if (
        perfil_acesso == "Visitante"
        and st.session_state.acao_pendente == "explicar_tema"
        and eh_aceite_acao_pendente(pergunta)
        and tema_conceitual
        and mapa_conceitual
    ):
        st.session_state.acao_pendente = None

        return montar_resposta_explicativa_verificada(
            mapa_conceitual,
            tema_conceitual,
        )
    if (
        perfil_acesso == "Leitor"
        and st.session_state.acao_pendente == "interpretar_tema"
        and eh_aceite_acao_pendente(pergunta)
        and tema_conceitual
        and mapa_conceitual
    ):
        st.session_state.acao_pendente = None

        return montar_resposta_interpretativa_verificada(
            mapa_conceitual,
            tema_conceitual,
        )
    tema_operacional = None

    if (
        perfil_acesso == "Premium"
        and consulta_busca
    ):
        tema_operacional = localizar_tema_operacional(
            consulta_busca
        )

    contexto_operacional = ""

    if tema_operacional:
        contexto_operacional = montar_contexto_operacional(
            tema_operacional
        )

    pergunta_minuscula = pergunta.lower()

    marcadores_aplicacao = [
        "monte um plano",
        "crie um plano",
        "faça um plano",
        "elabore um plano",
        "monte um cronograma",
        "crie um cronograma",
        "faça um cronograma",
        "monte um checklist",
        "crie um checklist",
        "faça um checklist",
        "passo a passo",
        "tarefas, responsáveis e prazos",
        "tarefas e prazos",
        "responsáveis e prazos",
        "como implementar",
        "como aplicar",
        "aplique à nossa loja",
        "aplicar à nossa loja",
    ]

    pedido_aplicacao = any(
        marcador in pergunta_minuscula
        for marcador in marcadores_aplicacao
    )
    
    marcadores_explicacao_conceitual = [
        "indique como o tema é tratado",
        "indique como o tema e tratado",
        "como o tema é tratado",
        "como o tema e tratado",
        "como esse tema é tratado",
        "como esse tema e tratado",
        "como este tema é tratado",
        "como este tema e tratado",
        "o que o livro diz sobre",
        "o que a obra diz sobre",
        "como o livro trata",
        "como a obra trata",
        "organizado conceitualmente",
        "organizada conceitualmente",
        "organização conceitual",
        "organizacao conceitual",
        "estrutura conceitual",
        "como o tema se organiza",
        "como esse tema se organiza",
        "como este tema se organiza",
    ]

    pedido_explicacao_conceitual = any(
        marcador in pergunta_minuscula
        for marcador in marcadores_explicacao_conceitual
    )    

    marcadores_consulta_simples = [
        "o livro fala sobre",
        "a obra fala sobre",
        "o livro aborda",
        "a obra aborda",
        "o livro trata de",
        "a obra trata de",
        "há no livro",
        "ha no livro",
        "existe no livro",
    ]

    pedido_consulta_simples = any(
        marcador in pergunta_minuscula
        for marcador in marcadores_consulta_simples
    )

    marcadores_importancia_continuidade = [
        "por que",
        "porque",
        "qual a importância",
        "qual a importancia",
        "por que é importante",
        "por que e importante",
        "qual a contribuição",
        "qual a contribuicao",
        "como contribui para a continuidade",
        "continuidade institucional",
    ]

    pedido_importancia_continuidade = any(
        marcador in pergunta_minuscula
        for marcador in marcadores_importancia_continuidade
    )

    if (
    pedido_importancia_continuidade
    and cobertura in ["DIRETA", "RELACIONADA"]
    and tema_conceitual
    and tema_conceitual.get("contribuicao_continuidade")
    ):
        denominacao = tema_conceitual.get(
            "denominacao_na_obra",
            tema_conceitual.get("tema", "esse tema"),
        )

        contribuicao = tema_conceitual.get(
            "contribuicao_continuidade"
        )

        return (
            f"Na obra, **{denominacao}** contribui para a "
            f"continuidade institucional da seguinte forma:\n\n"
            f"{contribuicao}"
        )

    if (
        pedido_consulta_simples
        and cobertura == "DIRETA"
        and tema_conceitual
    ):
        denominacao = tema_conceitual.get(
            "denominacao_na_obra",
            tema_conceitual.get("tema", "esse tema"),
        )

        return (
            f"Sim. A obra trata diretamente de **{denominacao}**. "
            "Posso também indicar como esse tema é organizado "
            "conceitualmente no livro."
        )

    if (
        perfil_acesso == "Visitante"
        and pedido_aplicacao
    ):
        st.session_state.acao_pendente = "explicar_tema"
        return (
            "Posso explicar o que a obra apresenta sobre esse tema "
            "e seus princípios gerais, mas o perfil Visitante não "
            "inclui a elaboração de planos personalizados, cronogramas, "
            "checklists ou estruturas de implementação. "
            "Posso, porém, indicar como o tema é tratado no livro."
        )
    if (
        perfil_acesso == "Leitor"
        and pedido_aplicacao
    ):
        st.session_state.acao_pendente = "interpretar_tema"

        return (
            "Como Leitor, posso ajudá-lo a interpretar essa situação "
            "à luz da obra, indicar os capítulos pertinentes e apresentar "
            "critérios para estruturar a decisão. "
            "A elaboração de um plano personalizado com tarefas, "
            "responsáveis, prazos ou instrumentos prontos de execução "
            "fica reservada ao nível Premium."
        )
    if (
        perfil_acesso == "Premium"
        and pedido_aplicacao
        and cobertura != "DIRETA"
    ):
        return (
            "Este pedido exige uma aplicação personalizada, mas a busca "
            "não encontrou cobertura direta suficiente na obra para "
            "construir o plano com segurança. "
            "Não vou preencher essas lacunas com orientações gerais "
            "ou conteúdos que não estejam sustentados pelo livro."
        )
    if (
        perfil_acesso == "Premium"
        and pedido_aplicacao
        and cobertura == "DIRETA"
        and tema_operacional
    ):
        return montar_resposta_operacional_verificada(
            tema_operacional,
            pergunta,
        )

    if (
        pedido_explicacao_conceitual
        and cobertura == "DIRETA"
        and tema_conceitual
        and mapa_conceitual
    ):
        return montar_resposta_conceitual_verificada(
            mapa_conceitual,
            tema_conceitual,
        )
    if secoes_globais_obra:
        return montar_resposta_secoes_globais_obra(
    secoes_globais_obra,
    pergunta,
)    
    # ---------------------------------------------------------
    # COBERTURA FRACA
    # O código bloqueia qualquer tentativa de completar lacunas.
    # ---------------------------------------------------------

    if cobertura == "FRACA":
        return (
            "Não encontrei na obra tratamento específico suficiente "
            "sobre essa questão. Há conteúdos mais gerais que podem "
            "ter relação com o tema, mas não devo atribuir ao livro "
            "uma orientação específica que ele não apresenta."
        )
    if cobertura == "RELACIONADA":
        return (
            "Não encontrei na obra tratamento específico sobre esse tema. "
            "A busca identificou conteúdos relacionados, mas eles não "
            "estabelecem explicitamente a conexão com o assunto perguntado. "
            "Por isso, não devo atribuir essa relação ao livro."
        )
    fontes = montar_fontes(resultados)
    
    if (
        perfil_acesso == "Premium"
        and pedido_aplicacao
        and cobertura == "DIRETA"
    ):
        resultados_aplicacao = resultados[:3]
        fontes = montar_fontes(resultados_aplicacao)

    resultados_relacionados = [
        resultado
        for resultado in resultados
        if resultado["semantica"] >= 0.46
        and resultado["lexical"] >= 0.20
    ]

    if not resultados_relacionados:
        resultados_relacionados = resultados[:1]

    fontes_relacionadas = montar_fontes(
        resultados_relacionados
    )

    # ---------------------------------------------------------
    # COBERTURA RELACIONADA
    # O sistema garante a ressalva.
    # O Llama apenas explica a relação encontrada nas fontes.
    # ---------------------------------------------------------

    if cobertura == "RELACIONADA":

        instrucao = f"""
Você está analisando trechos recuperados da obra
A Loja que Permanece.

A busca determinou que a pergunta do usuário NÃO possui
tratamento específico suficiente na obra.

Sua tarefa é apenas resumir, em UMA ou DUAS frases, o que
os trechos recuperados efetivamente apresentam.

REGRAS OBRIGATÓRIAS:

1. Não explique a relação entre os trechos e a pergunta.

2. Não tente aproximar conceitualmente o conteúdo recuperado
do tema perguntado.

3. Não use o tema da pergunta como rótulo para conteúdos
que não o mencionam.

4. Resuma somente aquilo que estiver explicitamente
sustentado pelas fontes.

5. Preserve os papéis institucionais corretamente.
Não transforme Oficiais em funcionários, empregados ou
categorias equivalentes se a fonte não usar esses termos.

6. Não invente recomendações, conceitos, funções,
relações ou consequências.

7. Não use conhecimento geral para completar lacunas.

8. Use português do Brasil.

9. Seja breve e natural.

10. Se os trechos forem muito vagos ou não permitirem
um resumo seguro, responda exatamente:

"Os trechos recuperados não oferecem conteúdo suficientemente
específico para um resumo seguro."

FONTES RECUPERADAS:

{fontes_relacionadas}
"""

        mensagens = [
            {
                "role": "system",
                "content": instrucao,
            },
            {
                "role": "user",
                "content": pergunta,
            },
        ]

    # ---------------------------------------------------------
    # COBERTURA DIRETA
    # O Llama responde normalmente com base na obra.
    # ---------------------------------------------------------
    elif (
        perfil_acesso == "Premium"
        and pedido_aplicacao
        and tema_operacional
    ):
        instrucao = f"""
Você é o Assistente A Loja que Permanece.

O usuário possui acesso Premium e pediu uma aplicação
personalizada.

Para esta resposta, existe uma BASE OPERACIONAL VERIFICADA.

Sua tarefa é adaptar essa base à situação apresentada pelo
usuário, sem criar novas ações.

REGRAS OBRIGATÓRIAS:

1. Use SOMENTE as ações, responsáveis e prazos existentes
na BASE OPERACIONAL VERIFICADA abaixo.

2. Não acrescente tarefas para completar o plano.

3. Não invente responsáveis.

4. Não invente prazos.

5. Não transforme indicadores, princípios ou objetivos em
tarefas.

6. Preserve a separação entre os diferentes protocolos e
Ofícios.

7. O usuário informou que a mudança ocorrerá no próximo mês.
Se a base recomendar antecedência maior, explique isso
claramente e indique que as providências devem começar
imediatamente, sem inventar novos prazos intermediários.

8. Você pode reorganizar a apresentação para torná-la útil
ao usuário, mas não pode criar conteúdo operacional novo.

9. Diferencie, quando necessário:
- orientação original da obra;
- adaptação necessária ao cenário informado pelo usuário.

10. Responda em português do Brasil.

11. Seja claro, prático e objetivo.

PERGUNTA DO USUÁRIO:

{pergunta}

BASE OPERACIONAL VERIFICADA:

{contexto_operacional}
"""

        mensagens = [
            {
                "role": "system",
                "content": instrucao,
            },
            {
                "role": "user",
                "content": pergunta,
            },
        ]
    
    else:
        instrucao = f"""
Você é um protótipo do Assistente A Loja que Permanece.

Responda sempre em português do Brasil.

A busca determinou que a obra trata diretamente da questão
apresentada.

REGRAS DO PERFIL ATUAL:

{instrucao_perfil}

MAPA CONCEITUAL VERIFICADO:

{contexto_conceitual}

REGRAS DE FIDELIDADE CONCEITUAL:

- Quando o MAPA CONCEITUAL VERIFICADO contiver informação,
ele define a organização conceitual que deve ser preservada.

- Preserve exatamente quantidades, nomes, classificações
e relações registradas no mapa.

- Não transforme seções distintas do capítulo em partes
de uma mesma classificação.

- Não crie novos pilares, categorias, etapas, princípios
ou agrupamentos que não estejam explicitamente registrados.

- Se os trechos recuperados contiverem outras informações
verdadeiras do capítulo, elas podem ser mencionadas, mas
não devem ser reorganizadas como se integrassem uma
classificação definida no mapa.

- Quando houver conflito entre uma inferência possível a
partir dos trechos e a estrutura registrada no mapa,
preserve a estrutura do mapa.

Sua tarefa é responder de forma breve e fiel ao conteúdo
recuperado.

REGRAS OBRIGATÓRIAS:

1. Você pode afirmar que a obra trata do tema.

2. Explique o conteúdo principal encontrado, em vez de apenas
dizer em quais fontes ele aparece.

3. Não mencione "FONTE 1", "FONTE 2", "FONTE 3" ou qualquer
outra numeração técnica interna.

4. Não exponha a mecânica da busca, do RAG, das pontuações ou
da seleção de trechos.

5. Não ignore os trechos mais relevantes sem motivo.
Priorize os conteúdos que melhor respondem à pergunta.

6. Não misture conteúdos de capítulos diferentes como se
formassem um único protocolo.

7. Quando diferentes capítulos apresentarem perspectivas
distintas sobre o mesmo tema, deixe isso claro de forma
natural.

8. Não invente conteúdo do livro.

9. Não use conhecimento geral para preencher silenciosamente
lacunas das fontes.

10. Não invente datas específicas, nomes de responsáveis,
prazos calendáricos ou marcos temporais que não tenham sido
fornecidos pelo usuário ou sustentados pelas fontes.

Se o usuário indicar apenas um horizonte relativo, como
"no próximo mês", use formulações relativas, por exemplo:
"4 semanas antes", "2 semanas antes", "na semana da posse"
ou "nos primeiros 30 dias", conforme o conteúdo recuperado.

11. Ao montar uma aplicação com base em mais de um capítulo,
preserve o escopo de cada cargo. Não transforme uma rotina
específica de um Ofício em obrigação geral da Loja.

12. Não reproduza longos trechos literalmente.
Explique com suas próprias palavras.

13. Seja conciso e natural, adequado a uma conversa de WhatsApp.

14. Para perguntas simples como "o livro fala sobre X?",
responda preferencialmente em duas etapas:
primeiro confirme ou negue de forma direta;
depois resuma em uma ou duas frases como o tema aparece na obra.

FONTES RECUPERADAS:

{fontes}
"""

        mensagens = [
            {
                "role": "system",
                "content": instrucao,
            }
        ]

        mensagens += historico[-6:]

    payload = {
        "model": MODELO_CHAT,
        "messages": mensagens,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    requisicao = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    
    inicio_llama = time.perf_counter()

    with urllib.request.urlopen(
        requisicao,
        timeout=600,
    ) as resposta:

        resultado = json.loads(
            resposta.read().decode("utf-8")
        )

    tempo_llama = time.perf_counter() - inicio_llama
    st.session_state.tempo_llama = tempo_llama

    st.session_state.ollama_total = (
        (resultado.get("total_duration") or 0)
        / 1_000_000_000
    )

    st.session_state.ollama_load = (
        (resultado.get("load_duration") or 0)
        / 1_000_000_000
    )

    st.session_state.ollama_prompt = (
        (resultado.get("prompt_eval_duration") or 0)
        / 1_000_000_000
    )

    st.session_state.ollama_prompt_tokens = (
        resultado.get("prompt_eval_count") or 0
    )

    st.session_state.ollama_geracao = (
        (resultado.get("eval_duration") or 0)
        / 1_000_000_000
    )

    st.session_state.ollama_geracao_tokens = (
        resultado.get("eval_count") or 0
    )


    texto_gerado = (
        resultado["message"]["content"].strip()
    )

    if cobertura == "RELACIONADA":
        return (
            "Não encontrei na obra tratamento específico "
            "sobre esse tema. "
            + texto_gerado
            + " A obra não estabelece explicitamente a conexão "
            "entre esses conteúdos e o tema perguntado."
        )

    return texto_gerado

st.set_page_config(
    page_title="A Loja que Permanece",
    page_icon="📖",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 18% 12%, #10243a 0, transparent 34%),
            linear-gradient(135deg, #0b1826 0%, #07111c 100%);
        color: #dfd2b4;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1040px;
        padding-top: 2.5rem;
    }

    h1, h2, h3 {
        color: #dfb75f !important;
        font-family: Georgia, "Times New Roman", serif !important;
        font-weight: 400 !important;
        letter-spacing: 0.04em;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #d8c9a7;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        min-height: 3rem;
        border: 1px solid #9f7931;
        border-radius: 2px;
        font-weight: 600;
        letter-spacing: 0.06em;
    }

    div.stButton > button[kind="primary"] {
        color: #07111c;
        background: linear-gradient(180deg, #d6ae56, #ac8135);
        border-color: #dfbd70;
    }

    div.stButton > button[kind="secondary"],
    div.stFormSubmitButton > button {
        color: #d9b45e;
        background: rgba(7, 17, 28, 0.72);
    }

    div.stDownloadButton > button,
    a[data-testid="stLinkButton"] {
        color: #f2d58f !important;
        background: rgba(7, 17, 28, 0.92) !important;
        border: 1px solid #b78e40 !important;
        border-radius: 2px !important;
        min-height: 3rem;
        font-weight: 600 !important;
        letter-spacing: 0.04em;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stDateInput"] input {
        background: #f4f1e9 !important;
        color: #111820 !important;
        caret-color: #111820 !important;
        -webkit-text-fill-color: #111820 !important;
    }

    div[data-testid="stTextInput"] input::placeholder,
    div[data-testid="stTextArea"] textarea::placeholder,
    div[data-testid="stDateInput"] input::placeholder {
        color: #69717a !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #69717a !important;
    }

    .st-key-apoio_barra_emergencia {
        position: sticky;
        top: 0.4rem;
        z-index: 990;
        padding: 0.35rem 0;
        background: linear-gradient(180deg, #0b1826 82%, transparent);
    }

    [data-testid="stImage"] img {
        border: 1px solid #9f7931;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.38);
    }

    .alq-kicker {
        color: #bc9348 !important;
        font-family: Arial, sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }

    .alq-title {
        color: #e1bc65 !important;
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.2rem, 5vw, 3.5rem);
        line-height: 1.05;
        letter-spacing: 0.05em;
        margin: 0 0 0.7rem;
        text-transform: uppercase;
    }

    .alq-subtitle {
        color: #a9946c !important;
        font-size: 0.92rem;
        letter-spacing: 0.08em;
        margin-bottom: 1.8rem;
        text-transform: uppercase;
    }

    .alq-rule {
        width: 4.5rem;
        height: 1px;
        background: #b78e40;
        margin-bottom: 1.8rem;
    }

    .alq-intro {
        color: #d6c59d !important;
        font-family: Arial, sans-serif;
        line-height: 1.65;
        margin-bottom: 1.2rem;
        max-width: 31rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "porta_entrada" not in st.session_state:
    st.session_state.porta_entrada = None

if st.user.is_logged_in and st.session_state.porta_entrada is None:
    st.session_state.porta_entrada = "Leitor"


def escolher_porta_entrada(destino):
    st.session_state.porta_entrada = destino


def voltar_para_entrada():
    st.session_state.porta_entrada = None
    st.session_state.pop("leitor_autenticado", None)


def validar_codigo_leitor(codigo):
    codigo_normalizado = codigo.strip().upper()

    if not codigo_normalizado:
        return False

    hash_informado = hashlib.sha256(
        codigo_normalizado.encode("utf-8")
    ).hexdigest()

    try:
        codigos_leitores = st.secrets["codigos_leitores"]
    except (FileNotFoundError, KeyError):
        return False

    return any(
        hmac.compare_digest(
            hash_informado,
            str(hash_armazenado),
        )
        for hash_armazenado in codigos_leitores.values()
    )


def requisicao_supabase(caminho, metodo="GET", dados=None, prefer=None):
    try:
        configuracao = st.secrets["supabase"]
        url_base = str(configuracao["url"]).rstrip("/")
        chave = str(configuracao["secret_key"])
    except (FileNotFoundError, KeyError):
        return None

    cabecalhos = {
        "apikey": chave,
        "Content-Type": "application/json",
    }

    # As chaves antigas do tipo service_role são tokens JWT.
    if chave.startswith("eyJ"):
        cabecalhos["Authorization"] = f"Bearer {chave}"

    if prefer:
        cabecalhos["Prefer"] = prefer

    corpo = None
    if dados is not None:
        corpo = json.dumps(dados).encode("utf-8")

    requisicao = urllib.request.Request(
        f"{url_base}/rest/v1/{caminho}",
        data=corpo,
        headers=cabecalhos,
        method=metodo,
    )

    try:
        with urllib.request.urlopen(requisicao, timeout=20) as resposta:
            conteudo = resposta.read().decode("utf-8")
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None

    if not conteudo:
        return True

    try:
        return json.loads(conteudo)
    except json.JSONDecodeError:
        return None


def sincronizar_codigos_leitores():
    if st.session_state.get("codigos_supabase_sincronizados"):
        return True

    try:
        codigos = st.secrets["codigos_leitores"]
    except (FileNotFoundError, KeyError):
        return False

    for identificador, codigo_hash in codigos.items():
        caminho = (
            "codigos_leitores?on_conflict=identificador"
        )
        resultado = requisicao_supabase(
            caminho,
            metodo="POST",
            dados={
                "identificador": str(identificador),
                "codigo_hash": str(codigo_hash),
                "ativo": True,
            },
            prefer="resolution=merge-duplicates,return=minimal",
        )
        if resultado is None:
            return False

    st.session_state.codigos_supabase_sincronizados = True
    return True


def localizar_leitor_google(google_sub, email_google):
    sub_codificado = urllib.parse.quote(str(google_sub), safe="")
    caminho = (
        "leitores?select=id,email,nome"
        f"&google_sub=eq.{sub_codificado}&limit=1"
    )
    resultado = requisicao_supabase(caminho)

    if isinstance(resultado, list) and resultado:
        return resultado[0]

    email_codificado = urllib.parse.quote(str(email_google), safe="")
    caminho = (
        "leitores?select=id,email,nome"
        f"&email=eq.{email_codificado}&limit=1"
    )
    resultado = requisicao_supabase(caminho)

    if isinstance(resultado, list) and resultado:
        return resultado[0]

    return None


def ativar_codigo_no_supabase(codigo, google_sub, email, nome):
    codigo_normalizado = codigo.strip().upper()
    codigo_hash = hashlib.sha256(
        codigo_normalizado.encode("utf-8")
    ).hexdigest()

    return requisicao_supabase(
        "rpc/ativar_codigo_leitor",
        metodo="POST",
        dados={
            "p_codigo_hash": codigo_hash,
            "p_google_sub": str(google_sub),
            "p_email": str(email),
            "p_nome": str(nome),
        },
    )


def registrar_evento_acesso(tipo, leitor_id=None):
    dados = {"tipo": tipo}
    if leitor_id is not None:
        dados["leitor_id"] = int(leitor_id)

    return requisicao_supabase(
        "eventos_acesso",
        metodo="POST",
        dados=dados,
        prefer="return=minimal",
    )


@st.cache_data
def carregar_conteudos_por_cargo():
    caminho = BASE / "base_conteudo_visitante.json"
    with caminho.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)

    conteudos = dados.get("conteudos", {})
    return [
        {"id": identificador, **conteudo}
        for identificador, conteudo in conteudos.items()
        if identificador.startswith("conteudo:cargo:")
        or identificador.startswith("conteudo:interludio:")
    ]


def abrir_cargo_leitor(identificador):
    st.session_state.leitor_cargo_atual = identificador
    st.session_state.leitor_tela = "cargo_detalhe"


def lista_cargos_leitor():
    st.session_state.leitor_tela = "cargos"
    st.session_state.pop("leitor_cargo_atual", None)


def inicio_leitor():
    st.session_state.leitor_tela = "inicio"
    st.session_state.pop("leitor_cargo_atual", None)


def iniciar_apoio_afastamento(etapa="entrada"):
    st.session_state.leitor_tela = "apoio_afastamento"
    st.session_state.apoio_etapa = etapa
    st.session_state.apoio_historico = []


def apoio_ir_para(etapa):
    atual = st.session_state.get("apoio_etapa", "entrada")
    st.session_state.setdefault("apoio_historico", []).append(atual)
    st.session_state.apoio_etapa = etapa


def apoio_voltar():
    historico = st.session_state.setdefault("apoio_historico", [])
    if historico:
        st.session_state.apoio_etapa = historico.pop()
    else:
        st.session_state.leitor_tela = "apoio_inicio"


def limpar_jornada_apoio():
    prefixos = ("apoio_", "widget_apoio_")
    preservadas = {"apoio_etapa", "apoio_historico"}
    for chave in list(st.session_state.keys()):
        if chave in preservadas:
            continue
        if chave.startswith(prefixos):
            st.session_state.pop(chave, None)


def criar_lembrete_ics(data_contato):
    inicio = datetime.combine(data_contato, horario(hour=9))
    fim = inicio + timedelta(minutes=30)
    agora = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    identificador = hashlib.sha256(
        f"{st.session_state.get('leitor_id', '')}-{data_contato}".encode()
    ).hexdigest()[:20]
    return "\r\n".join(
        [
            "BEGIN:VCALENDAR",
            "VERSION:2.0",
            "PRODID:-//A Loja que Permanece//Apoio Fraterno//PT-BR",
            "CALSCALE:GREGORIAN",
            "BEGIN:VEVENT",
            f"UID:{identificador}@alojaquepermanece",
            f"DTSTAMP:{agora}",
            f"DTSTART:{inicio.strftime('%Y%m%dT%H%M%S')}",
            f"DTEND:{fim.strftime('%Y%m%dT%H%M%S')}",
            "SUMMARY:Realizar acompanhamento fraterno",
            "DESCRIPTION:Retomar o contato fraterno previamente planejado.",
            "BEGIN:VALARM",
            "TRIGGER:-PT24H",
            "ACTION:DISPLAY",
            "DESCRIPTION:Lembrete de acompanhamento fraterno",
            "END:VALARM",
            "END:VEVENT",
            "END:VCALENDAR",
            "",
        ]
    )


if st.session_state.porta_entrada is None:
    coluna_capa, coluna_conteudo = st.columns(
        [0.78, 1.35],
        gap="large",
        vertical_alignment="center",
    )

    with coluna_capa:
        st.image(
            str(CAPA_APP),
            use_container_width=True,
        )

    with coluna_conteudo:
        st.markdown(
            """
            <p class="alq-kicker">Experiência digital da obra</p>
            <p class="alq-title">A Loja que<br>Permanece</p>
            <p class="alq-subtitle">
                Governança e gestão maçônica para gerações
            </p>
            <div class="alq-rule"></div>
            <p class="alq-intro">
                Explore os princípios da obra ou acesse o ambiente
                reservado a quem já possui o livro.
            </p>
            """,
            unsafe_allow_html=True,
        )

        coluna_visitante, coluna_leitor = st.columns(2)

        with coluna_visitante:
            if st.button(
                "CONHECER A OBRA",
                type="primary",
                use_container_width=True,
            ):
                if not st.session_state.get("visita_registrada"):
                    if registrar_evento_acesso("visita") is not None:
                        st.session_state.visita_registrada = True
                escolher_porta_entrada("Visitante")
                st.rerun()

        with coluna_leitor:
            if st.button(
                "ACESSO DO LEITOR",
                use_container_width=True,
            ):
                escolher_porta_entrada("Leitor")
                st.rerun()

        st.caption("Conteúdo orientado pela obra de Mauro Arantes")

    st.stop()


perfil_acesso = st.session_state.porta_entrada

st.title("A Loja que Permanece")
st.caption("Experiência digital da obra")

if perfil_acesso == "Leitor":
    if not st.user.is_logged_in:
        st.subheader("Acesso do Leitor")
        st.markdown(
            "Entre com sua conta Google para continuar."
        )

        if st.button(
            "ENTRAR COM GOOGLE",
            type="primary",
            use_container_width=True,
        ):
            st.login()

        if st.button(
            "VOLTAR",
            key="leitor_voltar_antes_google",
            use_container_width=True,
        ):
            voltar_para_entrada()
            st.rerun()

        st.stop()

    google_sub = str(st.user.get("sub", "")).strip()
    email_google = str(st.user.get("email", "")).strip()
    nome_google = str(st.user.get("name", "")).strip()

    if not google_sub:
        st.error(
            "Não foi possível identificar a conta Google. "
            "Saia e tente entrar novamente."
        )
        if st.button("SAIR DA CONTA GOOGLE", use_container_width=True):
            st.logout()
        st.stop()

    if not sincronizar_codigos_leitores():
        st.error(
            "Não foi possível conectar ao cadastro de leitores. "
            "Tente novamente em alguns instantes."
        )
        st.stop()

    if not st.session_state.get("leitor_autenticado"):
        leitor_cadastrado = localizar_leitor_google(
            google_sub,
            email_google,
        )
        if leitor_cadastrado:
            st.session_state.leitor_autenticado = True
            st.session_state.leitor_id = leitor_cadastrado["id"]

            if not st.session_state.get("login_leitor_registrado"):
                registrar_evento_acesso(
                    "login",
                    leitor_cadastrado["id"],
                )
                st.session_state.login_leitor_registrado = True
            st.rerun()

    if not st.session_state.get("leitor_autenticado"):
        st.subheader("Acesso do Leitor")
        st.success(f"Conta Google reconhecida: {email_google}")
        st.markdown(
            "Digite o código individual recebido com o seu acesso à obra."
        )

        with st.form("formulario_acesso_leitor"):
            codigo_leitor = st.text_input(
                "Código de acesso",
                type="password",
                placeholder="Digite seu código",
            )
            entrar_leitor = st.form_submit_button(
                "ENTRAR",
                use_container_width=True,
            )

        if entrar_leitor:
            if not codigo_leitor.strip():
                st.warning("Digite seu código de acesso.")
            else:
                resultado = ativar_codigo_no_supabase(
                    codigo_leitor,
                    google_sub,
                    email_google,
                    nome_google,
                )
                status = (
                    resultado.get("status")
                    if isinstance(resultado, dict)
                    else None
                )

                if status in {"ativado", "autorizado"}:
                    leitor_id = resultado.get("leitor_id")
                    st.session_state.leitor_autenticado = True
                    st.session_state.leitor_id = leitor_id
                    registrar_evento_acesso("login", leitor_id)
                    st.session_state.login_leitor_registrado = True
                    st.rerun()
                elif status == "codigo_ja_utilizado":
                    st.error(
                        "Este código já está vinculado a outra conta Google."
                    )
                elif status == "conta_ja_vinculada":
                    st.error(
                        "Esta conta Google já possui um código vinculado."
                    )
                elif status == "codigo_invalido":
                    st.error("Código de acesso inválido.")
                else:
                    st.error(
                        "Não foi possível validar o código agora. "
                        "Tente novamente em alguns instantes."
                    )

        if st.button(
            "SAIR DA CONTA GOOGLE",
            key="leitor_voltar_entrada",
            use_container_width=True,
        ):
            st.session_state.pop("leitor_autenticado", None)
            st.session_state.pop("leitor_id", None)
            st.session_state.pop("login_leitor_registrado", None)
            st.logout()

        st.stop()

    st.subheader("Ambiente do Leitor")
    st.success("Código reconhecido. Acesso do leitor autorizado.")

    nome_identificacao = nome_google or email_google
    nome_identificacao = html.escape(nome_identificacao)

    st.markdown(
        """
        <style>
        [data-testid="stMarkdownContainer"] {
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
        }

        [data-testid="stCodeBlock"] button,
        [data-testid="stElementToolbar"] {
            display: none !important;
        }

        .alq-identificacao-leitor {
            border: 1px solid rgba(183, 142, 64, 0.65);
            color: #d6c59d;
            font-size: 0.82rem;
            letter-spacing: 0.04em;
            margin: 0.75rem 0 1rem;
            padding: 0.65rem 0.8rem;
            text-align: center;
        }

        .alq-copyright-leitor {
            color: #8f8776;
            font-size: 0.74rem;
            line-height: 1.45;
            margin: 0.55rem 0 1rem;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="alq-identificacao-leitor">
            Acesso pessoal de {nome_identificacao}<br>
            Conteúdo de uso pessoal e intransferível.
        </div>
        <div class="alq-copyright-leitor">
            © 2026 Mauro Arantes. Todos os direitos reservados.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.get("acesso_leitor_registrado"):
        leitor_id = st.session_state.get("leitor_id")
        if leitor_id is not None:
            registrar_evento_acesso("acesso_leitor", leitor_id)
            st.session_state.acesso_leitor_registrado = True

    if st.button(
        "SAIR DO ACESSO",
        key="leitor_sair_acesso",
        use_container_width=True,
    ):
        st.session_state.pop("leitor_autenticado", None)
        st.session_state.pop("leitor_id", None)
        st.session_state.pop("login_leitor_registrado", None)
        st.session_state.pop("acesso_leitor_registrado", None)
        limpar_jornada_apoio()
        st.session_state.pop("leitor_tela", None)
        st.logout()

    if "leitor_tela" not in st.session_state:
        st.session_state.leitor_tela = "inicio"

    try:
        conteudos_cargos = carregar_conteudos_por_cargo()
    except (OSError, json.JSONDecodeError, TypeError) as erro:
        st.error(
            "Não foi possível carregar a navegação por cargos: "
            f"{erro}"
        )
        st.stop()

    tela_leitor = st.session_state.leitor_tela

    if tela_leitor == "inicio":
        st.markdown("### Como deseja explorar a obra?")
        st.markdown(
            "Escolha um caminho para consultar a obra ou receber uma "
            "orientação prática."
        )

        if st.button(
            "EXPLORAR POR CARGO",
            type="primary",
            use_container_width=True,
        ):
            lista_cargos_leitor()
            st.rerun()

        st.button(
            "ENFRENTAR UMA NECESSIDADE DA LOJA — EM PREPARAÇÃO",
            disabled=True,
            use_container_width=True,
        )
        st.button(
            "CONSULTAR INSTRUMENTOS PRÁTICOS — EM PREPARAÇÃO",
            disabled=True,
            use_container_width=True,
        )
        st.button(
            "COMPREENDER A PERMANÊNCIA INSTITUCIONAL — EM PREPARAÇÃO",
            disabled=True,
            use_container_width=True,
        )
        if st.button(
            "APOIO FRATERNO",
            use_container_width=True,
        ):
            st.session_state.leitor_tela = "apoio_inicio"
            st.rerun()

    elif tela_leitor == "cargos":
        st.markdown("### Explorar por cargo")
        st.markdown(
            "Selecione uma função para consultar sua missão, temas centrais, "
            "instrumentos, risco de gestão e uma ação inicial."
        )

        for indice in range(0, len(conteudos_cargos), 2):
            colunas = st.columns(2)
            for deslocamento, coluna in enumerate(colunas):
                posicao = indice + deslocamento
                if posicao >= len(conteudos_cargos):
                    continue
                conteudo = conteudos_cargos[posicao]
                rotulo = (
                    f"{conteudo.get('titulo', 'Cargo')} · "
                    f"{conteudo.get('referencia', '')}"
                )
                with coluna:
                    if st.button(
                        rotulo,
                        key=f"leitor_cargo_{conteudo['id']}",
                        use_container_width=True,
                    ):
                        abrir_cargo_leitor(conteudo["id"])
                        st.rerun()

    elif tela_leitor == "cargo_detalhe":
        identificador = st.session_state.get("leitor_cargo_atual")
        conteudo = next(
            (
                item
                for item in conteudos_cargos
                if item["id"] == identificador
            ),
            None,
        )

        if conteudo is None:
            lista_cargos_leitor()
            st.rerun()

        st.markdown(f"### {conteudo.get('titulo', 'Cargo')}")
        st.caption(conteudo.get("referencia", ""))

        st.markdown("#### Missão essencial")
        st.markdown(conteudo.get("missao", "Não informado."))

        st.markdown("#### Temas centrais")
        for tema in conteudo.get("temas", []):
            st.markdown(f"- {tema}")

        st.markdown("#### Instrumentos abordados")
        st.markdown(conteudo.get("instrumentos", "Não informado."))

        st.markdown("#### Risco de gestão")
        st.warning(conteudo.get("risco", "Não informado."))

        st.markdown("#### Ação inicial sugerida")
        st.info(conteudo.get("microacao", "Não informado."))

        st.markdown("#### Para aprofundar na obra")
        st.markdown(conteudo.get("aprofundamento", "Não informado."))

    elif tela_leitor == "apoio_inicio":
        st.markdown("### Apoio Fraterno")
        st.markdown(
            "Jornadas orientadas para transformar sinais de dificuldade em "
            "escuta, cuidado e encaminhamento responsável."
        )
        st.info(
            "As informações fornecidas durante a jornada permanecem apenas "
            "nesta sessão e não são enviadas ao cadastro do aplicativo."
        )
        if st.button(
            "UM IRMÃO ESTÁ SE AFASTANDO",
            type="primary",
            use_container_width=True,
        ):
            limpar_jornada_apoio()
            iniciar_apoio_afastamento()
            st.rerun()

    elif tela_leitor == "apoio_afastamento":
        etapa_apoio = st.session_state.get("apoio_etapa", "entrada")

        st.markdown("### Apoio Fraterno")
        st.caption("Jornada 1 — Um Irmão está se afastando")

        if etapa_apoio not in ("emergencia", "concluida"):
            with st.container(key="apoio_barra_emergencia"):
                st.error(
                    "Se houver preocupação com a segurança deste Irmão, "
                    "acesse imediatamente o protocolo de risco."
                )
                if st.button(
                    "HÁ PREOCUPAÇÃO COM A SEGURANÇA DESTE IRMÃO",
                    key=f"apoio_emergencia_{etapa_apoio}",
                    use_container_width=True,
                ):
                    st.session_state.apoio_etapa_anterior = etapa_apoio
                    apoio_ir_para("emergencia")
                    st.rerun()

        if etapa_apoio == "entrada":
            st.markdown("#### Em que momento você está?")
            st.markdown(
                "A jornada pode ser utilizada antes ou depois do contato, "
                "sem exigir que o aplicativo permaneça aberto."
            )
            if st.button(
                "QUERO PREPARAR O CONTATO",
                type="primary",
                use_container_width=True,
            ):
                apoio_ir_para("vinculo")
                st.rerun()
            if st.button(
                "JÁ CONVERSEI COM O IRMÃO",
                use_container_width=True,
            ):
                apoio_ir_para("necessidades")
                st.rerun()

        elif etapa_apoio == "vinculo":
            st.markdown("#### Qual é o seu vínculo com esse Irmão?")
            opcoes_vinculo = [
                "Sou o padrinho",
                "Sou Vigilante",
                "Sou Hospitaleiro",
                "Fui designado pela Loja",
                "Somos próximos, sem cargo formal",
                "Outro vínculo",
            ]
            with st.form("apoio_form_vinculo"):
                vinculo = st.radio(
                    "Escolha uma opção",
                    opcoes_vinculo,
                    index=None,
                    key="widget_apoio_vinculo",
                )
                avancar = st.form_submit_button(
                    "AVANÇAR",
                    use_container_width=True,
                )
            if avancar:
                if vinculo is None:
                    st.warning("Escolha o vínculo para continuar.")
                else:
                    st.session_state.apoio_vinculo = vinculo
                    apoio_ir_para("sinais")
                    st.rerun()

        elif etapa_apoio == "sinais":
            st.markdown("#### O que chamou sua atenção?")
            st.caption("Você pode marcar mais de uma opção.")
            opcoes_sinais = [
                "Ausências recentes ou progressivas",
                "Redução da participação e do entusiasmo",
                "Afastamento dos momentos de convivência",
                "Respostas mais curtas ou comportamento diferente",
                "Manifestação direta de insatisfação",
            ]
            with st.form("apoio_form_sinais"):
                sinais = [
                    opcao
                    for indice, opcao in enumerate(opcoes_sinais)
                    if st.checkbox(
                        opcao,
                        key=f"widget_apoio_sinal_{indice}",
                    )
                ]
                outra_mudanca = st.text_input(
                    "Outra mudança percebida (opcional)",
                    key="widget_apoio_outra_mudanca",
                )
                st.caption(
                    "Não informe nomes, diagnósticos ou outros detalhes "
                    "que permitam identificar o Irmão."
                )
                avancar = st.form_submit_button(
                    "AVANÇAR",
                    use_container_width=True,
                )
            if avancar:
                if not sinais and not outra_mudanca.strip():
                    st.warning("Selecione ou descreva ao menos um sinal.")
                else:
                    st.session_state.apoio_sinais = sinais
                    st.session_state.apoio_outra_mudanca = (
                        outra_mudanca.strip()
                    )
                    apoio_ir_para("providencia")
                    st.rerun()

        elif etapa_apoio == "providencia":
            st.markdown("#### Primeira providência")
            st.info(
                "Realize um contato pessoal, discreto e fraterno. O objetivo "
                "inicial é demonstrar que a ausência foi percebida e abrir "
                "espaço para escuta, sem cobrança de frequência, pressão ou "
                "julgamento."
            )
            if st.button(
                "AVANÇAR",
                key="apoio_avancar_providencia",
                use_container_width=True,
            ):
                apoio_ir_para("abertura")
                st.rerun()

        elif etapa_apoio == "abertura":
            st.markdown("#### Como iniciar a conversa")
            vinculo = st.session_state.get("apoio_vinculo", "Outro vínculo")
            aberturas = {
                "Sou o padrinho": [
                    "Cheguei a pensar em você esta semana. Como estão as coisas?",
                    "Meu Irmão, senti sua falta e quis saber como você está.",
                    "Queria conversar com calma e saber como têm sido seus dias.",
                ],
                "Sou Vigilante": [
                    "Reparei sua ausência nas últimas sessões e quis saber se está tudo bem.",
                    "Percebi uma mudança na sua participação e gostaria de ouvi-lo, sem qualquer cobrança.",
                    "Meu Irmão, como você tem se sentido em relação à Loja e às atividades?",
                ],
                "Sou Hospitaleiro": [
                    "Meu Irmão, procurei você para saber como está e se existe alguma forma discreta de apoiá-lo.",
                    "Quis conversar com você com toda reserva. Como estão as coisas?",
                    "Estou à disposição para ouvi-lo e compreender se há alguma dificuldade em que possamos ajudar.",
                ],
                "Fui designado pela Loja": [
                    "Meu Irmão, fui incumbido de procurá-lo, mas este contato não é uma cobrança. Gostaria apenas de saber como você está.",
                    "A Loja percebeu sua ausência e pediu que eu entrasse em contato para ouvi-lo, com discrição e respeito.",
                    "Quis abrir este espaço de conversa para saber como estão as coisas e se você deseja algum apoio.",
                ],
            }
            sugestoes = aberturas.get(
                vinculo,
                [
                    "Meu Irmão, percebi sua ausência e quis saber como você está. Não é cobrança. Sua presença e seu bem-estar importam para nós.",
                    "Pensei em você e resolvi escrever. Como estão as coisas?",
                    "Percebi que você está mais distante e quis abrir um espaço para ouvi-lo, se desejar conversar.",
                ],
            )
            with st.form("apoio_form_abertura"):
                escolha = st.radio(
                    "Escolha uma abertura ou escreva a sua",
                    sugestoes + ["Escrever a minha própria"],
                    key="widget_apoio_abertura_escolha",
                )
                abertura_propria = ""
                if escolha == "Escrever a minha própria":
                    abertura_propria = st.text_area(
                        "Sua abertura",
                        key="widget_apoio_abertura_propria",
                    )
                avancar = st.form_submit_button(
                    "USAR ESTA ABERTURA",
                    use_container_width=True,
                )
            if avancar:
                abertura_final = (
                    abertura_propria.strip()
                    if escolha == "Escrever a minha própria"
                    else escolha
                )
                if not abertura_final:
                    st.warning("Escreva uma abertura para continuar.")
                else:
                    st.session_state.apoio_abertura = abertura_final
                    apoio_ir_para("conversa")
                    st.rerun()

        elif etapa_apoio == "conversa":
            st.markdown("#### Durante a conversa")
            st.markdown(
                "- Escute antes de oferecer soluções.\n"
                "- Pergunte se existe alguma dificuldade para participar.\n"
                "- Respeite o que o Irmão não desejar compartilhar.\n"
                "- Evite promessas que dependam da Loja.\n"
                "- Solicite autorização antes de encaminhar a situação."
            )
            col_agora, col_depois = st.columns(2)
            with col_agora:
                if st.button(
                    "A CONVERSA ACONTECEU",
                    type="primary",
                    use_container_width=True,
                ):
                    apoio_ir_para("necessidades")
                    st.rerun()
            with col_depois:
                if st.button(
                    "VOU CONVERSAR E VOLTAR DEPOIS",
                    use_container_width=True,
                ):
                    st.session_state.leitor_tela = "apoio_inicio"
                    st.rerun()

        elif etapa_apoio == "necessidades":
            st.markdown("#### O que você percebeu na conversa?")
            st.caption("Você pode marcar mais de uma opção.")
            opcoes_necessidades = [
                "Dificuldade financeira",
                "Dificuldade formativa ou ritual",
                "Conflito ou insatisfação institucional",
                "Problema de horário, transporte ou organização",
                "Sofrimento emocional ou questão de saúde",
                "Motivo não informado",
            ]
            with st.form("apoio_form_necessidades"):
                necessidades = [
                    opcao
                    for indice, opcao in enumerate(opcoes_necessidades)
                    if st.checkbox(
                        opcao,
                        key=f"widget_apoio_necessidade_{indice}",
                    )
                ]
                avancar = st.form_submit_button(
                    "VER ENCAMINHAMENTOS",
                    use_container_width=True,
                )
            if avancar:
                if not necessidades:
                    st.warning("Selecione ao menos uma opção.")
                else:
                    st.session_state.apoio_necessidades = necessidades
                    apoio_ir_para("encaminhamento")
                    st.rerun()

            if st.button(
                "HÁ SINAL DE RISCO IMEDIATO OU DE AUTOAGRESSÃO",
                key="apoio_risco_na_necessidade",
                use_container_width=True,
            ):
                st.session_state.apoio_etapa_anterior = "necessidades"
                apoio_ir_para("emergencia")
                st.rerun()

        elif etapa_apoio == "emergencia":
            st.error("PROTOCOLO DE RISCO IMEDIATO")
            st.markdown(
                "**Isso não é um encaminhamento comum. Não tente resolver "
                "sozinho e não espere pelo acompanhamento habitual.**"
            )
            st.markdown(
                "Se o Irmão está tentando se ferir, possui um plano imediato, "
                "dispõe dos meios para executá-lo ou apresenta uma crise aguda, "
                "acione o **SAMU pelo número 192** ou conduza-o a um serviço de "
                "urgência, se isso puder ser feito com segurança."
            )
            st.markdown(
                "Permaneça com ele, quando for seguro, ou acione imediatamente "
                "uma pessoa de confiança que possa estar presente. O **CVV — "
                "188** oferece apoio emocional gratuito durante 24 horas, mas "
                "não substitui o atendimento de emergência."
            )
            st.markdown(
                "Comunique alguém capaz de participar concretamente do cuidado. "
                "O Venerável Mestre e o Hospitaleiro também podem organizar o "
                "apoio fraterno, sem substituir os profissionais de saúde."
            )
            st.warning(
                "Este aplicativo não realiza atendimento de emergência e não "
                "monitora as ações indicadas nesta tela."
            )
            col_samu, col_cvv = st.columns(2)
            with col_samu:
                st.markdown(
                    '<a href="tel:192" style="display:block;padding:0.65rem;'
                    'border:1px solid #b78e40;text-align:center;'
                    'text-decoration:none;">LIGAR PARA O SAMU — 192</a>',
                    unsafe_allow_html=True,
                )
            with col_cvv:
                st.markdown(
                    '<a href="tel:188" style="display:block;padding:0.65rem;'
                    'border:1px solid #b78e40;text-align:center;'
                    'text-decoration:none;">LIGAR PARA O CVV — 188</a>',
                    unsafe_allow_html=True,
                )
            st.link_button(
                "ABRIR ATENDIMENTO DO CVV",
                "https://cvv.org.br/quero-conversar/",
                use_container_width=True,
            )
            aviso_confirmado = st.checkbox(
                "Já avisei o Venerável Mestre ou o Hospitaleiro.",
                key="widget_apoio_aviso_emergencia",
            )
            st.caption(
                "Essa comunicação não substitui o acionamento do SAMU ou de "
                "um serviço de urgência quando houver risco imediato."
            )
            if st.button(
                "VOLTAR À JORNADA",
                key="apoio_sair_emergencia",
                use_container_width=True,
                disabled=not aviso_confirmado,
            ):
                apoio_voltar()
                st.rerun()

        elif etapa_apoio == "encaminhamento":
            st.markdown("#### Encaminhamentos sugeridos")
            encaminhamentos = {
                "Dificuldade financeira": (
                    "Encaminhamento reservado ao Hospitaleiro."
                ),
                "Dificuldade formativa ou ritual": (
                    "Encaminhamento ao Vigilante responsável."
                ),
                "Conflito ou insatisfação institucional": (
                    "Encaminhamento discreto ao Venerável Mestre."
                ),
                "Problema de horário, transporte ou organização": (
                    "Avaliar apoio prático possível, sem criar promessas que "
                    "dependam da Loja."
                ),
                "Sofrimento emocional ou questão de saúde": (
                    "Acolher e recomendar ajuda profissional. Se surgir "
                    "preocupação com a segurança, utilizar o protocolo de risco."
                ),
                "Motivo não informado": (
                    "Manter contato respeitoso, sem insistência, deixando a "
                    "porta aberta."
                ),
            }
            for necessidade in st.session_state.get(
                "apoio_necessidades", []
            ):
                st.markdown(
                    f"**{necessidade}**  \n"
                    f"{encaminhamentos[necessidade]}"
                )
            if st.button(
                "PLANEJAR ACOMPANHAMENTO",
                type="primary",
                use_container_width=True,
            ):
                apoio_ir_para("acompanhamento")
                st.rerun()

        elif etapa_apoio == "acompanhamento":
            st.markdown("#### Acompanhamento")
            with st.form("apoio_form_acompanhamento"):
                referencia = st.text_input(
                    "Quem permanecerá como referência?",
                    key="widget_apoio_referencia",
                    placeholder="Informe apenas para seu uso nesta sessão",
                )
                data_contato = st.date_input(
                    "Quando será o próximo contato?",
                    value=None,
                    min_value=date.today(),
                    format="DD/MM/YYYY",
                    key="widget_apoio_data",
                )
                lembrete = st.radio(
                    "Preparar lembrete para o calendário?",
                    ["Sim", "Não"],
                    horizontal=True,
                    key="widget_apoio_lembrete",
                )
                encaminhamento_ocorreu = st.radio(
                    "O encaminhamento ocorreu?",
                    ["Sim", "Ainda não", "Não se aplica"],
                    horizontal=True,
                    key="widget_apoio_encaminhamento_ocorreu",
                )
                avancar = st.form_submit_button(
                    "CONFIRMAR ACOMPANHAMENTO",
                    use_container_width=True,
                )
            st.caption(
                "Não use este campo para registrar o nome ou informações de "
                "saúde do Irmão acompanhado. Nenhum dado é enviado ao cadastro."
            )
            st.markdown(
                "- Preserve a confidencialidade.\n"
                "- Observe a reintegração sem expor o Irmão."
            )
            if avancar:
                if data_contato is None:
                    st.warning("Defina a data do próximo contato.")
                else:
                    st.session_state.apoio_referencia = referencia.strip()
                    st.session_state.apoio_data = data_contato
                    st.session_state.apoio_lembrete = lembrete
                    st.session_state.apoio_encaminhamento_ocorreu = (
                        encaminhamento_ocorreu
                    )
                    apoio_ir_para("evitar")
                    st.rerun()

        elif etapa_apoio == "evitar":
            st.markdown("#### O que evitar")
            st.markdown(
                "- Transformar o contato em cobrança.\n"
                "- Comentar o caso em grupos.\n"
                "- Interpretar ausência como desinteresse.\n"
                "- Tentar resolver tudo sozinho.\n"
                "- Oferecer diagnóstico psicológico.\n"
                "- Abandonar o acompanhamento depois da primeira conversa.\n"
                "- Insistir além de uma segunda tentativa respeitosa. Se "
                "permanecer uma preocupação concreta com a segurança, utilize "
                "o protocolo de risco imediato."
            )
            st.info(
                "Se não houver resposta, registre apenas para si que a tentativa "
                "foi feita, sem expor o Irmão, e mantenha a porta aberta."
            )
            if st.session_state.get("apoio_lembrete") == "Sim":
                data_contato = st.session_state.get("apoio_data")
                if data_contato is not None:
                    st.download_button(
                        "ADICIONAR ACOMPANHAMENTO AO CALENDÁRIO",
                        data=criar_lembrete_ics(data_contato),
                        file_name="acompanhamento_fraterno.ics",
                        mime="text/calendar",
                        use_container_width=True,
                    )
                    st.caption(
                        "Para usar no celular, baixe o arquivo e envie-o para "
                        "você mesmo pelo WhatsApp. Abra o arquivo recebido e "
                        "escolha o aplicativo de calendário. Se ele não abrir "
                        "diretamente, salve-o no aparelho e abra-o pelo "
                        "gerenciador de arquivos. O lembrete será programado "
                        "para 24 horas antes."
                    )
            if st.button(
                "CONCLUIR JORNADA",
                type="primary",
                use_container_width=True,
            ):
                apoio_ir_para("concluida")
                st.rerun()

        elif etapa_apoio == "concluida":
            st.success("Jornada concluída.")
            st.markdown(
                "### A intervenção fraterna começa quando a Loja transforma "
                "ausência em escuta."
            )
            if st.button(
                "VOLTAR AO INÍCIO",
                type="primary",
                use_container_width=True,
            ):
                limpar_jornada_apoio()
                inicio_leitor()
                st.rerun()

    st.divider()

    if tela_leitor == "inicio":
        if st.button(
            "VOLTAR À ENTRADA",
            key="leitor_voltar_inicio_publico",
            use_container_width=True,
        ):
            st.session_state.porta_entrada = None
            st.rerun()
    elif (
        tela_leitor == "apoio_afastamento"
        and etapa_apoio not in ("concluida", "emergencia")
    ):
        col_inicio, col_voltar = st.columns(2)
        with col_inicio:
            if st.button(
                "INÍCIO",
                key="apoio_navegacao_inicio",
                use_container_width=True,
            ):
                limpar_jornada_apoio()
                inicio_leitor()
                st.rerun()
        with col_voltar:
            if st.button(
                "VOLTAR",
                key=f"apoio_navegacao_voltar_{etapa_apoio}",
                use_container_width=True,
            ):
                apoio_voltar()
                st.rerun()
    elif tela_leitor == "apoio_afastamento":
        pass
    else:
        col_inicio, col_voltar = st.columns(2)
        with col_inicio:
            if st.button(
                "INÍCIO",
                key="leitor_navegacao_inicio",
                use_container_width=True,
            ):
                inicio_leitor()
                st.rerun()
        with col_voltar:
            if st.button(
                "VOLTAR",
                key="leitor_navegacao_voltar",
                use_container_width=True,
            ):
                if tela_leitor == "cargo_detalhe":
                    lista_cargos_leitor()
                else:
                    inicio_leitor()
                st.rerun()

    st.markdown(
        """
        <div class="alq-copyright-leitor">
            © 2026 Mauro Arantes. Todos os direitos reservados.<br>
            Conteúdo protegido pela Lei nº 9.610/1998. Acesso pessoal e
            intransferível. É vedada a reprodução, distribuição ou o
            compartilhamento sem autorização, ressalvadas as utilizações
            permitidas pela legislação aplicável.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "tema_atual" not in st.session_state:
    st.session_state.tema_atual = None

if "tema_global_atual" not in st.session_state:
    st.session_state.tema_global_atual = None

if "acao_pendente" not in st.session_state:
    st.session_state.acao_pendente = None


def navegar_visitante(destino):
    st.session_state.pop("visitante_link_palestra", None)
    atual = st.session_state.visitante_no_atual
    if destino == "historico":
        if st.session_state.visitante_historico:
            destino = st.session_state.visitante_historico.pop()
        else:
            destino = "triagem:pergunta"
    else:
        st.session_state.visitante_historico.append(atual)

    st.session_state.visitante_no_atual = destino


def reiniciar_visitante():
    st.session_state.visitante_no_atual = "triagem:pergunta"
    st.session_state.visitante_historico = []
    st.session_state.pop("visitante_link_palestra", None)


if perfil_acesso == "Visitante":
    try:
        nucleo_visitante = criar_nucleo(BASE)
    except Exception as erro:
        st.error(
            "Não foi possível carregar a jornada do Visitante: "
            f"{erro}"
        )
        st.stop()

    if "visitante_no_atual" not in st.session_state:
        st.session_state.visitante_no_atual = "triagem:pergunta"

    if "visitante_historico" not in st.session_state:
        st.session_state.visitante_historico = []

    resposta_visitante = nucleo_visitante.resolver(
        st.session_state.visitante_no_atual
    )

    with st.chat_message("assistant"):
        st.markdown(resposta_visitante["texto"])

    opcoes_visitante = resposta_visitante.get("opcoes", [])
    for numero, opcao in enumerate(opcoes_visitante):
        if st.button(
            opcao["rotulo"],
            key=(
                f"visitante_opcao_"
                f"{st.session_state.visitante_no_atual}_"
                f"{numero}"
            ),
            use_container_width=True,
        ):
            navegar_visitante(opcao["destino"])
            st.rerun()

    if resposta_visitante.get("tipo") == "coleta_orientada":
        with st.form("formulario_interesse_palestra"):
            nome_loja = st.text_input("Nome da Loja *")
            oriente = st.text_input("Oriente *")
            potencia = st.text_input("Potência *")

            col_cidade, col_estado = st.columns([3, 1])
            with col_cidade:
                cidade = st.text_input("Cidade *")
            with col_estado:
                estado = st.text_input("Estado *", max_chars=2)

            periodos = st.text_area(
                "Períodos possíveis",
                placeholder=(
                    "Exemplo: terças-feiras à noite, durante o mês de outubro."
                ),
            )
            contato_convidante = st.text_input(
                "Nome e contato do convidante *"
            )

            enviar_interesse = st.form_submit_button(
                "Preparar mensagem",
                use_container_width=True,
            )

        if enviar_interesse:
            obrigatorios = {
                "Nome da Loja": nome_loja,
                "Oriente": oriente,
                "Potência": potencia,
                "Cidade": cidade,
                "Estado": estado,
                "Contato do convidante": contato_convidante,
            }
            ausentes = [
                campo
                for campo, valor in obrigatorios.items()
                if not valor.strip()
            ]

            if ausentes:
                st.warning(
                    "Preencha os campos obrigatórios: "
                    + ", ".join(ausentes)
                    + "."
                )
            else:
                mensagem_base = resposta_visitante.get("botao", {}).get(
                    "mensagem_preenchida",
                    "Gostaria de receber informações sobre a palestra.",
                )
                linhas_mensagem = [
                    mensagem_base,
                    "",
                    f"Loja: {nome_loja.strip()}",
                    f"Oriente: {oriente.strip()}",
                    f"Potência: {potencia.strip()}",
                    f"Cidade/Estado: {cidade.strip()}/{estado.strip().upper()}",
                    (
                        "Períodos possíveis: "
                        + (periodos.strip() or "A combinar")
                    ),
                    f"Convidante: {contato_convidante.strip()}",
                ]
                contatos = nucleo_visitante.dados_comerciais.get(
                    "canais_de_contato",
                    {},
                )
                numero = "".join(
                    caractere
                    for caractere in str(contatos.get("whatsapp", ""))
                    if caractere.isdigit()
                )
                if numero and not numero.startswith("55"):
                    numero = f"55{numero}"

                if numero:
                    mensagem = "\n".join(linhas_mensagem)
                    st.session_state.visitante_link_palestra = (
                        f"https://wa.me/{numero}?text="
                        f"{urllib.parse.quote(mensagem)}"
                    )
                else:
                    st.error(
                        "O canal de WhatsApp não está disponível na "
                        "base comercial atual."
                    )

        if st.session_state.get("visitante_link_palestra"):
            st.success(
                "Mensagem preparada. Revise o conteúdo no WhatsApp antes "
                "de enviá-lo. O envio não confirma o agendamento."
            )
            st.link_button(
                "Falar sobre a palestra",
                st.session_state.visitante_link_palestra,
                use_container_width=True,
            )

    st.divider()

    if st.session_state.visitante_no_atual == "sessao:encerrar":
        if st.button(
            "INÍCIO",
            key="visitante_reiniciar",
            use_container_width=True,
        ):
            reiniciar_visitante()
            st.rerun()
    else:
        col_inicio, col_voltar, col_encerrar = st.columns(3)

        with col_inicio:
            if st.button(
                "INÍCIO",
                key="visitante_inicio",
                use_container_width=True,
            ):
                reiniciar_visitante()
                st.rerun()

        with col_voltar:
            if st.button(
                "VOLTAR",
                key="visitante_voltar",
                disabled=not st.session_state.visitante_historico,
                use_container_width=True,
            ):
                navegar_visitante("historico")
                st.rerun()

        with col_encerrar:
            if st.button(
                "ENCERRAR",
                key="visitante_encerrar",
                use_container_width=True,
            ):
                st.session_state.visitante_no_atual = "sessao:encerrar"
                st.session_state.visitante_historico = []
                st.session_state.pop("visitante_link_palestra", None)
                st.rerun()

    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Pergunte algo sobre A Loja que Permanece..."
)


if prompt:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)
    
    cobertura = "NÃO DISPONÍVEL"
    consulta_busca = ""
    resultados = []

    st.session_state.pop("tempo_llama", None)
    st.session_state.pop("ollama_total", None)
    st.session_state.pop("ollama_load", None)
    st.session_state.pop("ollama_prompt", None)
    st.session_state.pop("ollama_prompt_tokens", None)
    st.session_state.pop("ollama_geracao", None)
    st.session_state.pop("ollama_geracao_tokens", None)

    inicio_resposta = time.perf_counter()
    
    try:
        with st.spinner("Consultando o livro..."):
            aceite_pendente = (
                eh_aceite_acao_pendente(prompt)
                and st.session_state.acao_pendente is not None
            )
            if not aceite_pendente:
                st.session_state.acao_pendente = None
            consulta_resolvida = prompt

            dados_operacionais_diretos = localizar_dados_operacionais(
                prompt
            )
            dados_comerciais_diretos = localizar_dados_comerciais(
                prompt
            )

            secoes_globais_diretas = localizar_secoes_globais_obra(
                prompt
            )
            tema_global_direto = localizar_tema_global_obra(
                prompt
            )

            if (
                secoes_globais_diretas
                and not tema_global_direto
                and not eh_referencia_contextual(prompt)
            ):
                st.session_state.tema_global_atual = None

            if (
                not tema_global_direto
                and eh_referencia_contextual(prompt)
                and st.session_state.tema_global_atual
            ):
                tema_global_direto = (
                    st.session_state.tema_global_atual
                )

            if dados_operacionais_diretos:
                cobertura = "OPERACIONAL"
                consulta_busca = "Base Operacional"
                resultados = []
                resposta = montar_resposta_dados_operacionais(
                    dados_operacionais_diretos
                )

            elif dados_comerciais_diretos:
                cobertura = "COMERCIAL"
                consulta_busca = "Base Comercial"
                resultados = []
                resposta = montar_resposta_dados_comerciais(
                    dados_comerciais_diretos
                )

            elif (
                secoes_globais_diretas
                or tema_global_direto
            ):
                cobertura = "GLOBAL"
                consulta_busca = "Camada Global da Obra"
                resultados = []
                respostas_globais = []

                if secoes_globais_diretas:
                    resposta_secoes = montar_resposta_secoes_globais_obra(
                        secoes_globais_diretas,
                        prompt,
                    )

                    if resposta_secoes:
                        respostas_globais.append(
                            resposta_secoes
                        )

                if tema_global_direto:
                    st.session_state.tema_global_atual = (
                        tema_global_direto
                    )

                    resposta_tema = montar_resposta_tema_global_verificado(
                        tema_global_direto,
                        prompt,
                    )

                    if resposta_tema:
                        respostas_globais.append(
                            resposta_tema
                        )

                resposta = "\n\n".join(
                    respostas_globais
                )

            else:
                if (
                    (
                        eh_referencia_contextual(prompt)
                        or aceite_pendente
                    )
                    and st.session_state.tema_atual
                ):
                    consulta_resolvida = st.session_state.tema_atual

                cobertura, resultados, consulta_busca = buscar_no_livro(
                    consulta_resolvida
                )

                if not eh_referencia_contextual(prompt):
                    st.session_state.tema_atual = consulta_busca
                    st.session_state.tema_global_atual = None

                resposta = perguntar_ao_llama(
                    prompt,
                    cobertura,
                    resultados,
                    st.session_state.messages,
                    perfil_acesso,
                    consulta_busca,
                )

    except urllib.error.URLError:
        resposta = (
            "Não consegui me conectar ao Ollama. "
            "Verifique se ele está funcionando."
        )

    except Exception as erro:
        resposta = f"Ocorreu um erro: {erro}"
        
    tempo_resposta = time.perf_counter() - inicio_resposta

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": resposta,
        }
    )

    with st.chat_message("assistant"):
        st.markdown(resposta)
        st.caption(
            f"Tempo de resposta: {tempo_resposta:.2f} segundos"
        )

        if cobertura == "GLOBAL":
            titulo_diagnostico = "Ver dados da Camada Global"
        elif cobertura == "COMERCIAL":
            titulo_diagnostico = "Ver dados comerciais"
        elif cobertura == "OPERACIONAL":
            titulo_diagnostico = "Ver dados operacionais"
        else:
            titulo_diagnostico = "Ver diagnóstico da busca"

        with st.expander(
            titulo_diagnostico
        ):
            st.write(
                f"Cobertura: **{cobertura}**"
            )

            if cobertura == "GLOBAL":
                st.write(
                    "Fonte: **Camada Global da Obra**"
                )

            elif cobertura == "COMERCIAL":
                st.write(
                    "Fonte: **Base Comercial**"
                )

            elif cobertura == "OPERACIONAL":
                st.write(
                    "Fonte: **Base Operacional**"
                )

            else:
                st.write(
                    f"Consulta usada na busca: **{consulta_busca}**"
                )

                if "tempo_llama" in st.session_state:
                    st.markdown(
                        f"Tempo Llama: "
                        f"**{st.session_state.tempo_llama:.2f} segundos**"
                    )

                if "ollama_total" in st.session_state:
                    st.markdown(
                        f"Tempo total Ollama: "
                        f"**{st.session_state.ollama_total:.2f} segundos**"
                    )

                    st.markdown(
                        f"Carregamento do modelo: "
                        f"**{st.session_state.ollama_load:.2f} segundos**"
                    )

                    st.markdown(
                        f"Processamento do prompt: "
                        f"**{st.session_state.ollama_prompt:.2f} segundos** "
                        f"({st.session_state.ollama_prompt_tokens} tokens)"
                    )

                    st.markdown(
                        f"Geração da resposta: "
                        f"**{st.session_state.ollama_geracao:.2f} segundos** "
                        f"({st.session_state.ollama_geracao_tokens} tokens)"
                    )

                for numero, resultado in enumerate(
                    resultados,
                    start=1,
                ):
                    st.write(
                        f"**{numero}. "
                        f"{resultado['capitulo']} — "
                        f"página PDF {resultado['pagina_pdf']}**"
                    )

                    st.write(
                        f"Pontuação: "
                        f"{resultado['pontuacao']:.4f} | "
                        f"Semântica: "
                        f"{resultado['semantica']:.4f} | "
                        f"Lexical: "
                        f"{resultado['lexical']:.4f}"
                    )

if perfil_acesso == "Leitor" and st.session_state.get("leitor_autenticado"):
    st.markdown(
        """
        <div class="alq-copyright-leitor">
            © 2026 Mauro Arantes. Todos os direitos reservados.<br>
            Conteúdo protegido pela Lei nº 9.610/1998. Acesso pessoal e
            intransferível. É vedada a reprodução, distribuição ou o
            compartilhamento sem autorização, ressalvadas as utilizações
            permitidas pela legislação aplicável.
        </div>
        """,
        unsafe_allow_html=True,
    )
