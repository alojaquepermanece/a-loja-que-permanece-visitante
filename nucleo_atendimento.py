"""Núcleo multicanal do atendimento de A Loja que Permanece.

Este módulo não conhece Streamlit, WhatsApp ou Ollama. Ele recebe um
identificador estável da jornada e devolve uma resposta estruturada.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any


BASE = Path(__file__).resolve().parent


class ErroConfiguracaoAtendimento(RuntimeError):
    """Indica inconsistência ou ausência de uma base do atendimento."""


def _carregar_json(caminho: Path) -> dict[str, Any]:
    try:
        with caminho.open("r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
    except FileNotFoundError as erro:
        raise ErroConfiguracaoAtendimento(
            f"Arquivo obrigatório não encontrado: {caminho.name}"
        ) from erro
    except json.JSONDecodeError as erro:
        raise ErroConfiguracaoAtendimento(
            f"JSON inválido em {caminho.name}: linha {erro.lineno}, "
            f"coluna {erro.colno}."
        ) from erro

    if not isinstance(dados, dict):
        raise ErroConfiguracaoAtendimento(
            f"A raiz de {caminho.name} deve ser um objeto JSON."
        )
    return dados


def _obter_campo(dados: Any, caminho: str) -> Any:
    atual = dados
    for parte in caminho.split("."):
        if not isinstance(atual, dict) or parte not in atual:
            raise ErroConfiguracaoAtendimento(
                f"Campo externo não encontrado: {caminho}"
            )
        atual = atual[parte]
    return atual


def _lista_markdown(itens: list[Any]) -> str:
    return "\n".join(f"- {item}" for item in itens)


def _formatar_moeda(valor: Any, moeda: str = "BRL") -> str:
    if isinstance(valor, (int, float)):
        valor_formatado = f"{valor:,.2f}"
        valor_formatado = valor_formatado.replace(
            ",", "MARCADOR"
        ).replace(".", ",").replace("MARCADOR", ".")
    else:
        valor_formatado = str(valor)
    simbolo = "R$" if str(moeda).upper() == "BRL" else str(moeda)
    return f"{simbolo} {valor_formatado}"


class NucleoAtendimento:
    """Resolve a jornada do Visitante sem depender do canal de atendimento."""

    def __init__(self, pasta_base: str | Path | None = None) -> None:
        self.base = Path(pasta_base) if pasta_base else BASE
        self.jornada = _carregar_json(self.base / "jornada_visitante.json")
        self.conteudo_visitante = _carregar_json(
            self.base / "base_conteudo_visitante.json"
        )
        self.palestra = _carregar_json(self.base / "base_palestra.json")
        self._dados_comerciais: dict[str, Any] | None = None
        self._dados_operacionais: dict[str, Any] | None = None
        self.validar()

    @property
    def dados_comerciais(self) -> dict[str, Any]:
        if self._dados_comerciais is None:
            self._dados_comerciais = _carregar_json(
                self.base / "dados_comerciais.json"
            )
        return self._dados_comerciais

    @property
    def dados_operacionais(self) -> dict[str, Any]:
        if self._dados_operacionais is None:
            self._dados_operacionais = _carregar_json(
                self.base / "dados_operacionais.json"
            )
        return self._dados_operacionais

    def validar(self) -> None:
        nos = self.jornada.get("nos")
        conteudos = self.conteudo_visitante.get("conteudos")
        if not isinstance(nos, dict) or not isinstance(conteudos, dict):
            raise ErroConfiguracaoAtendimento(
                "A jornada ou a base pública não possui a estrutura esperada."
            )

        destinos_externos = set(conteudos)
        destinos_comerciais = {
            "comercial:comprar",
            "comercial:preco",
            "comercial:frete",
            "comercial:retirada",
            "comercial:desconto_quantidade",
            "comercial:contatos",
        }
        destinos_especiais = {"historico"}
        destinos_validos = (
            set(nos)
            | destinos_externos
            | destinos_comerciais
            | destinos_especiais
        )

        faltantes: set[str] = set()
        for no in nos.values():
            if not isinstance(no, dict):
                raise ErroConfiguracaoAtendimento(
                    "Todos os nós da jornada devem ser objetos."
                )
            for opcao in no.get("opcoes", []):
                destino = opcao.get("destino")
                if destino not in destinos_validos:
                    faltantes.add(str(destino))

        if faltantes:
            raise ErroConfiguracaoAtendimento(
                "Destinos não resolvidos: " + ", ".join(sorted(faltantes))
            )

    def resposta_inicial(self) -> dict[str, Any]:
        return self.resolver("triagem:pergunta")

    def resolver(
        self,
        identificador: str,
        *,
        destino_voltar: str | None = None,
    ) -> dict[str, Any]:
        if identificador == "nav:inicio":
            return self.resposta_inicial()
        if identificador == "nav:voltar":
            return self.resolver(destino_voltar or "triagem:pergunta")
        if identificador == "nav:encerrar":
            identificador = "sessao:encerrar"

        nos = self.jornada["nos"]
        if identificador in nos:
            return self._resolver_no(identificador, nos[identificador])

        conteudos = self.conteudo_visitante["conteudos"]
        if identificador in conteudos:
            return self._resolver_conteudo(identificador, conteudos[identificador])

        if identificador.startswith("comercial:"):
            return self._resolver_comercial(identificador)

        return self._resposta_texto_livre(identificador)

    def _resolver_no(
        self,
        identificador: str,
        no_original: dict[str, Any],
    ) -> dict[str, Any]:
        no = deepcopy(no_original)
        tipo = no.get("tipo")

        if tipo == "resposta_externa":
            textos: list[str] = []
            if "campo" in no:
                valor = _obter_campo(self.palestra, no["campo"])
                textos.append(self._formatar_valor_externo(valor))
            for campo in no.get("campos", []):
                valor = _obter_campo(self.palestra, campo)
                textos.append(self._formatar_valor_externo(valor))
            no["texto"] = "\n\n".join(texto for texto in textos if texto)
            no["tipo"] = "resposta"

        elif tipo == "coleta_orientada" and no.get("fonte") == "base_palestra.json":
            coleta = _obter_campo(self.palestra, no["campo"])
            condicoes = _obter_campo(self.palestra, "palestra.condicoes")
            no.update(deepcopy(coleta))
            no["tipo"] = "coleta_orientada"
            no["texto"] = self._texto_condicoes_palestra(condicoes, coleta)

        no["id"] = identificador
        return no

    @staticmethod
    def _formatar_valor_externo(valor: Any) -> str:
        if isinstance(valor, str):
            return valor
        if isinstance(valor, list):
            return _lista_markdown(valor)
        if isinstance(valor, dict):
            partes: list[str] = []
            descricao = valor.get("descricao")
            if descricao:
                partes.append(str(descricao))
            segmentos = valor.get("segmentos")
            if isinstance(segmentos, list):
                partes.append(_lista_markdown(segmentos))
            mensagem = valor.get("mensagem")
            if mensagem:
                partes.append(str(mensagem))
            return "\n\n".join(partes)
        return str(valor)

    @staticmethod
    def _texto_condicoes_palestra(
        condicoes: dict[str, Any],
        coleta: dict[str, Any],
    ) -> str:
        partes = [str(condicoes.get("texto_publico", "")).strip()]
        ajuda = condicoes.get("ajuda_de_custo", {})
        if ajuda.get("pode_ser_combinada"):
            partes.append(
                "Dependendo da distância e do esforço logístico envolvido, "
                "poderá ser combinada previamente uma ajuda de custo destinada "
                "apenas a cobrir as despesas de deslocamento."
            )
        aviso = coleta.get("aviso")
        if aviso:
            partes.append(str(aviso))
        return " ".join(parte for parte in partes if parte)

    @staticmethod
    def _resolver_conteudo(
        identificador: str,
        conteudo: dict[str, Any],
    ) -> dict[str, Any]:
        linhas = [
            f"**{conteudo['titulo']} — {conteudo['referencia']}**",
            conteudo["missao"],
            "**Alguns temas abordados:**\n" + _lista_markdown(conteudo["temas"]),
            "**Instrumentos:** " + conteudo["instrumentos"],
            "**Um risco observado:** " + conteudo["risco"],
            "**Para refletir:** " + conteudo["microacao"],
            conteudo["aprofundamento"],
        ]
        return {
            "id": identificador,
            "tipo": "resposta",
            "texto": "\n\n".join(linhas),
            "fonte": "base_conteudo_visitante.json",
        }

    def _resolver_comercial(self, identificador: str) -> dict[str, Any]:
        dados = self.dados_comerciais
        texto = ""

        if identificador == "comercial:preco":
            preco = dados.get("preco", {})
            valor = preco.get("valor")
            moeda = preco.get("moeda", "R$")
            if valor is not None:
                texto = f"**Preço:** {_formatar_moeda(valor, moeda)}"
            else:
                texto = "Preço não informado."
            if preco.get("observacao"):
                texto += f"\n\n{preco['observacao']}"

        elif identificador == "comercial:frete":
            frete = dados.get("frete", {})
            partes = ["**Frete econômico:**"]
            economico = frete.get("economico", {})
            for chave, valor in economico.items():
                rotulo = chave.replace("_", " ").capitalize()
                partes.append(f"- {rotulo}: {_formatar_moeda(valor)}")
            if frete.get("demais_situacoes"):
                demais = str(frete["demais_situacoes"]).strip().rstrip(".")
                partes.append(f"\n**Demais situações:** {demais.lower()}.")
            texto = "\n".join(partes)

        elif identificador == "comercial:retirada":
            retirada = str(
                dados.get("retirada", "Retirada não informada.")
            ).strip()
            if retirada.casefold() == "não disponível".casefold():
                texto = (
                    "**Retirada**\n\n"
                    "A retirada não está disponível."
                    "\n\nPara consultar a compra pelo site ou as condições "
                    "de frete, clique em **VOLTAR**."
                )
            else:
                texto = f"**Retirada**\n\n{retirada}"

        elif identificador == "comercial:desconto_quantidade":
            descontos = str(
                dados.get(
                    "descontos_por_quantidade",
                    "Descontos por quantidade não informados.",
                )
            ).strip()
            if descontos.casefold() == "não existem condições especiais".casefold():
                texto = (
                    "**Descontos por quantidade**\n\n"
                    "No momento, não existem condições especiais para "
                    "compras em quantidade."
                    "\n\nPara consultar preço, frete ou canais de "
                    "aquisição, clique em **VOLTAR**."
                )
            else:
                texto = f"**Descontos por quantidade**\n\n{descontos}"

        elif identificador == "comercial:comprar":
            canais = dados.get("canais_de_aquisicao", {})
            site = canais.get("site")
            if site:
                texto = (
                    "Você pode adquirir o livro diretamente pelo site: "
                    f"[{site}](http://{site.strip().removeprefix('http://').removeprefix('https://')}/)."
                    "\n\nA escolha da forma de pagamento e a conclusão da "
                    "compra são realizadas no próprio site. O chatbot não "
                    "recebe pagamentos nem dados bancários."
                    "\n\nPara consultar preço, frete, retirada, descontos "
                    "por quantidade ou contatos, clique em **VOLTAR**."
                )
            else:
                texto = "Canal de aquisição não informado."

        elif identificador == "comercial:contatos":
            contatos = dados.get("canais_de_contato", {})
            linhas = ["**Canais de contato:**"]
            email = contatos.get("email")
            instagram = contatos.get("instagram")
            whatsapp = contatos.get("whatsapp")
            if email:
                linhas.append(f"- E-mail: [{email}](mailto:{email})")
            if instagram:
                usuario = str(instagram).strip().lstrip("@")
                linhas.append(
                    f"- Instagram: [@{usuario}]"
                    f"(https://www.instagram.com/{usuario}/)"
                )
            if whatsapp:
                numero = "".join(
                    caractere
                    for caractere in str(whatsapp)
                    if caractere.isdigit()
                )
                numero_internacional = (
                    numero if numero.startswith("55") else f"55{numero}"
                )
                linhas.append(
                    f"- WhatsApp: [{whatsapp}]"
                    f"(https://wa.me/{numero_internacional})"
                )
            texto = "\n".join(linhas)

        else:
            return self._resposta_texto_livre(identificador)

        return {
            "id": identificador,
            "tipo": "resposta",
            "texto": texto,
            "fonte": "dados_comerciais.json",
        }

    def _resposta_texto_livre(self, entrada: str) -> dict[str, Any]:
        regra = self.jornada.get("comportamento_texto_livre", {})
        return {
            "id": "visitante:texto_livre_bloqueado",
            "tipo": "orientacao",
            "texto": regra.get(
                "resposta",
                "Escolha uma das opções disponíveis para continuar.",
            ),
            "entrada_recebida": entrada,
            "destino": "triagem:pergunta",
        }


def criar_nucleo(pasta_base: str | Path | None = None) -> NucleoAtendimento:
    """Fábrica simples para os adaptadores de canal."""

    return NucleoAtendimento(pasta_base=pasta_base)
