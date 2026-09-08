"""Aplicação pública da jornada Visitante de A Loja que Permanece."""

from pathlib import Path
import urllib.parse

import streamlit as st

from nucleo_atendimento import criar_nucleo


BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="A Loja que Permanece",
    page_icon="📖",
    layout="centered",
)

st.title("A Loja que Permanece")
st.caption("Governança e Gestão Maçônica para Gerações")


def navegar(destino: str) -> None:
    st.session_state.pop("link_palestra", None)
    atual = st.session_state.no_atual
    if destino == "historico":
        if st.session_state.historico:
            destino = st.session_state.historico.pop()
        else:
            destino = "triagem:pergunta"
    else:
        st.session_state.historico.append(atual)
    st.session_state.no_atual = destino


def reiniciar() -> None:
    st.session_state.no_atual = "triagem:pergunta"
    st.session_state.historico = []
    st.session_state.pop("link_palestra", None)


try:
    nucleo = criar_nucleo(BASE)
except Exception as erro:
    st.error(f"Não foi possível carregar o atendimento: {erro}")
    st.stop()

if "no_atual" not in st.session_state:
    st.session_state.no_atual = "triagem:pergunta"
if "historico" not in st.session_state:
    st.session_state.historico = []

resposta = nucleo.resolver(st.session_state.no_atual)

with st.chat_message("assistant"):
    st.markdown(resposta["texto"])

for numero, opcao in enumerate(resposta.get("opcoes", [])):
    if st.button(
        opcao["rotulo"],
        key=f"opcao_{st.session_state.no_atual}_{numero}",
        use_container_width=True,
    ):
        navegar(opcao["destino"])
        st.rerun()

if resposta.get("tipo") == "coleta_orientada":
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
            placeholder="Exemplo: terças-feiras à noite, durante o mês de outubro.",
        )
        contato_convidante = st.text_input("Nome e contato do convidante *")
        preparar = st.form_submit_button(
            "Preparar mensagem",
            use_container_width=True,
        )

    if preparar:
        obrigatorios = {
            "Nome da Loja": nome_loja,
            "Oriente": oriente,
            "Potência": potencia,
            "Cidade": cidade,
            "Estado": estado,
            "Contato do convidante": contato_convidante,
        }
        ausentes = [
            campo for campo, valor in obrigatorios.items() if not valor.strip()
        ]

        if ausentes:
            st.warning(
                "Preencha os campos obrigatórios: "
                + ", ".join(ausentes)
                + "."
            )
        else:
            mensagem_base = resposta.get("botao", {}).get(
                "mensagem_preenchida",
                "Gostaria de receber informações sobre a palestra.",
            )
            linhas = [
                mensagem_base,
                "",
                f"Loja: {nome_loja.strip()}",
                f"Oriente: {oriente.strip()}",
                f"Potência: {potencia.strip()}",
                f"Cidade/Estado: {cidade.strip()}/{estado.strip().upper()}",
                "Períodos possíveis: " + (periodos.strip() or "A combinar"),
                f"Convidante: {contato_convidante.strip()}",
            ]
            contatos = nucleo.dados_comerciais.get("canais_de_contato", {})
            numero = "".join(
                caractere
                for caractere in str(contatos.get("whatsapp", ""))
                if caractere.isdigit()
            )
            if numero and not numero.startswith("55"):
                numero = f"55{numero}"

            if numero:
                mensagem = "\n".join(linhas)
                st.session_state.link_palestra = (
                    f"https://wa.me/{numero}?text={urllib.parse.quote(mensagem)}"
                )
            else:
                st.error("O canal de WhatsApp não está disponível.")

    if st.session_state.get("link_palestra"):
        st.success(
            "Mensagem preparada. Revise o conteúdo no WhatsApp antes de "
            "enviá-lo. O envio não confirma o agendamento."
        )
        st.link_button(
            "Falar sobre a palestra",
            st.session_state.link_palestra,
            use_container_width=True,
        )

st.divider()

if st.session_state.no_atual == "sessao:encerrar":
    if st.button("INÍCIO", key="reiniciar", use_container_width=True):
        reiniciar()
        st.rerun()
else:
    col_inicio, col_voltar, col_encerrar = st.columns(3)
    with col_inicio:
        if st.button("INÍCIO", key="inicio", use_container_width=True):
            reiniciar()
            st.rerun()
    with col_voltar:
        if st.button(
            "VOLTAR",
            key="voltar",
            disabled=not st.session_state.historico,
            use_container_width=True,
        ):
            navegar("historico")
            st.rerun()
    with col_encerrar:
        if st.button("ENCERRAR", key="encerrar", use_container_width=True):
            st.session_state.no_atual = "sessao:encerrar"
            st.session_state.historico = []
            st.session_state.pop("link_palestra", None)
            st.rerun()

