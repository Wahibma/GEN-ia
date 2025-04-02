import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

from utils_rag import (
    charger_donnees_pdf,
    preparer_et_indexer_documents,
    construire_chatbot
)

def main():
    st.set_page_config(page_title="Chatbot Services Écosystémiques", layout="centered")
    st.title("Chatbot sur les Préférences pour les Services Écosystémiques")

    st.markdown("""
    Bienvenue sur le Chatbot des Services Écosystémiques.
    Posez vos questions ci-dessous concernant les services écosystémiques.
    """)

    dossier_pdf = "load_documents_pdf"
    chemin_chroma = "embeddings_pdf2"

    # 1) Charger et indexer les documents au démarrage (si pas déjà fait)
    if "vecteur_store" not in st.session_state or st.session_state.vecteur_store is None:
        st.write("Chargement et indexation des documents PDF...")
        docs = charger_donnees_pdf(dossier_pdf)
        st.session_state.docs = docs

        vecteur_store = preparer_et_indexer_documents(docs, chemin_chroma)
        st.session_state.vecteur_store = vecteur_store
        st.success("Données indexées avec succès !")

    # 2) Paramètres fixes
    temperature = 0.3
    model_name = "gpt-3.5-turbo"

    # 3) Construire le chatbot seulement s’il n’existe pas déjà
    if "chatbot" not in st.session_state or st.session_state.chatbot is None:
        st.session_state.chatbot = construire_chatbot(
            st.session_state.vecteur_store,
            temperature=temperature
        )

    # 4) Historique des messages (affichage type “chat”)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # 5) Champ de saisie sous forme de chat
    user_input = st.chat_input("Posez votre question ici...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reponse = st.session_state.chatbot.invoke({"question": user_input})
        bot_answer = reponse["answer"]
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
        with st.chat_message("assistant"):
            st.write(bot_answer)

    # 6) Barre latérale : historique et bouton “Effacer l’historique”
    st.sidebar.title("Historique de la session")
    with st.sidebar.expander("Voir l'historique complet"):
        if len(st.session_state.messages) == 0:
            st.write("Aucun échange pour le moment.")
        else:
            for i, msg in enumerate(st.session_state.messages, start=1):
                if msg["role"] == "user":
                    st.write(f"**Q{i} :** {msg['content']}")
                else:
                    st.write(f"**R{i} :** {msg['content']}")

    if st.sidebar.button("Effacer l'historique"):
        st.session_state.messages = []
        st.sidebar.success("Historique effacé.")

if __name__ == "__main__":
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "vecteur_store" not in st.session_state:
        st.session_state.vecteur_store = None
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    main()
