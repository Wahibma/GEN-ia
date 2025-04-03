import streamlit as st
import os
from dotenv import load_dotenv

from utils_rag import (
    charger_donnees_pdf,
    preparer_et_indexer_documents,
    construire_chatbot
)

load_dotenv()

def main():
    st.set_page_config(page_title="Chatbot Services Écosystématiques", layout="centered")
    st.title("Chatbot sur les Préférences pour les Services Écosystématiques")

    st.markdown(""" 
    Bienvenue sur le Chatbot des Services Écosystématiques.
    Posez vos questions ci-dessous concernant les services écosystématiques.
    """)

    dossier_pdf = "load_documents_pdf"
    chemin_chroma = "embeddings_pdf2"

    # 1) Chargement et indexation
    if "vecteur_store" not in st.session_state or st.session_state.vecteur_store is None:
        st.write("Chargement et indexation des documents PDF...")
        docs = charger_donnees_pdf(dossier_pdf)
        st.session_state.docs = docs

        vecteur_store = preparer_et_indexer_documents(docs, chemin_chroma)
        st.session_state.vecteur_store = vecteur_store
        st.success("Données indexées avec succès !")

    # 2) Affichage des documents PDF dans la sidebar
    st.sidebar.title("📄 Documents chargés")
    docs = st.session_state.get("docs", [])
    if docs:
        with st.sidebar.expander("Voir la liste des documents"):
            for doc in docs:
                nom = doc.metadata.get("source", "Inconnu")
                st.markdown(f"- **{nom}**")
    else:
        st.sidebar.write("Aucun document chargé.")

    # 3) Paramètres du modèle
    temperature = 0.3
    model_name = "gpt-3.5-turbo"

    # 4) Création du chatbot RAG
    st.session_state.chatbot = construire_chatbot(
        st.session_state.vecteur_store,
        temperature=temperature
    )

    # 5) Historique des messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 6) Saisie utilisateur
    user_input = st.chat_input("Posez votre question ici...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        bot_answer = st.session_state.chatbot(user_input)  # Appel direct à ta fonction
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
        with st.chat_message("assistant"):
            st.write(bot_answer)

    # 7) Historique dans la sidebar
    st.sidebar.title("💬 Historique de la session")
    with st.sidebar.expander("Voir l'historique complet"):
        if not st.session_state.messages:
            st.write("Aucun échange pour le moment.")
        else:
            for i, msg in enumerate(st.session_state.messages, 1):
                role = "Q" if msg["role"] == "user" else "R"
                st.write(f"**{role}{i} :** {msg['content']}")

    if st.sidebar.button("🧹 Effacer l'historique"):
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
