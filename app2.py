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
    st.set_page_config(page_title="Chatbot services écosystémiques", layout="centered")
    st.title("🌿 Chatbot sur les préférences pour les services écosystémiques")

    st.markdown("""
    Bienvenue sur le Chatbot des services écosystémiques.
    Posez vos questions ci-dessous concernant les services écosystémiques.
    """)

    dossier_pdf = "load_documents_pdf"
    chemin_chroma = "embeddings_pdf2"

    # 1) Charger et indexer les documents PDF
    if "vecteur_store" not in st.session_state or st.session_state.vecteur_store is None:
        st.write("Chargement et indexation des documents PDF...")
        docs = charger_donnees_pdf(dossier_pdf)
        st.session_state.docs = docs

        vecteur_store = preparer_et_indexer_documents(docs, chemin_chroma)
        st.session_state.vecteur_store = vecteur_store
        st.success("✅ Données indexées avec succès !")

    # ✅ 1bis) Affichage des fichiers PDF chargés
    st.sidebar.title("📄 Documents chargés")
    if "docs" in st.session_state and st.session_state.docs:
        st.sidebar.markdown(f"**Nombre de documents chargés :** {len(st.session_state.docs)}")
        with st.sidebar.expander("Voir les documents"):
            for doc in st.session_state.docs:
                nom = doc["nom"] if isinstance(doc, dict) else doc.metadata.get("source", "Inconnu")
                st.markdown(f"- {nom}")
    else:
        st.sidebar.info("Aucun document PDF chargé.")

    # 2) Paramètres
    temperature = 0.3

    # 3) Création du chatbot
    if "chatbot" not in st.session_state or st.session_state.chatbot is None:
        st.session_state.chatbot = construire_chatbot(
            st.session_state.vecteur_store,
            temperature=temperature
        )

    # 4) Historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 5) Saisie utilisateur
    user_input = st.chat_input("Posez votre question ici...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reponse = st.session_state.chatbot.invoke({"question": user_input})
        bot_answer = reponse["answer"]
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
        with st.chat_message("assistant"):
            st.write(bot_answer)

    # 6) Historique de la session
    st.sidebar.title("🗂️ Historique de la session")
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
