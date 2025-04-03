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
    st.set_page_config(page_title="Chatbot Services Écosystémiques", layout="centered")
    st.title("Chatbot sur les Préférences pour les Services Écosystémiques")

    st.markdown("""
    Bienvenue sur le Chatbot des Services Écosystémiques.
    Posez vos questions ci-dessous concernant les services écosystémiques.
    """)

    dossier_pdf = "load_documents_pdf"
    chemin_chroma = "embeddings_pdf2"

    # 1) Chargement et indexation
    if "vecteur_store" not in st.session_state or st.session_state.vecteur_store is None:
        st.write("Chargement et indexation des documents PDF...")
        docs = charger_donnees_pdf(dossier_pdf)
        st.session_state.docs = docs

        vecteur_store = preparer_et_indexer_documents(
            [doc["texte"] for doc in docs],
            chemin_chroma
        )
        st.session_state.vecteur_store = vecteur_store
        st.success("Données indexées avec succès !")

    # 2) LLM + RAG
    st.session_state.chatbot = construire_chatbot(
        st.session_state.vecteur_store,
        temperature=0.3
    )

    # 3) Historique
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 4) Interaction utilisateur
    user_input = st.chat_input("Posez votre question ici...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reponse = st.session_state.chatbot.invoke({"question": user_input})
        bot_answer = reponse["answer"]
        st.session_state.messages.append({"role": "assistant", "content": bot_answer})
        with st.chat_message("assistant"):
            st.write(bot_answer)

    # 5) Barre latérale avec documents et historique
    with st.sidebar:
        st.markdown("### 📄 Documents chargés")
        docs = st.session_state.get("docs", [])
        st.write(f"Nombre de documents chargés : {len(docs)}")

        with st.expander("Voir les documents"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"- **{doc['nom']}**")

        st.markdown("---")
        st.markdown("### 💬 Historique de la session")
        with st.expander("Voir l'historique complet"):
            if len(st.session_state.messages) == 0:
                st.write("Aucun échange pour le moment.")
            else:
                for i, msg in enumerate(st.session_state.messages, start=1):
                    role = "Q" if msg["role"] == "user" else "R"
                    st.write(f"**{role}{i} :** {msg['content']}")

        if st.button("🧹 Effacer l'historique"):
            st.session_state.messages = []
            st.success("Historique effacé.")

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
