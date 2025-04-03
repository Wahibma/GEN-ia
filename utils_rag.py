import os
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()


def charger_donnees_pdf(dossier_path):
    documents = []
    for nom_fichier in os.listdir(dossier_path):
        if nom_fichier.endswith(".pdf"):
            chemin = os.path.join(dossier_path, nom_fichier)
            doc = fitz.open(chemin)
            texte = ""
            for page in doc:
                texte += page.get_text()
            documents.append({
                "nom": nom_fichier,
                "texte": texte
            })
    return documents


def preparer_et_indexer_documents(documents, chemin_index=None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # ✅ Sécurité : on extrait uniquement les textes valides
    textes = [doc["texte"] for doc in documents if isinstance(doc, dict) and "texte" in doc]

    # ✅ Vérification finale
    for t in textes:
        if not isinstance(t, str):
            raise ValueError(f"Un document n'est pas une chaîne de caractères : {t}")

    docs_split = splitter.create_documents(textes)

    embeddings = OpenAIEmbeddings()
    vecteur_store = FAISS.from_documents(docs_split, embeddings)

    if chemin_index:
        vecteur_store.save_local(chemin_index)

    return vecteur_store


def construire_chatbot(vecteur_store, temperature=0.3):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vecteur_store.as_retriever(),
        memory=memory
    )
    return chatbot
