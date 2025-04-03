# utils_rag.py
#
# Ce module regroupe les fonctions utilitaires nécessaires
# pour construire un chatbot utilisant l'approche RAG (Retrieval-Augmented Generation)
# avec OpenAI et LangChain.

import os
import openai
import fitz  # PyMuPDF pour extraire le texte des PDFs
import streamlit as st  # Ajout pour l'affichage des documents dans l'application
from dotenv import load_dotenv

# Import des classes/fonctions nécessaires depuis langchain et extensions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1) Chargement de la clé OpenAI depuis le fichier .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def charger_donnees_pdf(dossier_pdf: str):
    """
    Parcourt tous les fichiers PDF du dossier,
    extrait le texte et crée une liste de Documents LangChain.
    
    :param dossier_pdf: Chemin du répertoire contenant les fichiers PDF
    :return: Liste de Documents extraits des PDFs
    """
    documents = []
    fichiers_disponibles = []
    for nom_fichier in os.listdir(dossier_pdf):
        if nom_fichier.endswith(".pdf"):
            fichiers_disponibles.append(nom_fichier)  # Stocker les noms de fichiers pour affichage
            chemin_fichier = os.path.join(dossier_pdf, nom_fichier)
            try:
                with fitz.open(chemin_fichier) as pdf:
                    texte = "".join(page.get_text("text") for page in pdf)
                    documents.append(
                        Document(
                            page_content=texte,
                            metadata={"source": nom_fichier}
                        )
                    )
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {nom_fichier} : {e}")
    
    # Sauvegarde des fichiers disponibles dans la session Streamlit
    st.session_state["documents_disponibles"] = fichiers_disponibles
    return documents

def preparer_et_indexer_documents(
    documents,
    chemin_chroma="embeddings_pdf",
    collection_name="preferences_services_eco_chatbot"
):
    """
    Prépare la liste de Documents pour l'indexation, en les découpant
    puis en créant/méttant à jour une base vectorielle Chroma.
    
    :param documents: Liste de Documents à indexer
    :param chemin_chroma: Chemin vers le dossier où persister la base Chroma
    :param collection_name: Nom de la collection Chroma
    :return: Un objet Chroma contenant les embeddings des documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splitted_docs = splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    
    vecteur_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=chemin_chroma
    )
    
    vecteur_store.add_documents(splitted_docs)
    vecteur_store.persist()
    
    return vecteur_store

def construire_chatbot(
    vectorstore,
    temperature=0.3,
    model_name="gpt-4-turbo",
    k=4,
    verbose=True
):
    """
    Construit un chatbot RAG basé sur LangChain :
    - Utilise un LLM OpenAI avec un paramètre de température
    - Interroge la base vectorielle (retriever) pour récupérer les passages pertinents
    - Gère la conversation via une mémoire (ConversationBufferMemory)
    
    :param vectorstore: Objet Chroma (ou autre base vectorielle) déjà indexé
    :param temperature: Contrôle la créativité du modèle (0 = + strict, 1 = + créatif)
    :param model_name: Nom du modèle OpenAI (ex: gpt-4-turbo)
    :param k: Nombre de documents à récupérer lors de chaque requête
    :param verbose: Si True, affiche des logs détaillés de la chaîne LangChain
    :return: Un objet ConversationalRetrievalChain prêt à répondre à des questions
    """
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=openai.api_key,
        temperature=temperature
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def custom_retriever(question):
        retrieved_docs = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        ).get_relevant_documents(question)
        
        sources = [doc.metadata.get("source", "Inconnu") for doc in retrieved_docs]
        formatted_sources = "\n".join(set(sources))
        
        return retrieved_docs, formatted_sources
    
    def custom_qa_chain(question):
        retrieved_docs, sources = custom_retriever(question)
        
        response = llm.predict(question)
        response += "\n\nSources:\n" + sources if sources else "\n\n(Aucune source spécifique trouvée)"
        
        return response
    
    return custom_qa_chain
