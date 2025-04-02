import os
import json
import openai
import fitz  # PyMuPDF pour extraire le texte des PDFs
from langdetect import detect  # Pour détecter la langue du texte
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("La clé API OpenAI n'a pas été trouvée. Vérifiez votre fichier .env.")

# Imports LangChain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# 1. Fonction pour charger et traiter les PDFs
def charger_donnees_pdf(dossier_pdf):
    """
    Parcourt tous les fichiers PDF du dossier_pdf,
    extrait le texte et crée une liste de Documents LangChain.
    """
    documents = []
    for nom_fichier in os.listdir(dossier_pdf):
        if nom_fichier.endswith(".pdf"):
            chemin_fichier = os.path.join(dossier_pdf, nom_fichier)
            try:
                with fitz.open(chemin_fichier) as pdf:
                    texte = "".join(page.get_text("text") for page in pdf)
                    langue = detect(texte)  # Détection automatique de la langue
                    
                    documents.append(
                        Document(
                            page_content=texte,
                            metadata={"source": nom_fichier, "langue": langue}
                        )
                    )
            except Exception as e:
                print(f"❌ Erreur lors du traitement de {nom_fichier} : {e}")
    return documents

# 2. Fonction pour indexer les documents dans Chroma
def preparer_et_indexer_documents(documents, chemin_chroma):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs_split = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vecteur_store = Chroma(
        collection_name="orientation_chatbot",
        embedding_function=embeddings,
        persist_directory=chemin_chroma
    )
    vecteur_store.add_documents(docs_split)
    vecteur_store.persist()
    return vecteur_store

# 3. Construire la chaîne RAG
def construire_chatbot(vectorstore, temperature=0.3):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai.api_key, temperature=temperature)
    memoire = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        memory=memoire,
        verbose=True
    )

# 4. Fonction pour détecter la langue d'une question
def detecter_langue_texte(texte):
    try:
        return detect(texte)
    except:
        return "fr"  # Par défaut français

# 5. Mode interactif
def mode_terminal():
    print("\n🔍 Chargement et indexation des documents...")
    docs = charger_donnees_pdf("./load_documents")
    vecteurs = preparer_et_indexer_documents(docs, "./embeddings")
    chatbot = construire_chatbot(vecteurs)
    print("\n🤖 Chatbot prêt ! Posez vos questions (ou tapez 'exit'):\n")
    
    while True:
        question = input("> ")
        if question.lower() in ["exit", "quit"]:
            break
        langue_utilisateur = detecter_langue_texte(question)
        reponse = chatbot.invoke({"question": question})
        texte_reponse = reponse["answer"]
        
        # Traduction si nécessaire
        if langue_utilisateur == "fr" and detecter_langue_texte(texte_reponse) == "en":
            texte_reponse = openai.ChatCompletion.create(
                model="gpt-4-turbo", messages=[{"role": "system", "content": "Traduisez en français."},
                                                {"role": "user", "content": texte_reponse}]
            )["choices"][0]["message"]["content"]
        elif langue_utilisateur == "en" and detecter_langue_texte(texte_reponse) == "fr":
            texte_reponse = openai.ChatCompletion.create(
                model="gpt-4-turbo", messages=[{"role": "system", "content": "Translate to English."},
                                                {"role": "user", "content": texte_reponse}]
            )["choices"][0]["message"]["content"]
        
        print("\n🧠 Réponse :", texte_reponse, "\n")

if __name__ == "__main__":
    mode_terminal()
