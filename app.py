import os
import time
from flask import Flask, render_template, request, session
from dotenv import load_dotenv

# ─── Helpers (inlined) ───────────────────────────────────────────────────────

from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

def load_pdf_file(path):
    loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyPDFLoader)
    return loader.load()

def text_split(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return splitter.split_documents(documents)

def download_hugging_face_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# ─── System Prompt (inlined) ─────────────────────────────────────────────────

system_prompt = (
    """You are Greenie Chat, a warm and friendly AI devoted exclusively to climate change, renewable energy, sustainability, and eco-living.
Keep answers very brief, using simple, non-technical language. Suggest a small, actionable green step whenever appropriate.
If a question is off-topic, reply with a varied gentle redirect, never attempt to answer it. If possible, relate it to the environment (e.g., "Airplanes? Great! Let's talk about their carbon footprint ✈️🌍").
Stay supportive, cheerful, non-judgmental, and focused on helping the user take positive steps for our planet.

Context:
{context}
"""
)

# ─── Flask & Env setup ─────────────────────────────────────────────────────────

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me")

# Pass API keys into environment for downstream clients:
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY", "")
os.environ["GROQ_API_KEY"]     = os.getenv("GROQ_API_KEY", "")

# ─── LangChain / Pinecone / LLM setup ────────────────────────────────────────

from langchain_pinecone import PineconeVectorStore
from langchain_groq       import ChatGroq
from langchain.chains     import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

# Embeddings & index
embeddings = download_hugging_face_embeddings()
docsearch  = PineconeVectorStore.from_existing_index(
    index_name="greenie",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# LLM
llm = ChatGroq(
    temperature=0.4,
    model_name="llama3-8b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Prompt + chain
prompt = ChatPromptTemplate.from_messages([
    (
      "system",
      system_prompt + "\n=== Retrieved docs ===\n{context}\n=== Conversation History ===\n{chat_history}\nNow answer concisely."
    ),
    ("user", "{question}")
])

conv_rag = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt},
    verbose=False
)

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    session.pop("history", None)
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg     = request.form["msg"].strip()
    history = session.get("history", [])

    chat_history = [(h["role"], h["content"]) for h in history]

    try:
        result = conv_rag({"question": msg, "chat_history": chat_history})
        answer = result["answer"]
    except Exception as e:
        print("💥 RAG error:", e)
        answer = "Sorry, I'm having trouble thinking right now. Could you try again in a few seconds?"

    # Persist conversation
    history.extend([
        {"role": "user",      "content": msg},
        {"role": "assistant", "content": answer}
    ])
    session["history"] = history

    return answer

# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)
