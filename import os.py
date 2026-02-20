import os
import pandas as pd
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "WorldCupMatches.csv")
DB_DIR = os.path.join(BASE_DIR, "fifa_vector_db")

def check_ollama():
    """Verify Ollama is running to prevent silent crashes."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def get_retriever():
    """Simply loads the database without doing any heavy indexing."""
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(
        collection_name="fifa_matches",
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    return vector_store.as_retriever(search_kwargs={"k": 5})

def start_indexing(progress_bar):
    """The heavy function that only runs when called by the user."""
    df = pd.read_csv(CSV_FILE).dropna(subset=['Home Team Name', 'Away Team Name'])
    df = df.drop_duplicates(subset=['MatchID'])
    
    docs = []
    for _, row in df.iterrows():
        yr = str(int(row['Year'])) if pd.notnull(row['Year']) else "N/A"
        content = f"Year: {yr} | {row['Home Team Name']} vs {row['Away Team Name']} | Result: {int(row['Home Team Goals'])}-{int(row['Away Team Goals'])}"
        docs.append(Document(page_content=content, metadata={"year": yr}))

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = Chroma(collection_name="fifa_matches", persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Tiny batches of 10 to keep the connection alive
    batch_size = 10
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i : i + batch_size]
        vector_store.add_documents(batch)
        progress_bar.progress((i + len(batch)) / total, text=f"Indexing match {i + len(batch)} of {total}...")
    
    return True