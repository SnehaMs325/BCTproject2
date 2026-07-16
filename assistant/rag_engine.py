import os
import numpy as np
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    model="mistral-small-latest",
    mistral_api_key=api_key
)

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=api_key
)

def process_uploaded_file(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    
    with NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        temp_path = temp_file.name

    try:
        if ext == '.pdf':
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
        elif ext == '.txt':
            loader = TextLoader(temp_path, encoding='utf-8')
            pages = loader.load()
        else:
            raise ValueError("Unsupported format.")
    finally:
        os.remove(temp_path) 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    text_list = [chunk.page_content for chunk in chunks]

    if not text_list:
        return []

    chunk_embeddings = embeddings.embed_documents(text_list)

    vector_store = []
    for text, vector in zip(text_list, chunk_embeddings):
        vector_store.append({
            "text": text,
            "vector": np.array(vector)
        })
    return vector_store

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def rag_engine(question, vector_store, chat_history_text="", top_k=5):
    """Retrieves relevant chunks and queries the LLM with chat history."""
    if not vector_store:
        return "Please upload a transcript first."

    query_vector = np.array(embeddings.embed_query(question))
    scored_chunks = []

    for item in vector_store:
        score = cosine_similarity(query_vector, item["vector"])
        scored_chunks.append((score, item["text"]))

    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    retrieved_chunks = scored_chunks[:top_k]

    context = "\n\n------------------\n\n".join(text for _, text in retrieved_chunks)

    # --- UPDATED PROMPT LOGIC ---
    prompt = f"""
You are an internal meeting minutes expert and a helpful AI assistant.

Follow these strict rules when responding to the New Question:
1. Greetings & Politeness: If the user says "hello", "hi", "thanks", "bye", or similar conversational greetings, respond naturally, politely, and briefly.
2. Out of Context: If the user asks about general knowledge, coding, or anything completely unrelated to the provided Context or the purpose of analyzing a document, reply EXACTLY with: "This query is out of context. Please ask about the document you uploaded."
3. Document Queries (Found): If the user asks about the document and the information is in the Context, provide the answer in brief, detailed bullet points.
4. Document Queries (Not Found): If the user asks about the document but the answer is NOT present in the Context, reply EXACTLY with: "Information not available."

Previous Conversation:
{chat_history_text}

Context:
{context}

New Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content

def general_chat_engine(question, chat_history_text=""):
    """Queries the LLM directly without vector retrieval for normal chat."""
    prompt = f"""
You are a helpful, intelligent AI assistant. 
Answer the user's question clearly and concisely.

Previous Conversation:
{chat_history_text}

New Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)
    return response.content