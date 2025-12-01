import os
import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

# CHANGED: Configure Google AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# CHANGED: Use Google's embedding model
EMBED_MODEL = "models/text-embedding-004" 
EMBED_DIM = 768

Splitter = SentenceSplitter(chunk_size=512, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(Splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    # CHANGED: Loop through texts and embed using Gemini
    # Note: For production, you might want to batch these 10-100 at a time, 
    # but a simple loop works fine for small files in the free tier.
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=text,
            task_type="retrieval_document",
            title="Embedded Document" 
        )
        embeddings.append(result['embedding'])
    
    return embeddings

# Helper for query embedding (single text)
def embed_query(text: str) -> list[float]:
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return result['embedding']