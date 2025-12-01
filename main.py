import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
# Removed inngest.experimental.ai to use raw Gemini SDK
from dotenv import load_dotenv
import uuid
import os
import datetime
import google.generativeai as genai 

# Import the specific embed function for queries
from data_loader import load_and_chunk_pdf, embed_texts, embed_query 
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGUpsertResult, RAGSearchResult

load_dotenv()

# CHANGED: Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit = 2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    
    # 1. Search Logic
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        # CHANGED: Use the specific query embedding function from data_loader
        query_vec = embed_query(question) 
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    # 2. Generation Logic (Gemini 2.0 Flash)
    def _generate_answer(contexts: list, question: str) -> str:
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        context_block = "\n\n".join(f"- {c}" for c in contexts)
        
        prompt = f"""
        You are a helpful assistant. Use the following context to answer the question concisely.
        
        Context:
        {context_block}

        Question: {question}
        """
        
        response = model.generate_content(prompt)
        return response.text

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Step 1: Retrieve
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Step 2: Generate (Replaces the step.ai.infer block)
    answer = await ctx.step.run("generate-answer", lambda: _generate_answer(found.contexts, question))

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])