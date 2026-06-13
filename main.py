import logging
import base64
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
load_dotenv()

import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, load_and_chunk_pdf_bytes, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from openai import OpenAI


# Set INNGEST_ENV=production in Vercel environment variables.
# Local dev leaves this unset (defaults to "development") so the Inngest
# dev server is still used without any changes to your local workflow.
_is_production = os.getenv("INNGEST_ENV", "development") == "production"

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=_is_production,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        source_id = ctx.event.data.get("source_id", "uploaded.pdf")
        pdf_content_b64 = ctx.event.data.get("pdf_content_b64")
        pdf_path = ctx.event.data.get("pdf_path")

        if pdf_content_b64:
            # Production / serverless path: PDF bytes are base64-encoded in the event.
            pdf_bytes = base64.b64decode(pdf_content_b64)
            chunks = load_and_chunk_pdf_bytes(pdf_bytes)
        else:
            # Local dev path: PDF lives on disk and the path is passed directly.
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
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
import pathlib

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the frontend page once at import time. It lives in web/ (not public/)
# because Vercel uploads public/ to its CDN and does NOT include it in the
# serverless function bundle, so the function can't read it at runtime.
_INDEX_PATH = pathlib.Path(__file__).parent / "web" / "index.html"
_INDEX_HTML = _INDEX_PATH.read_text(encoding="utf-8") if _INDEX_PATH.is_file() else None


@app.get("/", include_in_schema=False)
async def serve_index():
    """Serve the upload/query page. FastAPI is Vercel's catch-all, so the
    root route must be handled here rather than relying on static hosting."""
    if _INDEX_HTML is None:
        return PlainTextResponse(
            f"index.html not found at {_INDEX_PATH}", status_code=500
        )
    return HTMLResponse(_INDEX_HTML)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.post("/api/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF upload from the static frontend, base64-encodes it, and
    fires an Inngest event so the existing rag_ingest_pdf function handles
    chunking, embedding, and storage — with throttle/rate-limit intact.
    """
    pdf_bytes = await file.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    source_id = file.filename or "uploaded.pdf"

    await inngest_client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_content_b64": pdf_b64,
                "source_id": source_id,
            },
        )
    )
    return {"status": "triggered", "source_id": source_id}


@app.post("/api/query")
async def query_pdf(req: QueryRequest):
    """
    Synchronous RAG query endpoint used by the static frontend.
    Performs the same embed → search → LLM pipeline as rag_query_pdf_ai
    without requiring the frontend to poll the Inngest API.
    """
    oai = OpenAI()
    query_vec = embed_texts([req.question])[0]
    store = QdrantStorage()
    found = store.search(query_vec, req.top_k)

    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n"
        "Answer concisely using the context above."
    )

    res = oai.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You answer questions using only the provided context."},
            {"role": "user", "content": user_content},
        ],
    )

    answer = res.choices[0].message.content.strip()
    return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}


inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
