import os
import tempfile
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

client = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def load_and_chunk_pdf_bytes(pdf_bytes: bytes) -> list[str]:
    """
    Same as load_and_chunk_pdf but accepts raw PDF bytes instead of a path.
    Used in the Netlify / serverless path where the PDF is base64-encoded in
    the Inngest event payload rather than saved to disk.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        return load_and_chunk_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]
