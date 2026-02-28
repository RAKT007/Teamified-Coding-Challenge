import os
import re
import json
from dataclasses import dataclass
from typing import List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import fitz  # PyMuPDF
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


@dataclass
class Config:
    pdf_path: str
    groq_model: str
    embed_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    embed_max_length: int
    index_dir: str


def load_pdf_text_from_path(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc.load_page(i).get_text("text")
        text = re.sub(r"\s+\n", "\n", text)
        pages.append(text.strip())
    doc.close()
    return "\n\n".join(pages).strip()


def make_documents(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]


class TransformersMeanPoolingEmbeddings(Embeddings):
    def __init__(self, model_name: str, device: Optional[str] = None, max_length: int = 256):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        out = self.model(**enc)
        token_embeddings = out.last_hidden_state  # (B, T, H)

        attention_mask = enc["attention_mask"].unsqueeze(-1)  # (B, T, 1)
        masked = token_embeddings * attention_mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        mean_pooled = summed / counts

        mean_pooled = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        return mean_pooled.detach().cpu().numpy().astype(np.float32).tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 32
        vecs: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            vecs.extend(self._embed_batch(texts[i:i + batch_size]))
        return vecs

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]


def load_or_build_vectorstore(cfg: Config) -> FAISS:
    embeddings = TransformersMeanPoolingEmbeddings(cfg.embed_model, max_length=cfg.embed_max_length)

    if os.path.exists(cfg.index_dir):
        return FAISS.load_local(cfg.index_dir, embeddings, allow_dangerous_deserialization=True)

    text = load_pdf_text_from_path(cfg.pdf_path)
    docs = make_documents(text, cfg.chunk_size, cfg.chunk_overlap)
    vs = FAISS.from_documents(docs, embeddings)

    os.makedirs(cfg.index_dir, exist_ok=True)
    vs.save_local(cfg.index_dir)
    return vs


INTENT_LABELS = [
    "date_timeline",
    "person_bio",
    "summary",
    "comparison",
    "definition",
    "other",
]


def detect_intent_with_groq(groq_api_key: str, groq_model: str, question: str) -> str:
    llm = ChatGroq(groq_api_key=groq_api_key, model=groq_model, temperature=0.0)

    prompt = f"""
You are a strict classifier for a RAG system.

Classify the user's question into exactly ONE of these labels:
{INTENT_LABELS}

Return ONLY a JSON object with schema:
{{"intent": "<one_label_from_list>"}}

User question:
{question}
""".strip()

    resp = llm.invoke(prompt).content.strip()

    m = re.search(r"\{.*\}", resp, flags=re.DOTALL)
    if not m:
        return "other"

    try:
        obj = json.loads(m.group(0))
        intent = obj.get("intent", "other")
        return intent if intent in INTENT_LABELS else "other"
    except Exception:
        return "other"


def answer_with_groq(
    groq_api_key: str,
    groq_model: str,
    question: str,
    docs: List[Document],
    intent: str,
) -> str:
    llm = ChatGroq(groq_api_key=groq_api_key, model=groq_model, temperature=0.2)

    context = "\n\n---\n\n".join(
        [f"[chunk_id={d.metadata.get('chunk_id')}]\n{d.page_content}" for d in docs]
    )

    intent_instructions = {
        "date_timeline": "Answer with the exact date(s)/year(s) first, then 1–2 lines of supporting context.",
        "person_bio": "Answer as a short bio: who they are + why important, in 3–6 bullet points.",
        "summary": "Provide a concise summary in 5–8 bullet points.",
        "comparison": "Compare clearly using bullets: Aspect → A vs B.",
        "definition": "Define the term and give 2–3 key details from the context.",
        "other": "Answer clearly and concisely based on the context.",
    }
    instruction = intent_instructions.get(intent, intent_instructions["other"])

    prompt = (
        "You are a helpful assistant. Answer ONLY using the provided context.\n"
        "If the answer is not present in the context, say: 'Not enough information in the provided document.'\n\n"
        f"Intent: {intent}\n"
        f"Instruction: {instruction}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "Answer:"
    )

    return llm.invoke(prompt).content


def compact_snippet(text: str, max_len: int = 220) -> str:
    t = text.strip().replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    return t[:max_len] + ("..." if len(t) > max_len else "")