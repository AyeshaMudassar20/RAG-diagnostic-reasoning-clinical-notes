"""
Streamlit demo for the clinical RAG system.

Retrieves the most relevant preprocessed MIMIC-IV-Ext DiReCT notes for a
query (via a FAISS index over biomedical sentence embeddings), then asks
Phi-3-mini-4k-instruct to answer using only that retrieved context.

Requires `index.faiss` and `preprocessed_documents.jsonl` in the repo root
(both generated locally by Nlp_Ragg.ipynb from your own licensed copy of
MIMIC-IV-Ext DiReCT -- neither is committed to this repo, see README).

Run with:
    streamlit run app.py
"""

import json
import os

import numpy as np
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

INDEX_PATH = "index.faiss"
DOCS_PATH = "preprocessed_documents.jsonl"
EMBEDDING_MODEL_NAME = "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb"  # or a local path, e.g. "embedding_model" (saved by the notebook)
GENERATOR_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"


@st.cache_resource
def load_all():
    import faiss  # imported here to avoid Streamlit build issues

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(
            f"'{INDEX_PATH}' not found. Run Nlp_Ragg.ipynb end-to-end against your "
            "own licensed copy of MIMIC-IV-Ext DiReCT to generate it (see README)."
        )
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(
            f"'{DOCS_PATH}' not found. Run Nlp_Ragg.ipynb end-to-end against your "
            "own licensed copy of MIMIC-IV-Ext DiReCT to generate it (see README)."
        )

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = faiss.read_index(INDEX_PATH)

    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        docs = [json.loads(line) for line in f if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    generator = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    return embedder, index, docs, tokenizer, generator


def retrieve(embedder, index, documents, query: str, k: int = 5):
    """Return the top-k most relevant preprocessed notes for a query."""
    q_emb = embedder.encode([query], normalize_embeddings=True)
    _scores, idxs = index.search(np.array(q_emb).astype("float32"), k)
    return [documents[i] for i in idxs[0]]


def generate_answer(tokenizer, generator, question: str, retrieved_docs):
    """Generate a short, context-only clinical answer using Phi-3-mini."""
    context = "\n\n".join(doc["text"] for doc in retrieved_docs)

    prompt = f"""
You are a clinical question answering assistant. Use ONLY the context.

--- CONTEXT ---
{context}
----------------

Question: {question}
Answer (short, clinical, factual):
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(generator.device)
    outputs = generator.generate(**inputs, max_new_tokens=200, temperature=0.2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


st.set_page_config(page_title="Clinical RAG System", layout="centered")
st.title("Clinical RAG System")
st.write(
    "Ask a clinical question and get an answer grounded in retrieved evidence "
    "from preprocessed MIMIC-IV-Ext DiReCT notes."
)

try:
    embedder, index, documents, tokenizer, generator = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

query = st.text_input("Ask something:")

if st.button("Run"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving..."):
            chunks = retrieve(embedder, index, documents, query)

        st.subheader("Retrieved Evidence")
        for i, c in enumerate(chunks):
            st.markdown(f"**Chunk {i + 1}** ({c.get('diagnosis_category', 'unknown')}):\n{c['text']}")
            st.markdown("---")

        with st.spinner("Generating answer..."):
            answer = generate_answer(tokenizer, generator, query, chunks)

        st.subheader("Answer")
        st.write(answer)
