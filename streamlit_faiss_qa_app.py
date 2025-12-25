import streamlit as st
import tempfile, os, glob
from typing import List, Dict, Tuple

# -------- Core imports (ALL logic lives here) --------
from core.extraction import extract_any_bytes
from core.chunking import word_overlap_chunks
from core.embeddings import Embedder
from core.vector_store import FaissStore
from core.generation import generate_answer

# -------- Streamlit config --------
st.set_page_config(page_title="📚 Document QA (FAISS)", layout="wide")
st.title("📚 Document QA — FAISS + SentenceTransformers + FLAN-T5")

# -------- Session state --------
if "store" not in st.session_state:
    st.session_state.store = None
if "emb" not in st.session_state:
    st.session_state.emb = None

# -------- Sidebar --------
st.sidebar.header("⚙️ Indexing Settings")

target_words = st.sidebar.number_input(
    "Chunk size (words)", min_value=50, max_value=2000, value=200, step=50
)
overlap_words = st.sidebar.number_input(
    "Chunk overlap (words)", min_value=0, max_value=1000, value=50, step=10
)
batch_size = st.sidebar.number_input(
    "Embedding batch size", min_value=1, max_value=256, value=32
)
dedupe = st.sidebar.checkbox("Remove duplicate chunks", value=True)

top_k = st.sidebar.number_input(
    "Top-K search results", min_value=1, max_value=50, value=6
)
use_flan = st.sidebar.checkbox(
    "Use FLAN-T5 for answer generation (slow)", value=True
)

# -------- Build index UI --------
st.header("📂 Build Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload PDF / DOCX files",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)

folder_path = st.text_input(
    "OR provide a local folder path (PDF/DOCX)", value=""
)

build_btn = st.button("🔨 Build Index")

# -------- Build index logic --------
if build_btn:
    try:
        texts: List[str] = []
        meta: List[Dict] = []

        emb = Embedder()
        st.session_state.emb = emb

        files_to_process = []

        if uploaded_files:
            files_to_process = uploaded_files
        elif folder_path:
            files_to_process = (
                glob.glob(os.path.join(folder_path, "**/*.pdf"), recursive=True)
                + glob.glob(os.path.join(folder_path, "**/*.docx"), recursive=True)
            )
        else:
            st.error("Please upload files or provide a folder path.")
            st.stop()

        with st.spinner("📄 Extracting & chunking documents..."):
            for file in files_to_process:
                if uploaded_files:
                    filename = file.name
                    data = file.read()
                else:
                    filename = os.path.basename(file)
                    with open(file, "rb") as f:
                        data = f.read()

                text, _ = extract_any_bytes(filename, data)

                if not text.strip():
                    st.warning(f"Skipping empty document: {filename}")
                    continue

                chunks = word_overlap_chunks(
                    text,
                    target_words=target_words,
                    overlap_words=overlap_words,
                )

                for cid, (s, e, chunk) in enumerate(chunks):
                    texts.append(chunk)
                    meta.append(
                        {
                            "filename": filename,
                            "chunk_id": f"{filename}_chunk_{cid}",
                            "start": s,
                            "end": e,
                            "preview": chunk[:1500],
                        }
                    )

        if dedupe:
            uniq_texts, uniq_meta = [], []
            seen = set()
            for t, m in zip(texts, meta):
                h = hash(t)
                if h not in seen:
                    seen.add(h)
                    uniq_texts.append(t)
                    uniq_meta.append(m)
            texts, meta = uniq_texts, uniq_meta

        if not texts:
            st.error("No valid text chunks were created.")
            st.stop()

        with st.spinner("🧠 Creating embeddings..."):
            vectors = emb.encode(texts, batch_size=batch_size)

        store = FaissStore(vectors.shape[1])
        store.add(vectors, meta)

        st.session_state.store = store
        st.success(f"✅ Index built successfully ({len(texts)} chunks).")

    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")

# -------- Ask questions --------
st.header("💬 Ask Questions")

query = st.text_input("Enter your question")
ask_btn = st.button("Ask")

if ask_btn:
    if not st.session_state.store or not st.session_state.emb:
        st.error("Please build or load an index first.")
    else:
        with st.spinner("🔍 Searching documents..."):
            q_vec = st.session_state.emb.encode([query])
            hits = st.session_state.store.search(q_vec, top_k=top_k)

        st.subheader("📌 Top Matches")
        for i, (score, m) in enumerate(hits, start=1):
            st.markdown(
                f"**{i}. Score:** {score:.3f} | **File:** {m['filename']}"
            )
            st.write(m["preview"])
            st.markdown("---")

        if use_flan:
            with st.spinner("🤖 Generating answer..."):
                answer = generate_answer(query, hits)
            st.subheader("🧠 Answer")
            st.success(answer)
        else:
            st.subheader("🧠 Answer (Extractive)")
            st.write(" ".join(m["preview"] for _, m in hits[:3]))

# -------- Footer --------
st.markdown("---")
st.caption("RAG-based Document QA | FAISS + SentenceTransformers + FLAN-T5")
