

# Document QA Chatbot (RAG)

This project is a **Document Question Answering chatbot** that allows users to ask questions directly from **PDF and DOCX files** instead of manually searching through documents.

It is built using a **Retrieval-Augmented Generation (RAG)** approach, combining vector search with language models for accurate answers.



# Features

* Upload or load PDF/DOCX documents
* Automatic text extraction and chunking
* Semantic search using **FAISS**
* Embeddings with **Sentence Transformers (all-mpnet-base-v2)**
* Optional answer generation using **FLAN-T5**
* Interactive **Streamlit** interface



# How It Works

1. Extract text from documents
2. Split text into overlapping chunks
3. Generate embeddings and store them in FAISS
4. Retrieve relevant chunks for a query
5. Generate an answer using FLAN-T5 (optional)

# Project Structure

```
document-qa-chatbot/
├── core/                 # Extraction, chunking, embeddings, FAISS, generation
├── streamlit_faiss_qa_app.py
├── requirements.txt
└── README.md
```

---

# Run the App

```bash
pip install -r requirements.txt
streamlit run streamlit_faiss_qa_app.py
```

Open: `http://localhost:8501`

---

# Tech Stack

Python, Streamlit, FAISS, Sentence Transformers, HuggingFace Transformers, FLAN-T5

---

#Author

**Parishi Shah**
AI & NLP Enthusiast | RAG Systems | Chatbots

