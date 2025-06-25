# ⚡ RAG Pipeline with FastAPI + HTML/CSS Frontend

A lightweight **Retrieval-Augmented Generation (RAG)** pipeline powered by **FastAPI** for backend APIs and **HTML/CSS** for a clean user interface. This system answers questions based only on a provided knowledge base using semantic search and relevance filtering.

---

## 🎯 Project Overview

This project implements an end-to-end RAG pipeline with:
- ✅ **FastAPI** backend serving API endpoints  
- ✅ **Static HTML/CSS** frontend for user interaction  
- ✅ **FAISS vector store** for document similarity search  
- ✅ **Google Gemini** for intelligent, grounded answer generation  
---

## 🚀 Features

- 🔍 Document-based semantic search  
- 🧠 Gemini LLM-based generation  
- ❓ Returns fallback for out-of-scope queries  
- 🌐 Lightweight HTML/CSS frontend  
- ⚡ FastAPI-powered async API  

---

## 📋 Prerequisites

- Python 3.8+
- Google API Key for Gemini AI  
  👉 [Get your API key](https://makersuite.google.com/app/apikey)

---

## 🛠️ Installation

```bash
# 1. Clone the repo
git clone <your-repository-url>
cd rag-fastapi-app

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file in the project root
echo "GOOGLE_API_KEY=your_google_api_key_here" > .env

uvicorn main:app --host 0.0.0.0 --port 8000

```

## 🧠 How It Works

### RAG Pipeline Architecture

1. **Document Loading**: Sample AI/ML document is loaded as knowledge base
2. **Text Chunking**: Document split into 1000-character chunks with 200 overlap
3. **Embedding Generation**: Text chunks converted to vectors using Google embeddings
4. **Vector Storage**: Embeddings stored in FAISS for efficient similarity search
5. **Query Processing**: User query embedded and searched against document vectors
6. **Relevance Check**: Similarity score compared against threshold (default: 0.7)
7. **Response Generation**: 
   - If relevant: LLM generates answer using retrieved context
   - If irrelevant: Returns standard message

### Key Components

- **Embedding Model**: `models/embedding-001` (Google)
- **LLM Model**: `gemini-2.5-flash-preview-05-20`
- **Vector Store**: FAISS with cosine similarity

## Troubleshooting

**Common Issues:**

1. **API Key Error**: Make sure your Google API key is set in `.env` file
2. **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
3. **Model Access**: Verify your API key has access to Gemini models

**Debug Mode:**
Set logging level in the code to `DEBUG` for detailed logs.

## Performance

- **Response Time**: 2-5 seconds per query
- **Accuracy**: High for domain-specific questions
- **Scalability**: Handles documents up to 100k+ words
- **Memory Usage**: Efficient with FAISS indexing

