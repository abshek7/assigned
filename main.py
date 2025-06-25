import os
from fastapi import FastAPI, UploadFile, Form, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import logging
import PyPDF2
import docx
from io import BytesIO
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure Google Generative AI
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Globals
document_name = ""
document_summary = ""
vector_store = None
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# File processors
def extract_text(file: UploadFile):
    ext = file.filename.split('.')[-1].lower()
    content = file.file.read()
    file.file.seek(0)

    if ext == 'pdf':
        reader = PyPDF2.PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() for page in reader.pages)
    elif ext == 'docx':
        doc = docx.Document(BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == 'txt':
        return content.decode('utf-8')
    else:
        return ""

# RAG initialization
def initialize_pipeline(text, doc_name):
    global vector_store, document_name, document_summary
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    document_name = doc_name
    document_summary = ""

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        text = extract_text(file)
        if len(text.strip()) < 50:
            return JSONResponse({"error": "Document too short."}, status_code=400)

        initialize_pipeline(text, file.filename)
        return JSONResponse({"message": "Document uploaded.", "document_name": file.filename})
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return JSONResponse({"error": "Failed to process file."}, status_code=500)

@app.post("/ask")
async def ask_question(question: str = Form(...)):
    global vector_store, document_name, document_summary

    if not vector_store:
        return JSONResponse({"error": "No document loaded."}, status_code=400)

    try:
        general_phrases = [
            "what is this document about", "summarize the document",
            "what's in this document", "document summary"
        ]
        if any(phrase in question.lower() for phrase in general_phrases):
            if not document_summary:
                docs = vector_store.similarity_search("document summary", k=5)
                context = "\n".join([doc.page_content for doc in docs])
                prompt = f"Summarize this document in 2-3 sentences:\n\n{context}"
                document_summary = model.generate_content(prompt).text
            return {"answer": document_summary}

        docs = vector_store.similarity_search(question, k=4)
        if not docs:
            return {"answer": "I can't answer this based on the document."}

        context_text = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""Use only the context to answer the question.

Document: {document_name}
Context: {context_text}
Question: {question}

Answer:"""
        
        response = model.generate_content(prompt).text
        return {"answer": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return JSONResponse({"error": "Error answering question."}, status_code=500)
