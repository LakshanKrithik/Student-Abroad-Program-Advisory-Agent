import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to hold our vector store and chain temporarily in memory
vector_store = None
qa_chain = None

class QueryRequest(BaseModel):
    question: str

system_prompt = (
    "You are an AI assistant for the Student Abroad Program (SAP). "
    "Answer the user's questions using the provided brochure context.\n\n"

    "Guidelines:\n"
    "- Base your answers primarily on the given context.\n"
    "- Provide clear, well-structured, and helpful responses.\n"
    "- If the answer is not found in the context, respond with:\n"
    "'Information not available in the brochure. Kindly contact the Faculty Incharge.'\n"
    "- Do not invent facts or provide unsupported information.\n\n"

    "Keep answers professional, accurate, and concise.\n\n"

    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, qa_chain
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
            
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF. It might be scanned or empty.")

        # Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and vector store
        api_key = os.environ.get("GEMINI_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        # Setup Retrieval chain
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key) # using a fast model
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return {"message": "Brochure processed successfully.", "chunks": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_bot(request: QueryRequest):
    global qa_chain
    if not qa_chain:
        raise HTTPException(status_code=400, detail="No brochure has been uploaded yet.")
        
    try:
        response = qa_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
