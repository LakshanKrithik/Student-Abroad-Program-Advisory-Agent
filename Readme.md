# Student Abroad Program Advisory Chatbot

A Retrieval-Augmented Generation (RAG) based chatbot designed to answer queries from institutional documents and provide personalized study-abroad guidance.

---

## Overview
This project enables users to interact with a conversational AI system that retrieves relevant information from structured and unstructured documents, ensuring accurate and context-aware responses.

---

## Features
- Document-based question answering using RAG  
- Automated data extraction from PDFs using LlamaParse  
- Context-aware response generation with LangChain  
- FastAPI backend for efficient API handling  

---

## System Workflow
1. Documents are parsed and processed into structured data  
2. Relevant information is retrieved based on user queries  
3. The language model generates responses using retrieved context  

---

## Setup and Installation
```bash
git clone https://github.com/LakshanKrithik/Student-Abroad-Program-Advisory-Agent.git
cd SAP-bot   # replace with your backend folder name if different
pip install -r requirements.txt
uvicorn main:app --reload