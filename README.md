# 🚀 RAG PDF Chatbot (AI Context Builder)

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline over a PDF document.  
It allows users to ask questions in natural language and get answers grounded in the document.

---

## 🔧 Features

- PDF parsing using PyMuPDF  
- Text chunking for better retrieval  
- Transformer-based embeddings (HuggingFace)  
- Semantic search using FAISS  
- LLM-based response generation (Groq)  
- Query intent detection (LLM-based)  
- Continuous CLI chatbot  
- Unit and integration tests using pytest  

---

## ⚙️ Setup Instructions

### 1. Create virtual environment

```bash
python -m venv myenv
```

Activate (Windows):

```bash
myenv\Scripts\activate
```

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 3. Setup environment variables

Add groq_api_key in the `.env` file:

```bash
GROQ_API_KEY=your_api_key_here
```



---

### 4. Add PDF file 

Place the pdf file in the root directory and rename it as "philippine_history":

```
philippine_history.pdf
```

---

## ▶️ Run the Application  -" Please wait for some duration while running the code for the first time - Till " User Query: " seen on the screen "

```bash
python app.py
```

Example:

```
User Query: Who is José Rizal?

Retrieved Chunks:
- "José Rizal was a Filipino nationalist..."

LLM Response:
José Rizal was a Filipino nationalist and key figure in Philippine history.
```

Type `exit` to quit.

---

## 🧪 Run Tests

Run all tests:

```bash
python -m pytest -q
```



---

## 📁 Project Structure

```
run.py          # CLI chatbot
rag_core.py     # core logic
tests/          # unit tests
```

---

## 🧠 Key Highlights

- End-to-end RAG pipeline  
- Intent-aware response generation  
- Cached FAISS index for performance  
- Modular and clean code  
- Fully tested system  

---

## 📌 Notes

- API keys are not hardcoded  
- Uses `.env` for configuration  
- Defaults provided for easy execution  

---

## 👨‍💻 Author

Abhishek Verma
