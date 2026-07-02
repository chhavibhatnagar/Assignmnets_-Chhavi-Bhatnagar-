# 📚 Smart Research Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant built to make it easier to explore and understand PDF documents. Instead of searching through long research papers or reports manually, users can upload one or more PDFs, ask questions in natural language and receive answers based on the uploaded content.

## What it does

The assistant retrieves the most relevant information from the uploaded documents before generating a response, helping reduce hallucinations by keeping the answers grounded in the document itself. Along with each response, it also provides supporting source evidence and evaluation metrics to give users more confidence in the generated answer.

### Key Features

* Chat with uploaded PDF documents using natural language
* Upload and process multiple PDF documents
* Generate document summaries
* Extract important headings and sections
* Highlight important content directly in the original PDF
* Multi-turn conversational memory for follow-up questions
* Multi-language responses
* Domain-specific response modes (General, Legal, Medical, Finance, and Research)
* Quick Actions for common document tasks such as summaries, key points, viva questions, research gaps, and document highlighting
* Support for both FAISS and ChromaDB vector databases
* Support for MiniLM and MPNet embedding models
* User authentication
* Source evidence and response evaluation using RAGAs metrics

## Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **Framework:** LangChain
* **LLMs:** Groq (Llama 3.3 70B) and OpenAI GPT-4o-mini
* **Embeddings:** Sentence Transformers (MiniLM, MPNet)
* **Vector Databases:** FAISS, ChromaDB
* **Evaluation:** RAGAs
* **PDF Processing:** PyMuPDF (fitz)

## How it works

1. The user logs in and uploads one or more PDF documents.
2. The documents are extracted, split into overlapping chunks, and converted into vector embeddings.
3. The embeddings are stored in the selected vector database (FAISS or ChromaDB).
4. When a question is asked, the query is converted into an embedding and matched against the stored document embeddings.
5. The most relevant document chunks are retrieved and passed to the language model as context.
6. The model generates an answer using only the retrieved information.
7. The response is evaluated using RAGAs metrics and displayed together with the relevant source evidence.

## Future Improvements

* Voice-based interaction
* Automatic citation generation
* Multi-document comparison
* Export chat history and summaries
* Cloud deployment


# Steps

1. Clone or download the repository.

2. Open the project in your preferred IDE.

3. Create and activate a virtual environment.

4. Install all the required dependencies using the `requirements.txt` file.

5. Create a folder named `.streamlit` inside the `Smart_Research_Assistant` directory.

6. Inside the `.streamlit` folder, create a file named `secrets.toml` and add the following:

```toml
[credentials]
username = "your_username"
password = "your_password"

[openai]
api_key = "your_openai_api_key"

[groq]
api_key = "your_groq_api_key"
```

7. Replace the placeholder values with your own credentials and API keys.

8. Run the Streamlit application and open the local URL displayed in the terminal.


## Demo Login

Use the following credentials to access the application:

**Username:** `admin`
**Password:** `research123`
