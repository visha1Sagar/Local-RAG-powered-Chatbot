# RAG-Based PDF Chatbot with Ollama and Gradio

This project implements a Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents. It uses a local Large Language Model (LLM) - Llama 3.1 (3B) via Ollama, stores document embeddings in ChromaDB, and provides a user-friendly web interface built with Gradio.

## ✨ Features

* **PDF Upload & Management:** Upload multiple PDF files through the web interface.
* **Document Processing:** Extracts text from PDFs, splits it into chunks, generates embeddings using Ollama, and stores them in a persistent ChromaDB vector store.
* **Conversational Chat:** Ask questions about the content of your processed PDFs.
* **Local LLM:** Leverages Ollama to run LLMs locally (Llama 3.2), ensuring data privacy.
* **Conversation History:** Remembers the chat context for follow-up questions.
* **Web Interface:** Easy-to-use UI built with Gradio, featuring separate tabs for file management and chatting.
* **File Management:** List uploaded files, delete specific files (including their corresponding vector data), and process all uploaded files in a batch.
* **Persistent Storage:** Uses ChromaDB with a persistent client to store vectors between runs (though the current script clears it on startup).
* **Detailed Logging:** Logs processing steps and errors to `processing.log`.
* **Chat History Search:** Search functionality to search within the current chat session's history.

## 🔧 Technology Stack

* **Python:** Core programming language.
* **Gradio:** Web UI framework.
* **Langchain:** Framework for building LLM applications (loaders, splitters, chains, memory, embeddings).
* **Ollama:** Runs LLMs locally (used for both embeddings and generation).
* **ChromaDB:** Vector database for storing and retrieving document embeddings.

## ⚙️ Prerequisites

1.  **Python:** Version 3.8 or higher recommended.
2.  **Ollama:** Must be installed and running. You can download it from [https://ollama.com/](https://ollama.com/).
3.  **Ollama Model:** The LLM model specified in the script (`llama3.2` by default) must be downloaded. Run:
    ```bash
    ollama pull llama3.2
    ```
    (Replace `llama3.2` if you configure a different model).

## 🚀 Installation

1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🔧 Configuration

You can modify the following constants at the beginning of the `main.py` script:

* `UPLOAD_FOLDER`: Directory where uploaded PDFs are temporarily stored (Default: `"uploaded_files"`).
* `CHROMA_DB_PATH`: Directory where the ChromaDB vector store is persisted (Default: `"chroma_db"`).
* `LLM_MODEL`: The name of the Ollama model to use for embeddings and generation (Default: `"llama3.2"`). Make sure this model is available in your Ollama instance.
* `LOG_FILE`: The name of the file for logging application activity (Default: `"processing.log"`).
* **Ollama Base URL:** The script currently assumes Ollama is running at `http://localhost:11434`. If your Ollama instance is running elsewhere, modify the `base_url` parameter in the `OllamaEmbeddings` and `OllamaLLM` initializations.


## ▶️ Usage

1.  **Start Ollama:** Ensure the Ollama service is running in the background.
2.  **Run the script:**
    ```bash
    python main.py
    ```
3.  **Access the Web UI:** Open your web browser and navigate to the URL provided by Gradio (`http://127.0.0.1:7860`).

4.  **Manage Files:**
    * Go to the "Manage Files" tab.
    * Click "Browse" or drag-and-drop PDF files into the upload area.
    * Click "Upload Selected Files".
    * Once files are uploaded, click "Process All Uploaded Files". This extracts text, creates embeddings, and stores them in ChromaDB. Monitor the "Status" box for progress.
    * You can refresh the file list or delete specific files using the dropdown and "Delete Selected File" button. Deleting a file also removes its associated data from the vector store.

5.  **Chat with Documents:**
    * Go to the "Chatbot" tab.
    * Type your question about the content of the processed PDFs into the input box at the bottom.
    * Press Enter or click "Send".
    * The chatbot will retrieve relevant information from your documents and generate a response using the Ollama LLM. The response will stream into the chat window.
    * Use the "Clear" button to reset the chat history for the current session.
    * Use the search bar at the top of the chat tab to find text within the current conversation history.


## 📁 File Structure (Created by the script)

```
.
├── main.py                 # Your main Python script
├── uploaded_files/        # Stores uploaded PDF files (cleared on startup by default)
│   └── example_doc.pdf
├── chroma_db/           # Persistent storage for ChromaDB (cleared on startup by default)
│   └── ... (database files)
├── processing.log         # Log file for application activity (cleared on startup by default)
└── venv/                  # Virtual environment directory (if created)
```
