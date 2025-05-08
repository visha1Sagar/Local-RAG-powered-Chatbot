from typing import List, Dict, Any, Generator, Tuple, Union
import os
import tempfile
import shutil
import gradio as gr
import uuid
import fitz  # PyMuPDF
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory

from langchain.schema import HumanMessage, AIMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import logging

# Constants
UPLOAD_FOLDER = "uploaded_files"
CHROMA_DB_PATH = "chroma_db"
LLM_MODEL = "llama3.2"
LOG_FILE = "processing.log" # Define log file constant

# --- Initialization and Setup ---

# Erase previous logs at the beginning
if os.path.exists(LOG_FILE):
    try:
        os.remove(LOG_FILE)
        print(f"Cleared previous log file: {LOG_FILE}")
    except OSError as e:
        print(f"Error clearing previous log file {LOG_FILE}: {e}")

# Configure logging to file
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='[%Y-%m-%d %H:%M:%S]' # Added brackets for clarity
)
logging.info("--- Application Start ---")
logging.info(f"Log file '{LOG_FILE}' initialized.")

# Create upload folder if it doesn't exist
logging.info(f"Ensuring upload folder '{UPLOAD_FOLDER}' exists...")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.info(f"Upload folder '{UPLOAD_FOLDER}' check complete. Exists: {os.path.isdir(UPLOAD_FOLDER)}")

# Clear the upload_folder at the beginning
logging.info(f"Clearing contents of upload folder: {UPLOAD_FOLDER}")
try:
    # Check if the directory exists before listing its contents
    if os.path.exists(UPLOAD_FOLDER):
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    logging.info(f"Removed file: {file_path}")
                except OSError as e:
                    logging.error(f"Error removing file {file_path}: {e}")
        logging.info(f"Upload folder '{UPLOAD_FOLDER}' cleared.")
    else:
         logging.warning(f"Upload folder '{UPLOAD_FOLDER}' did not exist to clear.")
except Exception as e:
    logging.exception(f"Error during clearing upload folder {UPLOAD_FOLDER}")
    raise  # Re-raise the exception to prevent startup with a non-empty folder

# Clear ChromaDB directory at the beginning
logging.info(f"Clearing ChromaDB directory: {CHROMA_DB_PATH}")
try:
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        logging.info(f"Removed ChromaDB directory: {CHROMA_DB_PATH}")
    else:
        logging.info(f"ChromaDB directory '{CHROMA_DB_PATH}' did not exist to clear.")
except Exception as e:
    logging.exception(f"Error during clearing ChromaDB directory {CHROMA_DB_PATH}")
    raise  # Re-raise the exception to prevent startup with a non-empty ChromaDB directory

# Initialize ChromaDB client
logging.info(f"Initializing ChromaDB client at path: {CHROMA_DB_PATH}")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logging.info("ChromaDB client initialized successfully.")
except Exception as e:
    logging.exception(f"Failed to initialize ChromaDB client at path '{CHROMA_DB_PATH}'")
    raise

# Define a custom embedding function for ChromaDB using Ollama
class ChromaDBEmbeddingFunction:
    """Custom EmbeddingFunction for ChromaDB using Langchain's OllamaEmbeddings."""
    def __init__(self, langchain_embeddings):
        logging.info("Initializing ChromaDBEmbeddingFunction...")
        self.langchain_embeddings = langchain_embeddings
        logging.info("ChromaDBEmbeddingFunction initialized.")

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        """
        Generates embeddings for input text(s).
        ChromaDB expects either a single string or a list of strings.
        """
        logging.info(f"Generating embeddings for input...")
        try:
            if isinstance(input, str):
                input = [input]
            embeddings = self.langchain_embeddings.embed_documents(input)
            logging.info(f"Successfully generated embeddings for {len(input)} documents.")
            return embeddings
        except Exception as e:
            logging.exception(f"Error generating embeddings: {str(e)}")
            raise

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return self.langchain_embeddings.embed_documents(docs)

    def embed_query(self, query: str) -> List[float]:
        return self.langchain_embeddings.embed_query(query)


# Initialize the embedding function with Ollama embeddings
logging.info(f"Initializing Ollama embeddings with model '{LLM_MODEL}' and base URL 'http://localhost:11434'...")
try:
    ollama_embeddings = OllamaEmbeddings(
        model=LLM_MODEL,
        base_url="http://localhost:11434"
    )
    embedding_function = ChromaDBEmbeddingFunction(ollama_embeddings)
    logging.info("Ollama embeddings and custom embedding function initialized.")
except Exception as e:
    logging.exception("Failed to initialize Ollama embeddings or custom embedding function.")
    raise

# Initialize the Ollama LLM instance for generation
logging.info(f"Initializing Ollama LLM with model '{LLM_MODEL}' and base URL 'http://localhost:11434'...")
try:
    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url="http://localhost:11434"
    )
    logging.info("Ollama LLM initialized successfully.")
except Exception as e:
    logging.exception("Failed to initialize Ollama LLM.")
    raise

# Initialize Chroma vectorstore
logging.info("Initializing Chroma vectorstore...")
try:
    # First create the collection
    collection = chroma_client.get_or_create_collection(
        name="rag_collection",
        embedding_function=embedding_function
    )
    
    # Then initialize the vectorstore
    vectorstore = Chroma(
        collection_name="rag_collection",
        embedding_function=embedding_function,
        persist_directory=CHROMA_DB_PATH,
        client=chroma_client
    )
    logging.info("Chroma vectorstore initialized successfully.")
except Exception as e:
    logging.exception("Failed to initialize Chroma vectorstore.")
    raise

# Initialize memory for conversation
message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",  # Explicitly set which output to store in memory
    chat_memory=message_history  # Use the new chat memory format
)

# Initialize RAG chain
logging.info("Initializing RAG chain...")
try:
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        callbacks=[StreamingStdOutCallbackHandler()],
        return_source_documents=True,
        output_key="answer"  # Ensure consistent output key
    )
    logging.info("RAG chain initialized successfully.")
except Exception as e:
    logging.exception("Failed to initialize RAG chain.")
    raise

# Define a collection for the RAG workflow
collection_name = "rag_collection"
logging.info(f"Attempting to get or create ChromaDB collection '{collection_name}'...")
try:
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"description": "A collection for RAG with Ollama"},
        embedding_function=embedding_function
    )
    logging.info(f"Successfully got or created collection '{collection_name}'. Current count: {collection.count()}")
except Exception as e:
    logging.exception(f"Failed to get or create ChromaDB collection '{collection_name}'")
    raise

# Helper function to convert Gradio chat history to LangChain message history
def gradio_to_langchain_history(gradio_history: List[List[str]]) -> None:
    """
    Converts Gradio chat history format to LangChain message history.
    This updates the global message_history object.
    """
    logging.info("Converting Gradio chat history to LangChain message history...")
    message_history.clear()  # Clear existing history
    
    for user_msg, ai_msg in gradio_history:
        if user_msg:  # Ensure user message exists
            message_history.add_user_message(user_msg)
        if ai_msg:    # Ensure AI message exists
            message_history.add_ai_message(ai_msg)
    
    logging.info(f"Converted {len(gradio_history)} message pairs to LangChain format.")
    return None

# Helper function to convert LangChain chat messages to Gradio format
def langchain_to_gradio_history() -> List[List[str]]:
    """
    Converts LangChain message history to Gradio chat history format.
    """
    logging.info("Converting LangChain message history to Gradio chat history...")
    gradio_history = []
    messages = message_history.messages
    
    # Process messages in pairs (user -> AI)
    i = 0
    while i < len(messages) - 1:
        if isinstance(messages[i], HumanMessage) and isinstance(messages[i+1], AIMessage):
            gradio_history.append([messages[i].content, messages[i+1].content])
            i += 2
        else:
            # Handle case where messages don't follow user -> AI pattern
            if isinstance(messages[i], HumanMessage):
                gradio_history.append([messages[i].content, ""])
            elif isinstance(messages[i], AIMessage):
                gradio_history.append(["", messages[i].content])
            i += 1
    
    # Handle potential last message if odd number
    if i < len(messages):
        if isinstance(messages[i], HumanMessage):
            gradio_history.append([messages[i].content, ""])
        elif isinstance(messages[i], AIMessage):
            gradio_history.append(["", messages[i].content])
    
    logging.info(f"Converted LangChain messages to {len(gradio_history)} Gradio history pairs.")
    return gradio_history

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> List[List[Any]]:
    """
    Extracts text chunks and metadata from a PDF file using LangChain's PyMuPDFLoader.
    Returns a list containing [chunks, metadata].
    """
    logging.info(f"Starting text extraction from PDF: {file_path}")
    chunks = []
    metadata = []
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found for extraction: {file_path}")
            return [[], []]
        if os.path.getsize(file_path) == 0:
            logging.warning(f"File is empty, skipping extraction: {file_path}")
            return [[], []]

        # Use LangChain's PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
        
        # Initialize text splitter with semantic chunking parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        # Split documents into chunks
        for doc in documents:
            page_chunks = text_splitter.split_text(doc.page_content)
            for chunk in page_chunks:
                if len(chunk.strip()) > 50:  # Minimum chunk size to avoid tiny fragments
                    chunks.append(chunk.strip())
                    chunk_meta = {
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'page_num': doc.metadata.get('page', 1),
                        'pdf_title': doc.metadata.get('title', ''),
                        'pdf_author': doc.metadata.get('author', ''),
                        'pdf_subject': doc.metadata.get('subject', '')
                    }
                    metadata.append(chunk_meta)

        logging.info(f"Finished text extraction from {file_path}. Total {len(chunks)} chunks extracted.")
        return [chunks, metadata]
    except Exception as e:
        logging.exception(f"Error extracting text from {file_path}")
        return [[], []]

# Process single PDF and add to ChromaDB
def process_pdf(file_path: str) -> List[Any]:
    """
    Extracts text from a single PDF and adds it to ChromaDB.
    Returns a list containing [chunks_added, status_message].
    """
    logging.info(f"Starting processing for PDF file: {file_path}")
    try:
        if not os.path.exists(file_path):
            msg = f"Processing skipped: File not found at {file_path}"
            logging.error(msg)
            return [0, msg]
        if os.path.getsize(file_path) == 0:
            msg = f"Processing skipped: File is empty at {file_path}"
            logging.warning(msg)
            return [0, msg]

        # Use the refined extraction function
        result = extract_text_from_pdf(file_path)
        chunks, metadata_list = result[0], result[1]
        logging.info(f"Extraction returned {len(chunks)} chunks for {file_path}.")

        if not chunks:
            msg = f"No processable text chunks extracted from {os.path.basename(file_path)}."
            logging.warning(msg)
            return [0, msg]

        logging.info(f"Preparing to add {len(chunks)} chunks to ChromaDB for {os.path.basename(file_path)}.")
        
        # Generate unique IDs and update metadata with chunk numbers
        ids = []
        for i, (chunk, meta) in enumerate(zip(chunks, metadata_list)):
            # Add chunk number to metadata
            meta['chunk_num'] = i
            # Generate unique ID
            chunk_id = f"{os.path.basename(file_path).replace('.', '_')}_p{meta['page_num']}_c{i}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
        
        logging.debug(f"Generated {len(ids)} IDs for chunks.")

        # Add to ChromaDB collection
        logging.info(f"Adding {len(chunks)} chunks to ChromaDB...")
        try:
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadata_list
            )
            logging.info(f"Successfully added {len(chunks)} chunks to ChromaDB.")
            msg = f"Successfully processed and added {len(chunks)} chunks from {os.path.basename(file_path)}"
            logging.info(msg)
            return [len(chunks), msg]
        except Exception as db_error:
            logging.exception(f"Error adding chunks to ChromaDB for {file_path}")
            msg = f"Error adding chunks to DB for {os.path.basename(file_path)}: {str(db_error)}"
            return [0, msg]

    except Exception as e:
        logging.exception(f"Critical error during processing of {file_path}")
        return [0, f"Error processing {os.path.basename(file_path)}: {str(e)}"]
    finally:
        logging.info(f"Finished processing for PDF file: {file_path}")


# List uploaded PDFs
def get_uploaded_files() -> List[str]:
    """
    Lists valid PDF files in the upload folder. Cleans up empty files.
    """
    logging.info(f"Listing files in upload folder: {UPLOAD_FOLDER}")
    files = []
    if not os.path.isdir(UPLOAD_FOLDER):
        logging.warning(f"Upload folder not found during listing: {UPLOAD_FOLDER}")
        return []
    try:
        for f in os.listdir(UPLOAD_FOLDER):
            path = os.path.join(UPLOAD_FOLDER, f)
            if f.lower().endswith('.pdf'):
                # Check if file exists and is not empty before adding
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    files.append(f)
                    logging.debug(f"Found valid PDF: {f}")
                elif os.path.exists(path): # If it exists but is empty, log and remove
                    logging.warning(f"Removing empty file found during listing: {path}")
                    try:
                        os.remove(path)
                        logging.info(f"Successfully removed empty file: {path}")
                    except OSError as e:
                        logging.error(f"Error removing empty file {path}: {e}")
            else:
                 logging.debug(f"Skipping non-PDF file: {f}")

    except Exception as e:
        logging.exception("Error listing uploaded files")
    sorted_files = sorted(files)
    logging.info(f"Finished listing files. Found {len(sorted_files)} valid PDFs.")
    return sorted_files # Sort for consistent order

# Upload handler
def upload_file(files: List[tempfile._TemporaryFileWrapper]) -> List[str]:
    """
    Handles file uploads, saves them to the upload folder, and returns the updated list.
    """
    logging.info(f"Starting file upload process. Received {len(files) if files else 0} file objects.")
    saved = []
    if not files: # Handle case where no files are selected
        logging.warning("Upload function called with no files selected.")
        # Still return the current list of files
        return get_uploaded_files()

    for file_obj in files: # Gradio provides temp file objects
        if not file_obj:
            logging.warning("Skipping invalid file object during upload.")
            continue

        src_path = file_obj.name # Path to the temporary uploaded file
        original_filename = file_obj.orig_name if hasattr(file_obj, 'orig_name') else os.path.basename(src_path)
        logging.info(f"Processing upload: Original filename '{original_filename}', temporary path '{src_path}'")

        # Sanitize filename (optional but recommended)
        safe_filename = "".join(c for c in original_filename if c.isalnum() or c in (' ', '.', '_', '-')).rstrip()
        # Use a unique prefix to avoid collisions, keep original name part for readability
        # Add .pdf extension explicitly if missing after sanitization
        if not safe_filename.lower().endswith('.pdf'):
             safe_filename += '.pdf'

        filename = f"{uuid.uuid4().hex[:8]}_{safe_filename}"
        dest_path = os.path.join(UPLOAD_FOLDER, filename)
        logging.info(f"Saving uploaded file to: {dest_path}")

        try:
            shutil.copy(src_path, dest_path)
            logging.info(f"Successfully saved uploaded file: {original_filename} as {filename}")
            saved.append(filename)
        except Exception as e:
            logging.exception(f"Error saving uploaded file: {original_filename} from {src_path} to {dest_path}")
            # Optionally notify the user about the failure

    logging.info(f"Finished file upload process. Saved {len(saved)} files.")
    # Refresh the list after upload
    return get_uploaded_files()

# Delete handler
def delete_file(selected_filename: str) -> List[str]:
    """
    Deletes a selected file from the upload folder and attempts to remove its entries from ChromaDB.
    Returns the updated list of files.
    """
    logging.info(f"Starting file deletion process for: {selected_filename}")
    if selected_filename:
        path = os.path.join(UPLOAD_FOLDER, selected_filename)
        logging.info(f"Checking if file exists for deletion: {path}")
        if os.path.exists(path):
            try:
                # Optional: Also remove corresponding entries from ChromaDB
                # This requires querying ChromaDB for IDs associated with this filename
                logging.info(f"Attempting to delete ChromaDB entries for file: {selected_filename}")
                try:
                    # ChromaDB metadata filtering uses dictionary {key: value}
                    results = collection.get(where={"file_name": selected_filename})
                    if results and results.get('ids'):
                        ids_to_delete = results['ids']
                        collection.delete(ids=ids_to_delete)
                        logging.info(f"Deleted {len(ids_to_delete)} chunks from ChromaDB for {selected_filename}")
                    else:
                         logging.info(f"No chunks found in ChromaDB for file: {selected_filename}")
                except Exception as db_delete_error:
                     logging.exception(f"Error deleting ChromaDB entries for {selected_filename}")


                logging.info(f"Attempting to delete file from disk: {path}")
                os.remove(path)
                logging.info(f"Successfully deleted file from disk: {path}")
            except Exception as e:
                logging.exception(f"Error deleting file {path} or its DB entries")
        else:
            logging.warning(f"Attempted to delete non-existent file: {path}")
            # If the file isn't on disk, check if it's in the DB and maybe clean up?
            # This could be added as a separate cleanup function if needed.
    else:
        logging.warning("Delete button clicked but no file selected.")

    logging.info("Finished file deletion process.")
    # Refresh the list after deletion attempt
    return get_uploaded_files()

# Batch process all PDFs
def process_all_files(_=None) -> str:
    """
    Processes all PDF files currently in the upload folder.
    Returns a summary message.
    """
    logging.info("Starting batch processing for all uploaded files...")
    files_to_process = get_uploaded_files() # Get the current list of valid files
    logging.info(f"Found {len(files_to_process)} files to process in batch.")

    if not files_to_process:
        msg = "No valid PDF files found in the upload folder to process."
        logging.info(msg)
        return msg

    total_chunks_added = 0
    files_processed_successfully = 0
    files_skipped_or_failed = 0
    results_summary = []

    logging.info(f"Beginning processing loop for {len(files_to_process)} files...")

    for filename in files_to_process:
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        logging.info(f"--- Processing file: {filename} ---")
        result = process_pdf(file_path)
        count, message = result[0], result[1]
        logging.info(f"Processing result for {filename}: Chunks added = {count}, Message = '{message}'")

        if count > 0:
            files_processed_successfully += 1
            total_chunks_added += count
            results_summary.append(f"{filename}: Added {count} chunks.")
        else:
            files_skipped_or_failed += 1
            # The message from process_pdf contains the reason for skipping/failure
            results_summary.append(f"{filename}: Skipped/Failed ({message}).")
            logging.warning(f"Skipped or failed batch processing for {filename}: {message}")

    status = (f"Batch processing complete. "
              f"Successfully processed: {files_processed_successfully} files ({total_chunks_added} chunks added). "
              f"Skipped/Failed: {files_skipped_or_failed} files.")
    logging.info(status)
    # Optionally return a more detailed summary
    # return status + "\n\nDetails:\n" + "\n".join(results_summary)
    return status


# Query ChromaDB
def query_chromadb(query_text: str, n_results: int = 5) -> List[List[Any]]:
    """
    Queries ChromaDB for relevant documents based on the query text.
    Returns a list containing [documents, metadatas, distances].
    """
    logging.info(f"Starting ChromaDB query for text (first 50 chars): '{query_text[:50]}...' with n_results={n_results}")
    try:
        # Ensure embedding function is correctly used by the collection
        # The collection was created with the embedding_function, so query should use it implicitly
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            # Optional: Add filtering if needed, e.g., where={"file_name": "specific_doc.pdf"}
            # where_document={"$contains": "search_term"} # Example content filter
        )
        # Ensure structure is as expected, provide defaults if keys are missing
        docs = results.get('documents', [[]])
        metas = results.get('metadatas', [[]])
        distances = results.get('distances', [[]]) # Optionally get distances for relevance check
        logging.info(f"ChromaDB query returned {len(docs[0]) if docs and docs[0] else 0} results.")
        return [docs, metas, distances]
    except Exception as e:
        logging.exception(f"Error querying ChromaDB for query: {query_text}")
        return [[[]],[[]], [[]]] # Return empty lists on error
    finally:
         logging.info(f"Finished ChromaDB query for text (first 50 chars): '{query_text[:50]}...'")


def rag_query(query: str, gradio_history: List[List[str]]) -> Generator[Tuple[str, List[List[str]]], None, None]:
    """
    Handles user query, performs RAG, and streams LLM response using LangChain's RAG chain.
    Converts between Gradio and LangChain history formats properly.
    Yields tuples of (query, updated_history) for streaming to both textbox and chatbot components.
    """
    logging.info(f"Starting RAG query process for user query: '{query}'")
    logging.debug(f"Initial gradio_history received: {gradio_history}")

    # Convert Gradio history to LangChain format
    gradio_to_langchain_history(gradio_history)
    
    # Add current query to history in Gradio format for display
    current_gradio_history = gradio_history.copy()
    current_gradio_history.append([query, ""])
    # Yield both the query (to clear input box) and updated history
    yield "", current_gradio_history
    logging.debug("Yielded initial history state in Gradio format.")

    try:
        # Initialize streaming response
        response_text = ""
        
        # Add the user message to LangChain history
        message_history.add_user_message(query)
        
        # Use LangChain's RAG chain with streaming
        for chunk in qa_chain.stream({"question": query}):
            if "answer" in chunk:
                # Accumulate the response text
                response_text += chunk["answer"]
                # Update Gradio history with the current response
                current_gradio_history[-1][1] = response_text
                yield "", current_gradio_history
        
        # After streaming completes, add the final AI message to LangChain history
        message_history.add_ai_message(response_text)
        
        logging.info(f"RAG query completed successfully for query: '{query}'")

    except Exception as e:
        logging.exception("Error during RAG query")
        error_message = f"Error: {str(e)}"
        current_gradio_history[-1][1] = error_message
        # Add error message to LangChain history too
        message_history.add_ai_message(error_message)
        yield "", current_gradio_history
        logging.error("Yielded error message due to RAG chain failure.")

# --- Gradio UI Definition ---
logging.info("Defining Gradio UI...")
demo = gr.Blocks(theme=gr.themes.Soft())

with demo:
    gr.Markdown("# RAG-based PDF Chatbot")

    with gr.Tab("Manage Files"):
        gr.Markdown("Upload PDF files here and process them to make them available for the chatbot.")
        with gr.Row():
            with gr.Column(scale=2):
                file_upload = gr.File(
                    label="Upload PDF Files",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                with gr.Row():
                    upload_btn = gr.Button("Upload Selected Files")
                    process_btn = gr.Button("Process All Uploaded Files")
                status_box = gr.Textbox(label="Status", lines=3, interactive=False, value="Ready.")

            with gr.Column(scale=1):
                gr.Markdown("### Uploaded & Processable Files")
                # Use `value` to initialize, `choices` updated by function return
                file_list = gr.Dropdown(
                    label="Select a file to delete (Optional)",
                    choices=get_uploaded_files(), # Populate initial list
                    value=None, # Start with nothing selected
                    interactive=True,
                    allow_custom_value=True # Prevent adding non-existent files
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh List")
                    delete_btn = gr.Button("Delete Selected File")
                    


        # Define button actions for file management tab
        logging.info("Setting up file management tab button actions...")
        upload_btn.click(
            fn=upload_file,
            inputs=[file_upload],
            outputs=[file_list] # Output updates the choices in the dropdown
        ).then(
            fn=lambda files: f"Upload complete. Saved {len(files) if files else 0} files. Refresh list if needed." if files else "Upload complete. No files selected.", # Provide feedback
            inputs=[file_upload], # Pass the uploaded files list to count
            outputs=[status_box]
        )

        refresh_btn.click(
            fn=get_uploaded_files,
            inputs=None,
            outputs=[file_list] # Refreshes the dropdown choices
        ).then(
             fn=lambda: "File list refreshed.", # Provide feedback
             inputs=None,
             outputs=[status_box]
        )


        delete_btn.click(
            fn=delete_file,
            inputs=[file_list], # Pass the selected filename from dropdown
            outputs=[file_list] # Update the dropdown choices after deletion
        ).then(
            fn=lambda x: f"Attempted deletion of '{x}'. Refresh list." if x else "No file selected for deletion.",
            inputs=[file_list], # Get the filename that was selected for the message
            outputs=[status_box]
        )

        process_btn.click(
            fn=process_all_files,
            inputs=None, # No direct inputs needed, function gets files from disk
            outputs=[status_box] # Show processing status
        )
        logging.info("File management tab button actions configured.")

    with gr.Tab("Chatbot"):
        gr.Markdown("""
        # Chat with your Documents
        
        Use the search bar to find specific messages in your chat history, or ask questions about your documents below.
        """)
        
        # Add search bar in a more integrated way
        with gr.Row():
            with gr.Column(scale=4):
                search_query = gr.Textbox(
                    label="",
                    placeholder="🔍 Search through your chat history...",
                    lines=1,
                    show_label=False
                )
            with gr.Column(scale=1):
                search_btn = gr.Button("Search", variant="secondary")
        
        # Add search results in a collapsible section
        with gr.Accordion("Search Results", open=False):
            search_results = gr.Textbox(
                label="",
                lines=3,
                interactive=False,
                show_label=False
            )
        
        # Main chat interface
        chat = gr.Chatbot(
            label="",
            height=500,
            show_copy_button=True
        )
        
        # Chat input with improved styling
        with gr.Row():
            with gr.Column(scale=4):
                query = gr.Textbox(
                    label="",
                    placeholder="💬 Ask a question about your documents...",
                    lines=1,
                    show_label=False
                )
            with gr.Column(scale=1):
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

        # Define button actions for chatbot tab
        logging.info("Setting up chatbot tab button actions...")
        
        # Add search functionality for chat history
        def perform_search(query_text: str, chat_history: List[List[str]]) -> str:
            """
            Performs a search through the chat history and returns formatted results.
            """
            logging.info(f"Performing search in chat history for query: {query_text}")
            try:
                if not query_text.strip():
                    return "Please enter a search term."
                
                if not chat_history:
                    return "No chat history to search through."
                
                # Convert chat history to a list of messages
                messages = []
                for user_msg, bot_msg in chat_history:
                    if user_msg:
                        messages.append(f"👤 User: {user_msg}")
                    if bot_msg:
                        messages.append(f"🤖 Assistant: {bot_msg}")
                
                # Search through messages
                matching_messages = []
                query_lower = query_text.lower()
                for msg in messages:
                    if query_lower in msg.lower():
                        matching_messages.append(msg)
                
                if not matching_messages:
                    return "No matching messages found in chat history."
                
                return "\n\n".join(matching_messages)
            except Exception as e:
                logging.exception(f"Error during chat history search: {str(e)}")
                return f"Error performing search: {str(e)}"
        
        # Connect search button to search function
        search_btn.click(
            fn=perform_search,
            inputs=[search_query, chat],
            outputs=[search_results]
        )
        
        # Allow searching via Enter key
        search_query.submit(
            fn=perform_search,
            inputs=[search_query, chat],
            outputs=[search_results]
        )
        
        # Use the streaming function `rag_query`
        submit_btn.click(
            fn=rag_query,
            inputs=[query, chat],
            outputs=[query, chat]
        )
        # Allow submitting via Enter key press on the textbox
        query.submit(
            fn=rag_query,
            inputs=[query, chat],
            outputs=[query, chat]
        )

        # Clear button action
        clear_btn.click(
            fn=lambda: ("", None),
            inputs=None,
            outputs=[query, chat]
        ).then(
             fn=lambda: "Chat history cleared.",
             inputs=None,
             outputs=[status_box]
        )
        logging.info("Chatbot tab button actions configured.")

logging.info("Gradio UI definition complete.")

# Launch the Gradio app
if __name__ == "__main__":
    logging.info("Preparing to launch Gradio application...")
    # Example: Ensure collection is ready before launch
    if not collection:
        logging.error("ChromaDB collection failed to initialize before launch. Exiting.")
        # Consider a more graceful exit or handling in a real app
        exit(1)

    logging.info("Launching Gradio application...")
    # The server will start and handle requests
    demo.launch(debug=True) # Enable debug for more detailed errors during development
    logging.info("Gradio application launched (or launch attempt completed).")