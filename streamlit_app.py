import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import CSVReader
import os
from datetime import datetime
import uuid
import shutil
from pathlib import Path

st.set_page_config(page_title="Chat with any content", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.getenv("OPENAI_KEY", st.secrets.openai_key)

def get_data_folders():
    """Get all folders in the data directory"""
    if not os.path.exists("./data"):
        os.makedirs("./data")
    folders = [f for f in os.listdir("./data") if os.path.isdir(os.path.join("./data", f))]
    return sorted(folders, reverse=True)  # Most recent first

def get_folder_files(folder_path):
    """Get all files in a folder"""
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

@st.cache_resource(show_spinner=False)
def load_data(folder_path):
    """Load and index data from a specific folder"""
    # First, get all files in the directory
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                if file.lower().endswith('.csv'):
                    # Handle CSV files with the specialized reader
                    csv_reader = CSVReader(concat_rows=False)
                    with open(file_path, 'r') as f:
                        docs = csv_reader.load_data(Path(file_path))
                    all_files.extend(docs)
                else:
                    # Use SimpleDirectoryReader for non-CSV files
                    reader = SimpleDirectoryReader(input_files=[file_path])
                    docs = reader.load_data()
                    all_files.extend(docs)
            except Exception as e:
                print(f"Error loading file {file_path}: {str(e)}")
                continue
    
    print(f"\n=== Loading documents from {folder_path} ===")
    
    # Calculate statistics
    file_counts = {}
    for doc in all_files:
        file_name = doc.metadata.get('file_name', 'unknown')
        file_counts[file_name] = file_counts.get(file_name, 0) + 1
    
    print(f"\nTotal number of documents created: {len(all_files)}")
    print("\nDocuments per file:")
    for file_name, count in file_counts.items():
        print(f"- {file_name}: {count} documents")
    
    # Configure the embedding model
    if st.session_state.model_settings["model_type"] == "OpenAI":
        Settings.embed_model = OpenAIEmbedding(
            model=st.session_state.model_settings["embedding_model"],
        )
    else:
        Settings.embed_model = OllamaEmbedding(
            model_name=st.session_state.model_settings["embedding_model"],
            base_url=st.session_state.model_settings["ollama_base_url"],
        )

    # Configure the LLM
    if st.session_state.model_settings["model_type"] == "OpenAI":
        Settings.llm = OpenAI(
            model=st.session_state.model_settings["model"],
            temperature=0.2,
            system_prompt="""You are an expert on 
            the topic of interest and your 
            job is to answer questions based on the content you have. 
            Keep your answers precise and based on 
            facts – do not hallucinate features."""
        )
    else:
        Settings.llm = Ollama(
            model=st.session_state.model_settings["model"],
            base_url=st.session_state.model_settings["ollama_base_url"],
            temperature=0.2,
            system_prompt="""You are an expert on 
            the topic of interest and your 
            job is to answer questions based on the content you have. 
            Keep your answers precise and based on 
            facts – do not hallucinate features."""
        )
    
    index = VectorStoreIndex.from_documents(all_files)
    return index

st.title("Chat with any content")
st.info("Select a data folder to analyze or upload new documents", icon="📃")

# Sidebar for folder selection and file upload
with st.sidebar:
    st.header("Data Management")
    
    # Model Selection Settings
    st.header("Model Settings")
    model_type = st.selectbox(
        "Select Model Provider",
        ["OpenAI", "Ollama"]
    )
    
    if model_type == "OpenAI":
        model = st.selectbox(
            "Select OpenAI Model",
            ["gpt-4o", "gpt-4o-mini"]
        )
        embedding_model = "text-embedding-ada-002"  # OpenAI's default embedding model
        if not openai.api_key:
            st.error("Please set your OpenAI API key in the secrets.toml file")
    else:
        ollama_base_url = st.text_input(
            "Ollama Base URL",
            value="http://localhost:11434",
            help="The base URL where your Ollama instance is running"
        )
        model = st.text_input(
            "Ollama Model",
            value="deepseek-r1:8b",
            help="The name of the Ollama model to use (e.g.,deepseek-r1:8b, llama3.2:3b, mistral:7b, etc.)"
        )
        embedding_model = st.text_input(
            "Ollama Embedding Model",
            value="nomic-embed-text",
            help="The Ollama model to use for embeddings (e.g., nomic-embed-text)"
        )
    
    # Store model settings in session state
    if "model_settings" not in st.session_state:
        st.session_state.model_settings = {}
    st.session_state.model_settings.update({
        "model_type": model_type,
        "model": model,
        "embedding_model": embedding_model,
        "ollama_base_url": ollama_base_url if model_type == "Ollama" else None
    })
    
    # Folder selection
    folders = get_data_folders()
    folder_options = ["📁 Create New Folder"] + folders if folders else ["📁 Create New Folder"]
    selected_option = st.selectbox(
        "Select a folder to analyze",
        folder_options,
        format_func=lambda x: f"📁 {x}" if x not in ["📁 Create New Folder"] else x
    )
    
    if selected_option == "📁 Create New Folder":
        new_folder_name = st.text_input("Enter folder name:", 
                                      placeholder="my-project-name",
                                      help="Use only letters, numbers, and hyphens")
        if new_folder_name:
            if new_folder_name in folders:
                st.error("A folder with this name already exists")
            elif not all(c.isalnum() or c == '-' for c in new_folder_name):
                st.error("Please use only letters, numbers, and hyphens")
            else:
                if st.button("Create Folder"):
                    new_folder_path = os.path.join("./data", new_folder_name)
                    os.makedirs(new_folder_path, exist_ok=True)
                    st.success(f"Created folder: {new_folder_name}")
                    st.rerun()
        selected_folder = None
        selected_folder_path = None
    else:
        selected_folder = selected_option
        selected_folder_path = os.path.join("./data", selected_folder) if selected_folder else None
        
        # Folder action buttons
        col1, col2 = st.columns(2)
        with col1:
            view_folder = st.button("View Files", use_container_width=True)
        with col2:
            reindex_folder = st.button("Reindex", use_container_width=True)
            
        # View folder contents
        if selected_folder_path and view_folder:
            st.subheader(f"Files in {selected_folder}")
            files = get_folder_files(selected_folder_path)
            if files:
                for file in files:
                    file_col1, file_col2 = st.columns([3, 1])
                    with file_col1:
                        st.text(file)
                    with file_col2:
                        if st.button("🗑️", key=f"delete_{file}"):
                            file_path = os.path.join(selected_folder_path, file)
                            os.remove(file_path)
                            st.rerun()
            else:
                st.warning("No files in this folder")
                
        # Reindex folder
        if selected_folder_path and reindex_folder:
            with st.spinner("Indexing documents..."):
                st.session_state.index = load_data(selected_folder_path)
                st.session_state.current_folder = selected_folder_path
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": f"Documents in '{selected_folder}' have been indexed. Ask me questions about them!"
                    }
                ]
                st.session_state.chat_engine = (
                    st.session_state.index.as_chat_engine(
                        chat_mode="condense_question",
                        verbose=True,
                        streaming=True
                    )
                )
                st.success("Indexing complete!")
                
    # Handle case when no folders are available
    if not folders:
        st.warning("No data folders available. Please upload some documents.")
        selected_folder = None
        selected_folder_path = None

    st.divider()
    
    # File upload section
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['txt', 'md', 'pdf', 'docx', 'csv']
    )
    
    if uploaded_files:
        st.info(f"📤 {len(uploaded_files)} files ready to upload")
        
        # Option to create new folder or use existing
        upload_option = st.radio(
            "Upload to:",
            ["Create new folder", "Add to selected folder"],
            index=1 if selected_folder else 0
        )
        
        if upload_option == "Create new folder":
            folder_name = f"{datetime.now().strftime('%Y-%m-%d')}-{uuid.uuid4().hex[:8]}"
        else:
            if not selected_folder:
                st.error("Please select a folder above first")
                st.stop()
            folder_name = selected_folder
        
        if st.button("Upload Files"):
            upload_dir = os.path.join("./data", folder_name)
            if upload_option == "Create new folder":
                os.makedirs(upload_dir, exist_ok=True)
            
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(upload_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"Files saved in {upload_dir}")
            st.info("Click 'Reindex' to update the chat engine with the new files")
            st.rerun()

# Main chat interface
if "index" in st.session_state and "chat_engine" in st.session_state:
    # Display messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    if prompt := st.chat_input("Ask a question about the content"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            response = st.session_state.chat_engine.stream_chat(prompt)
            response_placeholder = st.empty()
            full_response = ''
            for chunk in response.response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.info("Select a folder and click 'Reindex' to start chatting about your documents.")
