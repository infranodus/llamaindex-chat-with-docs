import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
import os
from datetime import datetime
import uuid
import shutil

st.set_page_config(page_title="Chat with any content", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key

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
    reader = SimpleDirectoryReader(input_dir=folder_path, recursive=True)
    docs = reader.load_data()
    print(f"\n=== Loading documents from {folder_path} ===")
    print(f"Number of documents: {len(docs)}")
    
    print("\n=== First 10 documents ===")
    for i, doc in enumerate(docs[:10]):
        print(f"\n[Document {i+1}]")
        print(f"File name: {doc.metadata.get('file_name', 'N/A')}")
        print(f"Content preview: {doc.text[:200]}...")
    
    if len(docs) > 10:
        print("\n=== Last 10 documents ===")
        for i, doc in enumerate(docs[-10:]):
            print(f"\n[Document {len(docs)-9+i}]")
            print(f"File name: {doc.metadata.get('file_name', 'N/A')}")
            print(f"Content preview: {doc.text[:200]}...")
    
    Settings.llm = OpenAI(
        model="gpt-4",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the topic of interest and your 
        job is to answer questions based on the content you have. 
        Keep your answers precise and based on 
        facts – do not hallucinate features.""",
    )
    return VectorStoreIndex.from_documents(docs)

st.title("Chat with any content")
st.info("Select a data folder to analyze or upload new documents", icon="📃")

# Sidebar for folder selection and file upload
with st.sidebar:
    st.header("Data Management")
    
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
