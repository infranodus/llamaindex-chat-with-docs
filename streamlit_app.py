import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

st.set_page_config(page_title="Chat with any content", page_icon="🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Chat with the any content")
st.info("The content used is in this repository's /data folder", icon="📃")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask me a question about the content!",
        }
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
    docs = reader.load_data()
    print("\n=== INITIAL DOCUMENTS ===")
    print(f"Number of documents loaded by SimpleDirectoryReader: {len(docs)}")
    for i, doc in enumerate(docs[:3]):
        print(f"\nOriginal Document {i + 1}:")
        print(f"Length: {len(doc.text)} characters")
        print(f"Start: {doc.text[:100]}...")
        print(f"End: ...{doc.text[-100:]}")
        print(f"Metadata: {doc.metadata}")

    Settings.llm = OpenAI(
        model="gpt-4o",
        temperature=0.2,
        system_prompt="""You are an expert on 
        the the topic of interest and your 
        job is to answer questions based on the content you have. 
        Keep your answers precise and based on 
        facts – do not hallucinate features.""",
    )
    index = VectorStoreIndex.from_documents(docs)
    print("\n=== INDEXED DOCUMENTS ===")

    print(f"Total number of documents: {len(index.docstore.docs)}")
    for i, doc_id in enumerate(list(index.docstore.docs.keys())[:10]):
        doc = index.docstore.docs[doc_id]
        print(f"\n{'='*50}")
        print(f"DOCUMENT {i + 1}")
        print(f"{'='*50}")
        print(f"Document ID: {doc_id}")
        print(f"File name: {doc.metadata.get('file_name', 'N/A')}")
        print(f"File path: {doc.metadata.get('file_path', 'N/A')}")
        print(f"Creation date: {doc.metadata.get('creation_date', 'N/A')}")
        print(f"Last modified: {doc.metadata.get('last_modified', 'N/A')}")
        print(f"File type: {doc.metadata.get('file_type', 'N/A')}")
        print(f"File size: {doc.metadata.get('file_size', 'N/A')} bytes")
        print("\nFull Document Text:")
        print(f"{'='*20}")
        print(doc.text)
        print(f"{'='*50}\n")
    return index


index = load_data()

if "chat_engine" not in st.session_state.keys():  # Initialize the chat engine
    st.session_state.chat_engine = index.as_chat_engine(
        chat_mode="condense_question", verbose=True, streaming=True
    )

if prompt := st.chat_input(
    "Ask a question"
):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Write message history to UI
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_stream = st.session_state.chat_engine.stream_chat(prompt)
        st.write_stream(response_stream.response_gen)
        message = {"role": "assistant", "content": response_stream.response}
        # Add response to message history
        st.session_state.messages.append(message)
