import streamlit as st
import requests
import json
import os

# Configure page
st.set_page_config(
    page_title="RAG Physics Assistant",
    page_icon="⚛️",
    layout="wide"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("⚛️ Hybrid RAG Physics Assistant")
st.markdown("Ask me questions about physics! Powered by BM25 + Vector Hybrid Search and Cohere Reranking.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for uploading PDFs
with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a new PDF chapter", type="pdf")
    
    if st.button("Ingest Document") and uploaded_file is not None:
        with st.spinner("Processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            try:
                response = requests.post(f"{API_URL}/ingest", files=files)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Successfully ingested {data.get('num_chunks')} chunks!")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if available
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['filename']}** (Chapter: {source['chapter']}, Page {source['page_number']})")
                    st.info(source['content_preview'])

# Accept user input
if prompt := st.chat_input("What is potential energy?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            with st.spinner("Retrieving and generating answer..."):
                response = requests.post(f"{API_URL}/query", json={"question": prompt})
                
            if response.status_code == 200:
                data = response.json()
                answer = data.get("answer", "No answer provided.")
                sources = data.get("sources", [])
                
                # Update placeholder
                message_placeholder.markdown(answer)
                
                if sources:
                    with st.expander(f"View {len(sources)} Sources"):
                        for source in sources:
                            st.markdown(f"**{source['filename']}** (Chapter: {source['chapter']}, Page {source['page_number']})")
                            st.info(source['content_preview'])
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
            else:
                error_msg = f"Error from API: {response.text}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"Failed to connect to backend. Is the server running? ({e})"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
