import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema import Document
import yt_dlp
import requests
import os
import pickle
import traceback
from dotenv import load_dotenv
import time

# Load environment variables

# Get your Hugging Face API key from Streamlit Secrets
hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="🎥 YouTube RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0f172a, #1e293b);
            color: white;
        }
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.95));
            backdrop-filter: blur(20px);
        }
        h1, h2, h3 {
            color: #38bdf8 !important;
        }
        .stTextInput, .stTextArea, .stButton>button {
            border-radius: 10px;
        }
        .stButton>button {
            background: linear-gradient(90deg, #2563eb, #06b6d4);
            color: white;
            font-weight: 600;
            padding: 0.5rem 1.2rem;
            border: none;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #06b6d4, #2563eb);
        }
        .video-container {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 0 25px rgba(0,0,0,0.3);
            margin-top: 1rem;
        }
        .navbar {
            background-color: rgba(15,23,42,0.9);
            padding: 1rem 2rem;
            border-radius: 12px;
            margin-bottom: 1rem;
            text-align: center;
            box-shadow: 0 2px 15px rgba(0,0,0,0.3);
        }
        .navbar h1 {
            font-size: 2rem;
            color: #38bdf8;
            margin-bottom: 0;
        }
        .navbar p {
            color: #9ca3af;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ NAVBAR ------------------ #
st.markdown("""
<div class="navbar">
    <h1>🎥 YouTube RAG Chatbot</h1>
    <p>AI-powered video understanding assistant using RAG + HuggingFace + Chroma</p>
</div>
""", unsafe_allow_html=True)

# ------------------ MAIN CONTENT ------------------ #
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 📺 Enter YouTube Details")
    video_id = st.text_input("YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
    question = st.text_area("💬 Ask a Question", placeholder="What is this video about?")

    run_query = st.button("🚀 Ask Question", use_container_width=True)

with col2:
    st.markdown("### 🎬 Video Preview")
    if video_id:
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.info("Enter a YouTube Video ID to preview the video here.")

st.markdown("---")

# ------------------ MAIN FUNCTIONALITY (UNCHANGED) ------------------ #
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_transcript_optimized(video_id: str) -> str:
    st.info("🎧 Fetching transcript from YouTube...")
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'subtitlesformat': 'vtt',
        'quiet': True,
        'no_warnings': True,
    }
    transcript_text = ""
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            subs = info.get('subtitles', {}) or info.get('automatic_captions', {})
            en_subs = subs.get('en') or subs.get('en-US') or subs.get('en-GB')
            if en_subs:
                sub_url = en_subs[0].get('url')
                response = requests.get(sub_url, timeout=10)
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    transcript_text = "\n".join([
                        line for line in lines
                        if not line.strip().startswith(('WEBVTT', 'Kind:', 'Language:', '-->'))
                        and line.strip()
                    ])
    except Exception as e:
        st.warning(f"Transcript extraction warning: {str(e)}")
    return transcript_text[:50000]

@st.cache_resource
def load_embedding_model():
    st.info("🚀 Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_llm():
    st.info("🧠 Loading language model...")
    return ChatHuggingFace(llm=HuggingFaceEndpoint(
        model='HuggingFaceH4/zephyr-7b-beta',
        task='text-generation',
        max_new_tokens=512,
        temperature=0.3,
        timeout=30,
        huggingfacehub_api_token=hf_token
    ))

def get_vectorstore_optimized(video_id: str, docs):
    embedding_model = load_embedding_model()
    
    # ✅ Ensure directories exist before saving
    os.makedirs("rag_data/transcripts", exist_ok=True)
    os.makedirs("rag_data/chroma_db", exist_ok=True)
    
    metadata_file = f"rag_data/transcripts/{video_id}.pkl"
    vector_store_path = "rag_data/chroma_db"
    if os.path.exists(metadata_file):
        st.info(f"📂 Using cached embeddings for `{video_id}`.")
        return Chroma(persist_directory=vector_store_path, embedding_function=embedding_model)
    st.info(f"🧠 Creating embeddings for `{video_id}`...")
    all_texts = [doc.page_content for doc in docs]
    vector_store = Chroma(embedding_function=embedding_model, persist_directory=vector_store_path)
    progress_bar = st.progress(0)
    batch_size = 50
    for i in range(0, len(all_texts), batch_size):
        batch_texts = all_texts[i:i + batch_size]
        vector_store.add_texts(batch_texts)
        progress = min((i + batch_size) / len(all_texts), 1.0)
        progress_bar.progress(progress)
    vector_store.persist()
    with open(metadata_file, "wb") as f:
        pickle.dump({"video_id": video_id, "chunks": len(docs), "created_at": time.time()}, f)
    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(retriever, model):
    prompt = PromptTemplate(
        template="You are a helpful assistant. Use ONLY the following context to answer.\nIf unsure, say 'I don't know.'\n\nContext:\n{context}\n\nQuestion:\n{query}\nAnswer:",
        input_variables=["query", "context"]
    )
    return (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

# ------------------ EXECUTION ------------------ #
if run_query:
    if not video_id or not question:
        st.warning("Please provide both a YouTube video ID and a question.")
    else:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            status_text.text("🔄 Loading AI models...")
            model = load_llm()
            progress_bar.progress(20)

            status_text.text("🎧 Fetching transcript...")
            full_text = fetch_transcript_optimized(video_id)
            progress_bar.progress(40)
            if not full_text.strip():
                st.error("No transcript found. Try another video.")
                st.stop()

            status_text.text("✂️ Splitting transcript...")
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
            docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]
            progress_bar.progress(60)

            status_text.text("🔍 Setting up vector search...")
            vector_store = get_vectorstore_optimized(video_id, docs)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": {"video_id": video_id}})
            progress_bar.progress(80)

            status_text.text("🤖 Generating AI answer...")
            rag_chain = create_rag_chain(retriever, model)
            response_container = st.empty()
            full_response = ""
            for chunk in rag_chain.stream(question):
                full_response += chunk

                # ✅ Clean unwanted user/assistant tokens before display
                cleaned_response = (
                    full_response.replace("[/USER]", "")
                    .replace("[/ASS]", "")
                    .replace("<|user|>", "")
                    .replace("<|assistant|>", "")
                    .strip()
                )

                response_container.markdown(f"### 🗣️ Answer\n{cleaned_response}")

            progress_bar.progress(100)
            status_text.text("✅ Done!")

        except Exception as e:
            st.error("An error occurred. Check below.")
            st.exception(e)
            traceback.print_exc()

# ------------------ SIDEBAR ------------------ #
with st.sidebar:
    st.markdown("### 💡 App Info")
    st.info("Ask intelligent questions about YouTube videos using RAG and HuggingFace models.")
    st.markdown("### ⚡ Performance Tips")
    st.success("""
    - Cached transcripts for faster reuse  
    - Persistent Chroma DB  
    - Optimized embeddings (MiniLM)  
    - Real-time answer streaming  
    """)

# ------------------ FOOTER ------------------ #
st.markdown("---")
st.caption("© 2025 YouTube RAG Chatbot — Powered by LangChain & HuggingFace")
