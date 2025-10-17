# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers.string import StrOutputParser
# from langchain.schema import Document
# import yt_dlp
# import requests
# import traceback
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Streamlit UI
# st.set_page_config(page_title="🎥 YouTube RAG Chatbot", page_icon="🤖", layout="centered")
# st.title("🎥 YouTube RAG Chatbot")
# st.markdown("Ask any question about a YouTube video — powered by **RAG + HuggingFace + Chroma**")

# # Inputs
# video_id = st.text_input("📺 Enter YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
# question = st.text_area("💬 Ask a Question", placeholder="What is the video about?")

# # Helper: Fetch transcript
# def fetch_transcript(video_id: str) -> str:
#     """Fetch transcript using yt-dlp (captions only)."""
#     url = f"https://www.youtube.com/watch?v={video_id}"
#     ydl_opts = {
#         'skip_download': True,
#         'writesubtitles': True,
#         'writeautomaticsub': True,
#         'subtitleslangs': ['en'],
#         'subtitlesformat': 'vtt',
#         'quiet': True
#     }
#     transcript_text = ""
#     with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#         info = ydl.extract_info(url, download=False)
#         if 'requested_subtitles' in info and info['requested_subtitles']:
#             subs = info['requested_subtitles'].get('en') or info['requested_subtitles'].get('en-US')
#             if subs and 'url' in subs:
#                 r = requests.get(subs['url'])
#                 if r.status_code == 200:
#                     lines = r.text.splitlines()
#                     transcript_text = "\n".join([line for line in lines if "-->" not in line and line.strip() != ""])
#     return transcript_text

# # Process on button click
# if st.button("🚀 Ask"):
#     if not video_id or not question:
#         st.warning("Please provide both a YouTube video ID and a question.")
#     else:
#         try:
#             with st.spinner("Fetching transcript and processing... ⏳"):
#                 # Load LLM
#                 llm = HuggingFaceEndpoint(
#                     model='meta-llama/Llama-3.1-8B-Instruct',
#                     task='text-generation'
#                 )
#                 model = ChatHuggingFace(llm=llm)

#                 # Fetch transcript
#                 full_text = fetch_transcript(video_id)
#                 if not full_text.strip():
#                     st.error("No transcript found for this video. Try another one.")
#                     st.stop()

#                 # Split text into chunks
#                 splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#                 docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]

#                 # Embeddings + Chroma
#                 embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
#                 collection_name = f"youtube_{video_id}"
#                 vector_store = Chroma(
#                     embedding_function=embedding_model,
#                     persist_directory='rag',
#                     collection_name=collection_name
#                 )

#                 # Add new embeddings if not cached
#                 if len(vector_store.get()) == 0:
#                     texts = [doc.page_content for doc in docs]
#                     embeddings = embedding_model.embed_documents(texts)
#                     vector_store.add_texts(texts=texts, embeddings=embeddings)
#                     st.info(f"🧠 Added {len(docs)} transcript chunks to the vector store.")
#                 else:
#                     st.info(f"📂 Using cached embeddings for video `{video_id}`.")

#                 retriever = vector_store.as_retriever(search_kwargs={'k': 4})

#                 # Define RAG chain
#                 def format_docs(docs):
#                     return "\n\n".join([doc.page_content for doc in docs])

#                 prompt = PromptTemplate(
#                     template=(
#                         "You are a helpful assistant.\n"
#                         "Answer the question using ONLY the context below.\n"
#                         "If the answer is not in the context, reply 'I don't know.'\n\n"
#                         "Context:\n{context}\n\nQuestion:\n{query}"
#                     ),
#                     input_variables=['query', 'context']
#                 )

#                 chain1 = RunnableParallel({
#                     'context': retriever | RunnableLambda(format_docs),
#                     'query': RunnablePassthrough()
#                 })
#                 parser = StrOutputParser()
#                 chain2 = prompt | model | parser
#                 final_chain = chain1 | chain2

#                 # Generate answer
#                 st.subheader("🧩 Generating Answer...")
#                 answer = final_chain.invoke(question)
#                 st.success("✅ Done!")

#                 # Display
#                 st.markdown("### 🗣️ Answer")
#                 st.write(answer)

#         except Exception as e:
#             st.error("An error occurred. Check details below.")
#             st.exception(e)
#             traceback.print_exc()
# import streamlit as st
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
# from langchain_core.output_parsers.string import StrOutputParser
# from langchain.schema import Document
# import yt_dlp
# import requests
# import os
# import pickle
# import traceback
# from dotenv import load_dotenv
# import asyncio
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import time

# # Load environment variables
# load_dotenv()

# # Streamlit Page Config
# st.set_page_config(
#     page_title="🎥 YouTube RAG Chatbot", 
#     page_icon="🤖", 
#     layout="centered"
# )
# st.title("🎥 YouTube RAG Chatbot")
# st.markdown("Ask any question about a YouTube video — powered by **RAG + HuggingFace + Chroma**")

# # Directory to persist data
# os.makedirs("rag_data", exist_ok=True)
# os.makedirs("rag_data/transcripts", exist_ok=True)

# # Inputs
# video_id = st.text_input("📺 Enter YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
# question = st.text_area("💬 Ask a Question", placeholder="What is the video about?")

# # -------------------------- OPTIMIZED Utility Functions -------------------------- #

# @st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
# def fetch_transcript_optimized(video_id: str) -> str:
#     """Fetch transcript using optimized yt-dlp settings."""
#     st.info("🎧 Fetching transcript from YouTube...")
    
#     url = f"https://www.youtube.com/watch?v={video_id}"
    
#     # Optimized yt-dlp options for faster transcript extraction
#     ydl_opts = {
#         'skip_download': True,
#         'writesubtitles': True,
#         'writeautomaticsub': True,
#         'subtitleslangs': ['en'],
#         'subtitlesformat': 'vtt',
#         'quiet': True,
#         'no_warnings': True,
#         'extract_flat': False,
#         'force_json': False,
#         'simulate': True,
#         'nooverwrites': True
#     }
    
#     transcript_text = ""
#     try:
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             # Use extract_info with no download for faster processing
#             info = ydl.extract_info(url, download=False)
            
#             # Check for available subtitles
#             subtitles = info.get('subtitles', {}) or info.get('automatic_captions', {})
            
#             # Prefer English subtitles
#             en_subs = subtitles.get('en') or subtitles.get('en-US') or subtitles.get('en-GB')
            
#             if en_subs:
#                 # Get the first available subtitle URL
#                 sub_url = en_subs[0].get('url') if en_subs else None
                
#                 if sub_url:
#                     # Faster request with timeout
#                     response = requests.get(sub_url, timeout=10)
#                     if response.status_code == 200:
#                         lines = response.text.splitlines()
#                         # Efficient filtering of timestamp lines
#                         transcript_text = "\n".join([
#                             line for line in lines 
#                             if not line.strip().startswith(('WEBVTT', 'Kind:', 'Language:', '-->')) 
#                             and line.strip()
#                         ])
            
#             # Fallback: if no subtitles found, try transcript API
#             if not transcript_text:
#                 transcript_url = f"https://youtubetranscript.com/?server_vid2={video_id}"
#                 fallback_response = requests.get(transcript_url, timeout=10)
#                 if fallback_response.status_code == 200:
#                     # Simple text extraction from transcript service
#                     transcript_text = fallback_response.text[:10000]  # Limit length
                    
#     except Exception as e:
#         st.warning(f"Transcript extraction warning: {str(e)}")
    
#     return transcript_text[:50000]  # Limit transcript length for performance

# @st.cache_resource
# def load_embedding_model():
#     """Load embedding model once and cache it."""
#     st.info("🚀 Loading embedding model...")
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",  # Faster alternative
#         model_kwargs={'device': 'cpu'},  # Use CPU for faster loading
#         encode_kwargs={'normalize_embeddings': True}
#     )

# @st.cache_resource
# def load_llm():
#     """Load LLM once with optimized settings."""
#     st.info("🧠 Loading language model...")
#     return ChatHuggingFace(llm=HuggingFaceEndpoint(
#         model='HuggingFaceH4/zephyr-7b-beta',  # Faster alternative
#         task='text-generation',
#         max_new_tokens=512,
#         temperature=0.3,
#         timeout=30
#     ))

# def get_vectorstore_optimized(video_id: str, docs):
#     """Optimized vector store creation with batch processing."""
#     embedding_model = load_embedding_model()
    
#     # Check if vector store already exists for this video
#     metadata_file = f"rag_data/transcripts/{video_id}.pkl"
#     vector_store_path = "rag_data/chroma_db"
    
#     if os.path.exists(metadata_file):
#         st.info(f"📂 Using cached embeddings for video `{video_id}`.")
#         return Chroma(
#             persist_directory=vector_store_path,
#             embedding_function=embedding_model
#         )
    
#     # Create new vector store with progress tracking
#     st.info(f"🧠 Creating embeddings for video `{video_id}`...")
    
#     # Process in smaller batches for better performance
#     batch_size = 50
#     all_texts = [doc.page_content for doc in docs]
    
#     # Initialize Chroma with persistence
#     vector_store = Chroma(
#         embedding_function=embedding_model,
#         persist_directory=vector_store_path
#     )
    
#     # Add texts in batches with progress
#     progress_bar = st.progress(0)
#     for i in range(0, len(all_texts), batch_size):
#         batch_texts = all_texts[i:i + batch_size]
#         batch_metadatas = [{"video_id": video_id, "chunk_index": j} 
#                           for j in range(i, i + len(batch_texts))]
        
#         vector_store.add_texts(
#             texts=batch_texts, 
#             metadatas=batch_metadatas
#         )
        
#         # Update progress
#         progress = min((i + batch_size) / len(all_texts), 1.0)
#         progress_bar.progress(progress)
    
#     vector_store.persist()
    
#     # Save metadata
#     with open(metadata_file, "wb") as f:
#         pickle.dump({
#             "video_id": video_id, 
#             "chunks": len(docs),
#             "created_at": time.time()
#         }, f)
    
#     return vector_store

# def format_docs(docs):
#     """Efficient document formatting."""
#     return "\n\n".join([doc.page_content for doc in docs])

# def create_rag_chain(retriever, model):
#     """Create optimized RAG chain."""
#     prompt = PromptTemplate(
#         template=(
#             "You are a helpful assistant. Use ONLY the following context to answer.\n"
#             "If unsure, say 'I don't know.'\n\n"
#             "Context:\n{context}\n\nQuestion:\n{query}\nAnswer:"
#         ),
#         input_variables=["query", "context"]
#     )
    
#     # Simplified chain without unnecessary parallel processing
#     return (
#         {"context": retriever | format_docs, "query": RunnablePassthrough()}
#         | prompt
#         | model
#         | StrOutputParser()
#     )

# # -------------------------- OPTIMIZED Main Logic -------------------------- #

# if st.button("🚀 Ask Question"):
#     if not video_id or not question:
#         st.warning("Please provide both a YouTube video ID and a question.")
#     else:
#         try:
#             # Progress tracking
#             progress_bar = st.progress(0)
#             status_text = st.empty()
            
#             # Step 1: Load models in parallel (conceptually)
#             status_text.text("🔄 Loading AI models...")
#             model = load_llm()
#             progress_bar.progress(20)
            
#             # Step 2: Fetch transcript
#             status_text.text("🎧 Fetching transcript...")
#             full_text = fetch_transcript_optimized(video_id)
#             progress_bar.progress(40)
            
#             if not full_text.strip():
#                 st.error("No transcript found for this video. Try another one.")
#                 st.stop()
            
#             # Step 3: Split text efficiently
#             status_text.text("✂️ Processing text...")
#             splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=800,  # Smaller chunks for faster processing
#                 chunk_overlap=50,
#                 length_function=len
#             )
#             docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]
#             progress_bar.progress(60)
            
#             # Step 4: Get vector store
#             status_text.text("🔍 Setting up search...")
#             vector_store = get_vectorstore_optimized(video_id, docs)
#             retriever = vector_store.as_retriever(
#                 search_type="similarity",
#                 search_kwargs={
#                     "k": 3,  # Fewer results for faster processing
#                     "filter": {"video_id": video_id}
#                 }
#             )
#             progress_bar.progress(80)
            
#             # Step 5: Create and run RAG chain
#             status_text.text("🤖 Generating answer...")
#             rag_chain = create_rag_chain(retriever, model)
            
#             # Stream the response for better UX
#             response_container = st.empty()
#             full_response = ""
            
#             # Simulate streaming for better user experience
#             for chunk in rag_chain.stream(question):
#                 full_response += chunk
#                 response_container.markdown(f"### 🗣️ Answer\n{full_response}")
            
#             progress_bar.progress(100)
#             status_text.text("✅ Complete!")
            
#             # Show performance info
#             st.sidebar.markdown("### ⚡ Performance Tips")
#             st.sidebar.info("""
#             - Transcripts are cached for 1 hour
#             - Embeddings are stored permanently
#             - Using faster embedding model
#             - Smaller chunks for quicker processing
#             """)
            
#         except Exception as e:
#             st.error("An error occurred. Check details below.")
#             st.exception(e)
#             traceback.print_exc()

# # Add performance optimization tips
# with st.sidebar:
#     st.markdown("### 🚀 Performance Optimizations")
#     st.markdown("""
#     **Key Improvements:**
#     - Faster embedding model (MiniLM vs MPNet)
#     - Batched embedding generation
#     - Better transcript extraction
#     - Optimized chunk sizes
#     - Efficient caching strategy
#     - Progress tracking
#     - Response streaming
#     """)
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
load_dotenv()

# ------------------ PAGE CONFIG ------------------ #
st.set_page_config(
    page_title="🎥 YouTube RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------ #
st.markdown("""
    <style>
        /* Background Gradient */
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

    # Action Button
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
        timeout=30
    ))

def get_vectorstore_optimized(video_id: str, docs):
    embedding_model = load_embedding_model()
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
                response_container.markdown(f"### 🗣️ Answer\n{full_response}")

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
