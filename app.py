from fastapi import FastAPI
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from langchain.schema import Document
import traceback  # <-- For full error trace
from dotenv import load_dotenv
load_dotenv()


app = FastAPI()

class QueryRequest(BaseModel):
    video_id: str
    question: str

@app.post("/ask")
def ask_question(req: QueryRequest):
    try:
        video_id = req.video_id
        question = req.question
        print(f"[INFO] Video ID: {video_id}, Question: {question}")

        # LLM setup
        llm = HuggingFaceEndpoint(
            model='meta-llama/Llama-3.1-8B-Instruct',
            task='text-generation'
        )
        model = ChatHuggingFace(llm=llm)
        print("[INFO] HuggingFace model loaded successfully.")

        # Fetch transcript
        transcript = YouTubeTranscriptApi().fetch(video_id,languages=['en', 'en-US'])
        full_text = " ".join([snippet.text for snippet in transcript])
        print(f"[INFO] Transcript fetched, length: {len(full_text)} characters.")

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(full_text)]
        print(f"[INFO] Transcript split into {len(docs)} chunks.")

        # Embeddings + Chroma
        embedding_model = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")
        vector_store = Chroma(
            embedding_function=embedding_model,
            persist_directory='rag',
            collection_name='youtube'
        )
        vector_store.add_documents(docs)
        print("[INFO] Documents added to Chroma vector store.")

        # Retriever
        retriever = vector_store.as_retriever(search_kwargs={'k': 4})
        print("[INFO] Retriever ready.")

        # Prompt + RAG chain
        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])

        prompt = PromptTemplate(
            template="""You are helpful assistant.
Provide answer to any question from the given context. If not present, just say "I don't know". Context: {context} Question: {query}""",
            input_variables=['query', 'context']
        )

        chain1 = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'query': RunnablePassthrough()
        })
        parser = StrOutputParser()
        chain2 = prompt | model | parser
        final_chain = chain1 | chain2

        answer = final_chain.invoke(question)
        print(f"[INFO] Answer generated: {answer}")

        return {"answer": answer}

    except Exception as e:
        # Print full error trace
        print("[ERROR] Exception occurred:")
        traceback.print_exc()
        return {"error": str(e)}
