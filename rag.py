from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_classic.document_loaders import UnstructuredURLLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

COLLECTION_NAME = "real_estate_listings"
CHUNK_SIZE = 1000
VECTOR_STORE_PATH = Path(__file__).parent / "resources/vector_store"

llm,vector_store = None, None

def initialize_components():
    global llm,vector_store
    if not llm:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9,max_tokens=500)
    if not vector_store:
        ef = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"trust_remote_code": True}
                )

        vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_PATH),
            collection_name=COLLECTION_NAME,
            embedding_function=ef
        )

def process_urls(urls):
    yield 'Initiallizing components...'
    initialize_components()
    vector_store.reset_collection()

    yield 'Loading data from URLs...'
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    yield 'Splitting documents and adding to vector store...'
    text_splitter = RecursiveCharacterTextSplitter(
                            separators=["\n\n", "\n", " ", ""], 
                                chunk_size=CHUNK_SIZE, 
                                chunk_overlap=200
                                )
    docs = text_splitter.split_documents(data)

    yield f'Adding {len(docs)} documents to vector store...'
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs,ids=uuids)

    yield 'Added documents to vector store successfully.'

def generate_answer(question):
    if not vector_store:
        raise RuntimeError("Vector store is not initialized. Please process URLs first.")
    # yield f'Generating answer for question: {question}'
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(
        """You are an expert real estate assistant. Use the following pieces of information to answer the question at the end.
        
        Context: {context}
        
        Question: {question}"""
    )


    qa_chain = (
        {"context": retriever,"question": RunnablePassthrough()}
            | prompt
            | llm
    )
         
    result = qa_chain.invoke(question)

    docs = retriever.invoke(question)
    sources = [doc.metadata for doc in docs]

    return result.content, sources

if __name__ == "__main__":
    urls = ['https://www.homes.com/news/roundup-homeownership-rates-stay-low-unemployment-stable-in-most-states-and-fed-bank-presidents-reconfirmed/1749518220/']
    process_urls(urls)
    # results = vector_store.similarity_search("What are the top real estate markets to watch in 2026?", k=2)
    result,sources = generate_answer("What is the rate of home ownership in people 45 to 55 years old?")

    print("Answer:", result)
    print("Sources:", sources)