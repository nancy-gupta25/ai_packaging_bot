import os
from typing import List, Dict
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from config import Config

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        self.llm = ChatOpenAI(
            model_name=Config.OPENAI_MODEL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY
        )
        self.vectorstore = None
        self.retriever = None
        
    def load_documents(self):
        """Load documents from data directory"""
        documents = []
        
        if os.path.exists(Config.DATA_DIR):
            for file in os.listdir(Config.DATA_DIR):
                file_path = os.path.join(Config.DATA_DIR, file)
                try:
                    if file.endswith('.txt') or file.endswith('.ps1'):
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                        print(f"Loaded: {file}")
                    elif file.endswith('.pdf'):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        print(f"Loaded: {file}")
                    elif file.endswith('.docx'):
                        loader = Docx2txtLoader(file_path)
                        documents.extend(loader.load())
                        print(f"Loaded: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
        
        if not documents:
            print("No documents found in data folder")
            return
        
        print(f"Loaded {len(documents)} documents")
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks")
        
        # Create vectorstore
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=Config.VECTORSTORE_DIR
        )
        self.vectorstore.persist()
        
        # Create retriever
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": Config.TOP_K_RESULTS}
        )
        print("Vectorstore and retriever created successfully")
    
    def search_documents(self, query: str) -> Dict:
        """Search documents using RAG"""
        if not self.retriever:
            self.load_documents()
            if not self.retriever:
                return {
                    "answer": "No documents loaded. Please add files to the data folder.",
                    "sources": []
                }
        
        # Get relevant documents
        docs = self.retriever.get_relevant_documents(query)
        
        if not docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": []
            }
        
        # Prepare context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""You are a helpful assistant for application packaging questions.
Use the following context to answer the question about silent installation parameters.
If you don't know the answer, say you don't know.

Context:
{context}

Question: {query}

Answer: """
        
        # Get response from LLM
        response = self.llm.predict(prompt)
        
        return {
            "answer": response,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                }
                for doc in docs
            ]
        }
