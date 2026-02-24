import os
from typing import List, Dict
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
        self.chain = None
        
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
            chunk_overlap=Config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
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
        
        # Create the chain using the new method
        self._create_chain()
        print("RAG chain created successfully")
    
    def _create_chain(self):
        """Create the retrieval chain using the new LangChain approach"""
        # Define the prompt template
        system_prompt = (
            "You are a helpful assistant for application packaging questions. "
            "Use the following context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Keep the answer concise and focused on silent installation parameters.\n\n"
            "Context: {context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])
        
        # Create the document combining chain
        combine_docs_chain = create_stuff_documents_chain(self.llm, prompt)
        
        # Create the retrieval chain
        self.chain = create_retrieval_chain(self.retriever, combine_docs_chain)
    
    def search_documents(self, query: str) -> Dict:
        """Search documents using RAG"""
        if not self.chain:
            self.load_documents()
            if not self.chain:
                return {
                    "answer": "No documents loaded. Please add files to the data folder.",
                    "sources": []
                }
        
        # Invoke the chain
        result = self.chain.invoke({"input": query})
        
        # Extract sources from the result
        sources = []
        if 'context' in result:
            for doc in result['context']:
                sources.append({
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata
                })
        
        return {
            "answer": result['answer'],
            "sources": sources
        }