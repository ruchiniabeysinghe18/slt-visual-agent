"""
PDF Document Processor
Handles PDF processing, embeddings, and retrieval
"""
import os
import PyPDF2
import time
from typing import List, Dict
import pathlib
from utils.logger import get_debug_logger
from utils.prompts import context_filter_prompt_template
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS

import chromadb


logger = get_debug_logger(
    "pdf_kb", pathlib.Path.joinpath(pathlib.Path(__file__).parent.resolve(), "../logs/pdf_kb.log")
)

load_dotenv()
import os
from langchain_core.output_parsers import StrOutputParser

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(
                model='gpt-4.1-mini',
                api_key=OPENAI_API_KEY,
                max_tokens=2048,
                temperature=0,
            )


class PDFProcessor:
    """
    Handles PDF document processing operations
    """
    
    def __init__(self, upload_dir: str = "uploaded_pdfs"):
        """
        Initialize the PDF processor.
        
        Args:
            upload_dir: Directory to store uploaded PDFs
        """
        self.upload_dir = upload_dir
        
        # Create upload directory if it doesn't exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            logger.info(f"Created PDF upload directory: {upload_dir}")
    
    def save_fastapi_pdf(self, uploaded_file, filename: str) -> str:
        """
        Save an uploaded FastAPI PDF file to disk.

        Args:
            uploaded_file: The PDF file uploaded through FastAPI
            filename: Name of the file
            
        Returns:
            str: Path to the saved file
        """
        file_path = os.path.join(self.upload_dir, filename)
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Write file to disk
            with open(file_path, "wb") as f:
                content = uploaded_file.read()
                f.write(content)

            logger.info(f"Saved PDF file: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error saving PDF file: {str(e)}")
            raise e
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.info(f"Processing PDF with {num_pages} pages")
              

                # Extract text from each page
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                logger.info(f"Complete extracted text from PDF: {text}")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise e
    
    


class DocumentQAModel:
    """
    QA model for document retrieval and embedding
    """

    def __init__(self, api_key: str):
        """
        Args:
            api_key: API key (reserved for future use)
        """
        self.api_key = api_key
        
        # Initialize embeddings
        # self.embeddings = GoogleGenerativeAIEmbeddings(
        #     model="models/embedding-001",
        #     google_api_key=self.api_key
        # )

        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
    

        logger.info("DocumentQAModel initialized successfully")
    
    
    def create_vectorstore_from_text(self, document_text: str, org_id: str,pdf_name: str):
        """
        Create a vectorstore from document text.

        Args:
            document_text: The text extracted from the PDF
            org_id: organization identifier
            pdf_name: Name of the PDF document
        Returns:
            collection: ChromaDB collection object containing the embedded text chunks
        """
        try:
            start_time = time.time()
            
            # Chunk text
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            text_chunks = splitter.split_text(document_text)
            
            # Create ChromaDB collection
            collection_name = f"{org_id}_documents"
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            
            # Embed chunks and store in ChromaDB
            for i, chunk in enumerate(text_chunks):
                embedding = self.embeddings.embed_query(chunk)
                collection.add(
                    ids=[f"{pdf_name}_chunk_{i}"],
                    documents=[chunk],
                    embeddings=[embedding],
                    metadatas=[{"source" : pdf_name}]
                )
            
    


            end_time = time.time()
            logger.info(f"Created collection in {end_time - start_time:.2f} seconds")

            return collection
            
        except Exception as e:
            logger.error(f"Error creating chroma collection: {str(e)}")
            raise e
    
    def retrieve_context_chunks(self, org_id: str, query: str, k: int) -> List[str]:
        """Retrieve top_k relevant chunks from ChromaDB for a given query"""
        try:
            logger.info(f"Searching for {k} chunks similar to: {query}")  
            
            collection_name = f"{org_id}_documents"
            collection = self.chroma_client.get_or_create_collection(name=collection_name)
            
            query_emb = self.embeddings.embed_query(query)
            results = collection.query(query_embeddings=[query_emb], n_results=k)
            if results["documents"]:
                retrieved_docs = results["documents"][0]
                logger.info(f"Retrieved {len(retrieved_docs)} documents for query: {query}")  
                # log the full retrieved document
                for i, doc in enumerate(retrieved_docs):
                    logger.info(f"Document {i+1}: {doc}")  
                return retrieved_docs
            else:
                logger.info(f"No documents found for query: {query}")
                return []

            
        except Exception as e:
            logger.error(f"Error retrieving context chunks: {str(e)}")
            return []


class DocumentKB:
    """
    Document Knowledge Base — manages PDF storage and retrieval
    """

    def __init__(self, mongo_manager, api_key: str):
        """
        Args:
            mongo_manager: MongoDB manager instance
            api_key: API key (passed through to DocumentQAModel)
        """
        self.mongo_manager = mongo_manager
        self.qa_model = DocumentQAModel(api_key)
        logger.info("DocumentKB initialized")

    def store_pdf_document(self, org_id: str, pdf_text: str, pdf_name: str, metadata: dict = None):
        """
        Store PDF document chunks and embeddings in the database.
        """
        try:

            # Create vectorstore and get chunks
            collection = self.qa_model.create_vectorstore_from_text(pdf_text,org_id,pdf_name)
            

            # Store document metadata in MongoDB (no need to store text_chunks)
            doc_data = {
                "org_id": org_id,
                "document_name": pdf_name,
                "document_type": "pdf",
                "metadata": metadata or {},
                "created_at": time.time()
            }
           
            # Store in MongoDB
            result = self.mongo_manager.insert_one(doc_data)
            logger.info(f"Stored PDF '{pdf_name}' in MongoDB + ChromaDB")

        
            
            return result, collection
            
        except Exception as e:
            logger.error(f"Error storing PDF document: {str(e)}")
            raise e
    
    def get_context_for_question(self, org_id: str, question: str) -> str:
        """
        Retrieve relevant context from the organization's PDF documents.
        """
        try:
            logger.info(f"RAG QUERY: {question}")  

            # Retrieve relevant chunks
            relevant_chunks = self.qa_model.retrieve_context_chunks(org_id, question, k=10)
            logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks")  
            context = "\n\n".join(relevant_chunks)

            return context

        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return ""
    