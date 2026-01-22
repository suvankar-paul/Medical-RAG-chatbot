from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from src.logger_config import get_logger
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set up logger
logger = get_logger(__name__)

try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}", exc_info=True)


class DocumentProcessor:
    """Class to handle document loading, filtering, and text splitting operations."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.DocumentProcessor")
        self.logger.info("DocumentProcessor initialized")
    
    def load_pdf_files(self, data):
        """Extract text from PDF files."""
        try:
            self.logger.info(f"Loading PDF files from: {data}")
            
            if not os.path.exists(data):
                error_msg = f"Data path does not exist: {data}"
                self.logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            loader = DirectoryLoader(
                data,
                glob="*.pdf",
                loader_cls=PyPDFLoader
            )
            
            self.logger.debug("Loading documents...")
            documents = loader.load()
            
            if not documents:
                error_msg = f"No PDF files found in directory: {data}"
                self.logger.warning(error_msg)
                raise ValueError(error_msg)
            
            self.logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}")
            raise
        except ValueError as e:
            self.logger.error(f"Value error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error loading PDF files: {str(e)}", exc_info=True)
            raise
    
    def filter_to_minimal_docs(self, docs: List[Document]) -> List[Document]:
        """
        Given a list of Document objects, return a new list of Document objects
        containing only 'source' in metadata and the original page_content.
        """
        try:
            self.logger.info(f"Filtering {len(docs)} documents to minimal metadata")
            
            if not docs:
                self.logger.warning("Empty document list provided for filtering")
                return []
            
            minimal_docs: List[Document] = []
            for doc in docs:
                try:
                    src = doc.metadata.get("source")
                    minimal_docs.append(
                        Document(
                            page_content=doc.page_content,
                            metadata={"source": src}
                        )
                    )
                except Exception as e:
                    self.logger.warning(f"Error processing document: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully filtered to {len(minimal_docs)} documents")
            return minimal_docs
            
        except Exception as e:
            self.logger.error(f"Error filtering documents: {str(e)}", exc_info=True)
            raise
    
    def text_split(self, minimal_docs):
        """Split documents into text chunks."""
        try:
            self.logger.info(f"Splitting {len(minimal_docs)} documents into chunks")
            
            if not minimal_docs:
                self.logger.warning("Empty document list provided for splitting")
                return []
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=20,
            )
            
            self.logger.debug("Splitting documents...")
            texts_chunk = text_splitter.split_documents(minimal_docs)
            
            self.logger.info(f"Successfully created {len(texts_chunk)} text chunks")
            return texts_chunk
            
        except Exception as e:
            self.logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
            raise


class EmbeddingModel:
    """Class to handle embedding model operations."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.logger = get_logger(f"{__name__}.EmbeddingModel")
        self.model_name = model_name
        self.embeddings = None
        self.logger.info(f"EmbeddingModel initialized with model: {model_name}")
    
    def download_embeddings(self):
        """
        Download and return the HuggingFace embeddings model.
        """
        try:
            if self.embeddings is not None:
                self.logger.info("Embeddings already loaded, returning existing instance")
                return self.embeddings
            
            self.logger.info(f"Downloading embeddings model: {self.model_name}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name
            )
            self.logger.info("Embeddings model downloaded successfully")
            return self.embeddings
            
        except Exception as e:
            self.logger.error(f"Error downloading embeddings: {str(e)}", exc_info=True)
            raise


def load_pdf_files(data):
    """Wrapper function for backward compatibility."""
    try:
        logger.info(f"load_pdf_files called with data path: {data}")
        processor = DocumentProcessor()
        return processor.load_pdf_files(data)
    except Exception as e:
        logger.error(f"Error in load_pdf_files wrapper: {str(e)}", exc_info=True)
        raise


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Wrapper function for backward compatibility."""
    try:
        logger.info(f"filter_to_minimal_docs called with {len(docs)} documents")
        processor = DocumentProcessor()
        return processor.filter_to_minimal_docs(docs)
    except Exception as e:
        logger.error(f"Error in filter_to_minimal_docs wrapper: {str(e)}", exc_info=True)
        raise


def text_split(minimal_docs):
    """Wrapper function for backward compatibility."""
    try:
        logger.info(f"text_split called with {len(minimal_docs)} documents")
        processor = DocumentProcessor()
        return processor.text_split(minimal_docs)
    except Exception as e:
        logger.error(f"Error in text_split wrapper: {str(e)}", exc_info=True)
        raise


def download_embeddings():
    """Wrapper function for backward compatibility."""
    try:
        logger.info("download_embeddings called")
        embedding_model = EmbeddingModel()
        return embedding_model.download_embeddings()
    except Exception as e:
        logger.error(f"Error in download_embeddings wrapper: {str(e)}", exc_info=True)
        raise

try:
    logger.info("Downloading embeddings for backward compatibility...")
    embedding = download_embeddings()
    logger.info("Embeddings downloaded successfully")
except Exception as e:
    logger.error(f"Error downloading embeddings at module level: {str(e)}", exc_info=True)
    embedding = None
