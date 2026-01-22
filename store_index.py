from dotenv import load_dotenv
import os
from src.helper import DocumentProcessor, EmbeddingModel
from pinecone import Pinecone
from pinecone import ServerlessSpec 
from langchain_pinecone import PineconeVectorStore
from src.logger_config import get_logger

# Set up logger
logger = get_logger(__name__)

try:
    load_dotenv()
    logger.info("Environment variables loaded successfully")
except Exception as e:
    logger.error(f"Error loading environment variables: {str(e)}", exc_info=True)


class VectorStoreManager:
    """Class to manage Pinecone vector store operations."""
    
    def __init__(self, index_name: str = "medical-chatbot", data_path: str = "data/"):
        self.logger = get_logger(f"{__name__}.VectorStoreManager")
        self.index_name = index_name
        self.data_path = data_path
        self.pinecone_api_key = None
        self.openai_api_key = None
        self.pc = None
        self.index = None
        self.docsearch = None
        
        try:
            self.logger.info(f"Initializing VectorStoreManager with index_name={index_name}, data_path={data_path}")
            self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
            
            if not self.pinecone_api_key:
                error_msg = "PINECONE_API_KEY not found in environment variables"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY not found in environment variables"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            self._setup_environment()
            self._initialize_pinecone()
            self.logger.info("VectorStoreManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing VectorStoreManager: {str(e)}", exc_info=True)
            raise
    
    def _setup_environment(self):
        """Set up environment variables."""
        try:
            self.logger.info("Setting up environment variables...")
            os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            self.logger.info("Environment variables set up successfully")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {str(e)}", exc_info=True)
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone client and create index if needed."""
        try:
            self.logger.info(f"Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            self.logger.info("Pinecone client initialized successfully")
            
            self.logger.info(f"Checking if index '{self.index_name}' exists...")
            if not self.pc.has_index(self.index_name):
                self.logger.info(f"Index '{self.index_name}' does not exist, creating new index...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # Dimension of the embeddings
                    metric="cosine",  # Cosine similarity
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                self.logger.info(f"Index '{self.index_name}' created successfully")
            else:
                self.logger.info(f"Index '{self.index_name}' already exists")
            
            self.logger.info(f"Connecting to index '{self.index_name}'...")
            self.index = self.pc.Index(self.index_name)
            self.logger.info("Connected to index successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing Pinecone: {str(e)}", exc_info=True)
            raise
    
    def process_documents(self):
        """Process PDF documents and create vector store."""
        try:
            self.logger.info("Starting document processing...")
            
            self.logger.info("Initializing DocumentProcessor...")
            processor = DocumentProcessor()
            
            self.logger.info("Initializing EmbeddingModel...")
            embedding_model = EmbeddingModel()
            
            self.logger.info(f"Loading PDF files from {self.data_path}...")
            extracted_data = processor.load_pdf_files(data=self.data_path)
            self.logger.info(f"Loaded {len(extracted_data)} documents")
            
            self.logger.info("Filtering documents to minimal metadata...")
            filter_data = processor.filter_to_minimal_docs(extracted_data)
            self.logger.info(f"Filtered to {len(filter_data)} documents")
            
            self.logger.info("Splitting documents into chunks...")
            text_chunks = processor.text_split(filter_data)
            self.logger.info(f"Created {len(text_chunks)} text chunks")
            
            self.logger.info("Downloading embeddings...")
            embedding = embedding_model.download_embeddings()
            self.logger.info("Embeddings downloaded successfully")
            
            self.logger.info(f"Creating vector store in Pinecone index '{self.index_name}'...")
            self.docsearch = PineconeVectorStore.from_documents(
                documents=text_chunks,
                embedding=embedding,
                index_name=self.index_name
            )
            self.logger.info("Vector store created successfully")
            
            return self.docsearch
            
        except FileNotFoundError as e:
            self.logger.error(f"File not found error: {str(e)}", exc_info=True)
            raise
        except ValueError as e:
            self.logger.error(f"Value error: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            self.logger.error(f"Error processing documents: {str(e)}", exc_info=True)
            raise


# Create instance and process documents using OOP
try:
    logger.info("Creating VectorStoreManager instance...")
    vector_store_manager = VectorStoreManager()
    logger.info("VectorStoreManager instance created successfully")
except Exception as e:
    logger.error(f"Error creating VectorStoreManager instance: {str(e)}", exc_info=True)
    vector_store_manager = None

# Maintain backward compatibility variables
try:
    logger.info("Setting up backward compatibility variables...")
    
    if vector_store_manager is None:
        raise RuntimeError("VectorStoreManager instance is not available")
    
    PINECONE_API_KEY = vector_store_manager.pinecone_api_key
    OPENAI_API_KEY = vector_store_manager.openai_api_key
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    processor = DocumentProcessor()
    embedding_model = EmbeddingModel()
    
    logger.info("Loading and processing documents...")
    extracted_data = processor.load_pdf_files(data='data/')
    filter_data = processor.filter_to_minimal_docs(extracted_data)
    text_chunks = processor.text_split(filter_data)
    embedding = embedding_model.download_embeddings()

    pinecone_api_key = PINECONE_API_KEY
    pc = vector_store_manager.pc
    index_name = vector_store_manager.index_name
    index = vector_store_manager.index

    logger.info("Creating vector store...")
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        embedding=embedding,
        index_name=index_name
    )
    logger.info("Backward compatibility variables set up successfully")
    
except RuntimeError as e:
    logger.error(f"Runtime error: {str(e)}", exc_info=True)
except Exception as e:
    logger.error(f"Error setting up backward compatibility variables: {str(e)}", exc_info=True)
