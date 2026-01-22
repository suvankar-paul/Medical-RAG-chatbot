from flask import Flask, render_template, jsonify, request
from src.helper import EmbeddingModel
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import PromptManager
from src.logger_config import get_logger
import os
import traceback

# Set up logger
logger = get_logger(__name__)


class RAGChain:
    """Class to manage RAG (Retrieval-Augmented Generation) chain operations."""
    
    def __init__(self, index_name: str = "medical-chatbot", model_name: str = "gpt-4o", k: int = 3):
        self.logger = get_logger(f"{__name__}.RAGChain")
        self.index_name = index_name
        self.model_name = model_name
        self.k = k
        self.embeddings = None
        self.docsearch = None
        self.retriever = None
        self.chat_model = None
        self.prompt_manager = None
        self.question_answer_chain = None
        self.rag_chain = None
        
        try:
            self.logger.info(f"Initializing RAGChain with index_name={index_name}, model={model_name}, k={k}")
            self._initialize()
            self.logger.info("RAGChain initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize RAGChain: {str(e)}", exc_info=True)
            raise Exception(f"Failed to initialize RAGChain: {str(e)}") from e
    
    def _initialize(self):
        """Initialize embeddings, vector store, and RAG chain."""
        try:
            self.logger.info("Downloading embeddings...")
            embedding_model = EmbeddingModel()
            self.embeddings = embedding_model.download_embeddings()
            self.logger.info("Embeddings downloaded successfully")
            
            self.logger.info(f"Connecting to Pinecone index: {self.index_name}")
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            self.logger.info("Connected to Pinecone index successfully")
            
            self.logger.info(f"Creating retriever with k={self.k}")
            self.retriever = self.docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k}
            )
            self.logger.info("Retriever created successfully")
            
            self.logger.info(f"Initializing chat model: {self.model_name}")
            self.chat_model = ChatOpenAI(model=self.model_name)
            self.logger.info("Chat model initialized successfully")
            
            self.logger.info("Setting up prompt manager...")
            self.prompt_manager = PromptManager()
            prompt = self.prompt_manager.get_prompt()
            self.logger.info("Prompt manager set up successfully")
            
            self.logger.info("Creating question-answer chain...")
            self.question_answer_chain = create_stuff_documents_chain(self.chat_model, prompt)
            self.logger.info("Question-answer chain created successfully")
            
            self.logger.info("Creating RAG chain...")
            self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
            self.logger.info("RAG chain created successfully")
            
        except Exception as e:
            self.logger.error(f"Error during RAGChain initialization: {str(e)}", exc_info=True)
            raise Exception(f"Error during RAGChain initialization: {str(e)}") from e
    
    def invoke(self, input_text: str):
        """Invoke the RAG chain with input text."""
        try:
            self.logger.debug(f"Invoking RAG chain with input: {input_text[:100]}...")
            
            if not input_text or not isinstance(input_text, str):
                error_msg = "Input text must be a non-empty string"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            if self.rag_chain is None:
                error_msg = "RAG chain is not initialized"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            self.logger.info("Invoking RAG chain...")
            response = self.rag_chain.invoke({"input": input_text})
            self.logger.info("RAG chain invocation completed successfully")
            self.logger.debug(f"Response received: {response.get('answer', '')[:100]}...")
            
            return response
            
        except ValueError as e:
            self.logger.warning(f"Validation error in RAGChain.invoke: {str(e)}")
            raise ValueError(f"Validation error in RAGChain.invoke: {str(e)}") from e
        except RuntimeError as e:
            self.logger.error(f"Runtime error in RAGChain.invoke: {str(e)}")
            raise RuntimeError(f"Runtime error in RAGChain.invoke: {str(e)}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error in RAGChain.invoke: {str(e)}", exc_info=True)
            raise Exception(f"Unexpected error in RAGChain.invoke: {str(e)}") from e


class MedicalChatbot:
    """Main Flask application class for the Medical Chatbot."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.MedicalChatbot")
        self.app = Flask(__name__)
        self.rag_chain = None
        
        try:
            self.logger.info("Initializing MedicalChatbot...")
            self._setup_environment()
            self._initialize_rag_chain()
            self._register_routes()
            self.logger.info("MedicalChatbot initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize MedicalChatbot: {str(e)}", exc_info=True)
            raise Exception(f"Failed to initialize MedicalChatbot: {str(e)}") from e
    
    def _setup_environment(self):
        """Set up environment variables."""
        try:
            self.logger.info("Loading environment variables...")
            load_dotenv()
            PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
            OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
            
            if not PINECONE_API_KEY:
                self.logger.warning("PINECONE_API_KEY not found in environment")
            else:
                self.logger.debug("PINECONE_API_KEY loaded successfully")
            
            if not OPENAI_API_KEY:
                self.logger.warning("OPENAI_API_KEY not found in environment")
            else:
                self.logger.debug("OPENAI_API_KEY loaded successfully")
            
            os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.logger.info("Environment variables set up successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up environment: {str(e)}", exc_info=True)
            raise Exception(f"Error setting up environment: {str(e)}") from e
    
    def _initialize_rag_chain(self):
        """Initialize the RAG chain."""
        try:
            self.logger.info("Initializing RAG chain...")
            self.rag_chain = RAGChain()
            self.logger.info("RAG chain initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing RAG chain: {str(e)}", exc_info=True)
            raise Exception(f"Error initializing RAG chain: {str(e)}") from e
    
    def _register_routes(self):
        """Register Flask routes."""
        try:
            self.logger.info("Registering Flask routes...")
            
            @self.app.route("/")
            def index():
                try:
                    self.logger.info("Index route accessed")
                    return render_template('chat.html')
                except Exception as e:
                    self.logger.error(f"Error in index route: {str(e)}", exc_info=True)
                    return "An error occurred while loading the page.", 500
            
            @self.app.route("/get", methods=["GET", "POST"])
            def chat():
                try:
                    msg = request.form.get("msg", "")
                    self.logger.info(f"Chat request received: {msg[:100]}...")
                    
                    if not msg:
                        self.logger.warning("Empty message received")
                        return "Please provide a message.", 400
                    
                    self.logger.debug(f"Processing message: {msg}")
                    response = self.rag_chain.invoke(msg)
                    answer = response.get("answer", "I'm sorry, I couldn't generate a response.")
                    
                    self.logger.info("Response generated successfully")
                    self.logger.debug(f"Response: {answer[:100]}...")
                    
                    return str(answer)
                    
                except ValueError as e:
                    self.logger.warning(f"Validation error in chat route: {str(e)}")
                    return f"Invalid input: {str(e)}", 400
                except RuntimeError as e:
                    self.logger.error(f"Runtime error in chat route: {str(e)}", exc_info=True)
                    return f"System error: {str(e)}", 500
                except Exception as e:
                    self.logger.error(f"Unexpected error in chat route: {str(e)}", exc_info=True)
                    return f"An error occurred: {str(e)}", 500
            
            self.logger.info("Flask routes registered successfully")
            
        except Exception as e:
            self.logger.error(f"Error registering routes: {str(e)}", exc_info=True)
            raise Exception(f"Error registering routes: {str(e)}") from e
    
    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = True):
        """Run the Flask application."""
        try:
            self.logger.info(f"Starting Flask application on {host}:{port} (debug={debug})")
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            self.logger.error(f"Error running Flask application: {str(e)}", exc_info=True)
            raise Exception(f"Error running Flask application: {str(e)}") from e


# Create Flask app instance for backward compatibility
try:
    logger.info("Creating MedicalChatbot instance...")
    medical_chatbot = MedicalChatbot()
    app = medical_chatbot.app
    logger.info("MedicalChatbot instance created successfully")
except Exception as e:
    logger.error(f"Failed to create MedicalChatbot instance: {str(e)}", exc_info=True)
    medical_chatbot = None
    app = None
    raise Exception(f"Failed to create MedicalChatbot instance: {str(e)}") from e

# Maintain backward compatibility variables
try:
    logger.info("Setting up backward compatibility variables...")
    load_dotenv()
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    embedding_model = EmbeddingModel()
    embeddings = embedding_model.download_embeddings()

    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )

    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    chatModel = ChatOpenAI(model="gpt-4o")
    prompt_manager = PromptManager()
    prompt = prompt_manager.get_prompt()

    question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    logger.info("Backward compatibility variables set up successfully")
    
except Exception as e:
    logger.error(f"Error setting up backward compatibility variables: {str(e)}", exc_info=True)
    # Continue execution even if backward compatibility setup fails
    logger.warning("Continuing without backward compatibility variables")


if __name__ == '__main__':
    try:
        logger.info("Starting application...")
        medical_chatbot.run(host="0.0.0.0", port=8080, debug=True)
    except Exception as e:
        logger.error(f"Fatal error starting application: {str(e)}", exc_info=True)
        raise Exception(f"Fatal error starting application: {str(e)}") from e
