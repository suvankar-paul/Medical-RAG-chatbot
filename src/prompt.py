from langchain_core.prompts import ChatPromptTemplate
from src.logger_config import get_logger

# Set up logger
logger = get_logger(__name__)


class PromptManager:
    """Class to manage prompt templates for the medical chatbot."""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.PromptManager")
        
        try:
            self.logger.info("Initializing PromptManager...")
            self.system_prompt = (
                "You are an Medical assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
            self.prompt = None
            self._create_prompt()
            self.logger.info("PromptManager initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing PromptManager: {str(e)}", exc_info=True)
            raise
    
    def _create_prompt(self):
        """Create the chat prompt template."""
        try:
            self.logger.debug("Creating chat prompt template...")
            self.prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt),
                    ("human", "{input}"),
                ]
            )
            self.logger.debug("Chat prompt template created successfully")
        except Exception as e:
            self.logger.error(f"Error creating prompt template: {str(e)}", exc_info=True)
            raise
    
    def get_system_prompt(self):
        """Get the system prompt string."""
        try:
            self.logger.debug("Retrieving system prompt")
            return self.system_prompt
        except Exception as e:
            self.logger.error(f"Error retrieving system prompt: {str(e)}", exc_info=True)
            raise
    
    def get_prompt(self):
        """Get the ChatPromptTemplate."""
        try:
            self.logger.debug("Retrieving prompt template")
            if self.prompt is None:
                error_msg = "Prompt template is not initialized"
                self.logger.error(error_msg)
                raise RuntimeError(error_msg)
            return self.prompt
        except RuntimeError as e:
            self.logger.error(f"Runtime error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving prompt template: {str(e)}", exc_info=True)
            raise


# Create instance for backward compatibility
try:
    logger.info("Creating PromptManager instance for backward compatibility...")
    prompt_manager = PromptManager()
    system_prompt = prompt_manager.get_system_prompt()
    prompt = prompt_manager.get_prompt()
    logger.info("PromptManager instance created successfully")
except Exception as e:
    logger.error(f"Error creating PromptManager instance: {str(e)}", exc_info=True)
    prompt_manager = None
    system_prompt = None
    prompt = None
