import logging
from enum import Enum

class ModelType(Enum):
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    GROK = "grok"

class AIRouter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_model(self, research_depth):
        """
        Select appropriate model based on research depth
        Args:
            research_depth (str): 'shallow', 'medium', or 'deep'
        Returns:
            ModelType: The selected model type
        """
        try:
            if research_depth == "shallow":
                return ModelType.OPENAI  # Use GPT-4o for basic tasks
            elif research_depth == "medium":
                return ModelType.OPENAI  # Use GPT-4o for medium depth
            elif research_depth == "deep":
                return ModelType.DEEPSEEK  # Use DeepSeek for deep research
            else:
                return ModelType.OPENAI  # Default to OpenAI

        except Exception as e:
            self.logger.error(f"Error in model selection: {str(e)}")
            return ModelType.OPENAI  # Fallback to OpenAI