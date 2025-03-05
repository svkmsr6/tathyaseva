import os
import json
import logging
from datetime import datetime
from crewai import Agent, Task, Crew
from openai import OpenAI
from typing import Dict, Any
from ai_router import ModelType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ResearchCrew:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.client = None

    def initialize_client(self):
        """Initialize the appropriate AI client based on model type and available API keys."""
        try:
            if self.model_type == ModelType.OPENAI:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "model": "gpt-4"
                }

            elif self.model_type == ModelType.GROK:
                api_key = os.environ.get("XAI_API_KEY")
                if not api_key:
                    raise ValueError("X.AI API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "base_url": "https://api.x.ai/v1",
                    "model": "grok-2-1212"
                }

            elif self.model_type == ModelType.LLAMA:
                # Fallback to OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "model": "gpt-4"
                }

            else:
                # Default to OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "model": "gpt-4"
                }

        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI client: {str(e)}")

    def create_researcher_agent(self) -> Agent:
        client_config = self.initialize_client()
        return Agent(
            role='Researcher',
            goal='Gather comprehensive information on the given topic',
            backstory='Expert researcher with vast knowledge in multiple domains',
            allow_delegation=True,
            verbose=True,
            **client_config
        )

    def create_fact_checker_agent(self) -> Agent:
        client_config = self.initialize_client()
        return Agent(
            role='Fact Checker',
            goal='Verify the accuracy of information and provide a veracity score',
            backstory='Experienced fact checker with attention to detail',
            allow_delegation=True,
            verbose=True,
            **client_config
        )

    def create_writer_agent(self) -> Agent:
        client_config = self.initialize_client()
        return Agent(
            role='Content Writer',
            goal='Create engaging and factual content',
            backstory='Professional writer specializing in accurate content creation',
            allow_delegation=True,
            verbose=True,
            **client_config
        )

    def create_editor_agent(self) -> Agent:
        client_config = self.initialize_client()
        return Agent(
            role='Editor',
            goal='Refine and polish content while maintaining accuracy',
            backstory='Senior editor with years of experience in content optimization',
            allow_delegation=True,
            verbose=True,
            **client_config
        )

    def run_fact_check(self, content: str) -> Dict[str, Any]:
        try:
            researcher = self.create_researcher_agent()
            fact_checker = self.create_fact_checker_agent()

            research_task = Task(
                description=f"Research the following content and provide evidence:\n{content}",
                agent=researcher,
                expected_output="Detailed research findings with supporting evidence and citations"
            )

            fact_check_task = Task(
                description="""
                Based on the research findings, analyze the veracity and provide the result in this exact JSON format:
                {
                    "score": <number between 0 and 100>,
                    "details": "Veracity Score: <same exact number as score> - <detailed explanation>"
                }
                IMPORTANT: The score in the details text MUST MATCH EXACTLY the score value.
                Make sure to respond ONLY with the JSON object, no additional text.
                """,
                agent=fact_checker,
                expected_output="JSON string containing score and details"
            )

            crew = Crew(
                agents=[researcher, fact_checker],
                tasks=[research_task, fact_check_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug(f"Raw fact check result: {result_str}")

            try:
                # Try direct JSON parsing first
                parsed_result = json.loads(result_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON from the text
                start_idx = result_str.find('{')
                end_idx = result_str.rfind('}')
                if start_idx != -1 and end_idx != -1:
                    json_str = result_str[start_idx:end_idx + 1]
                    parsed_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON object found in response")

            # Ensure score consistency
            score = float(parsed_result.get("score", 0))
            details = parsed_result.get("details", "Analysis failed")

            # Validate score consistency
            if not details.startswith(f"Veracity Score: {score}"):
                # If inconsistent, reformat the details to include the correct score
                details = f"Veracity Score: {score} - {details.split('-', 1)[1].strip() if '-' in details else details}"

            return {
                "score": score,
                "details": details
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"score": 0, "details": f"Failed to parse result: {str(e)}"}
        except Exception as e:
            logger.error(f"Fact check error: {e}")
            return {"score": 0, "details": f"Error during fact check: {str(e)}"}

    def generate_content(self, topic: str) -> Dict[str, Any]:
        try:
            researcher = self.create_researcher_agent()
            writer = self.create_writer_agent()
            editor = self.create_editor_agent()

            research_task = Task(
                description=f"Research the topic: {topic}\nProvide comprehensive findings including key points and references.",
                agent=researcher,
                expected_output="Comprehensive research findings with key points and references"
            )

            writing_task = Task(
                description="Write an engaging article based on the research findings. Include proper citations and maintain accuracy.",
                agent=writer,
                expected_output="Well-structured article with accurate information and engaging tone"
            )

            editing_task = Task(
                description="Polish and refine the content. Ensure proper formatting, flow, and maintain accuracy.",
                agent=editor,
                expected_output="Final polished content with proper formatting and flow"
            )

            crew = Crew(
                agents=[researcher, writer, editor],
                tasks=[research_task, writing_task, editing_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug(f"Raw content generation result: {result_str}")

            return {
                "content": result_str,
                "metadata": {
                    "topic": topic,
                    "model_used": self.model_type.value,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Content generation error: {e}")
            return {
                "content": f"Error generating content: {str(e)}",
                "metadata": {
                    "topic": topic,
                    "model_used": self.model_type.value,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }