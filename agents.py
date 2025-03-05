import os
import json
import logging
import re
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
                    "model": "gpt-4o"  # Using GPT-4o for Shallow/Medium depth
                }

            elif self.model_type == ModelType.DEEPSEEK:
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    raise ValueError("DeepSeek API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "base_url": "https://api.deepseek.com/v1",
                    "model": "deepseek-research-1"
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

            else:
                # Default to OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                return {
                    "api_key": api_key,
                    "model": "gpt-4o"
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
                # Look for JSON pattern
                json_pattern = r'\{[^}]+\}'
                json_match = re.search(json_pattern, result_str)

                if json_match:
                    json_str = json_match.group(0)
                    logger.debug(f"Extracted JSON string: {json_str}")
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

            research_task = Task(
                description=f"Research the topic: {topic}\nProvide comprehensive findings including key points and references.",
                agent=researcher,
                expected_output="Comprehensive research findings with key points and references"
            )

            writing_task = Task(
                description="""
                Write an engaging article based on the research findings.
                You must format your response as a valid JSON object with this exact structure:
                {
                    "content": "your article content here",
                    "metadata": {
                        "word_count": number
                    }
                }

                STRICT REQUIREMENTS:
                1. Only respond with the JSON object exactly as specified
                2. No text before or after the JSON
                3. Use double quotes for all strings
                4. Do not add any explanations or comments
                5. The response must start with '{' and end with '}'
                """,
                agent=writer,
                expected_output="Valid JSON string containing article"
            )

            crew = Crew(
                agents=[researcher, writer],
                tasks=[research_task, writing_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug("Raw content generation result:")
            logger.debug(result_str)

            try:
                # Step 1: Clean the string
                result_str = result_str.strip()
                logger.debug("After initial cleaning:")
                logger.debug(result_str)

                # Step 2: Extract JSON object
                start_idx = result_str.find('{')
                end_idx = result_str.rfind('}')

                if start_idx == -1 or end_idx == -1:
                    raise ValueError("No JSON object found in response")

                json_str = result_str[start_idx:end_idx + 1]
                logger.debug("Extracted JSON string:")
                logger.debug(json_str)

                # Step 3: Parse JSON
                parsed_result = json.loads(json_str)
                logger.debug("Successfully parsed JSON")

                # Step 4: Validate structure
                if not isinstance(parsed_result, dict):
                    raise ValueError("Parsed result is not a dictionary")
                if "content" not in parsed_result:
                    raise ValueError("Missing 'content' field in response")

                # Step 5: Prepare response
                return {
                    "content": parsed_result["content"],
                    "metadata": {
                        "topic": topic,
                        "model_used": self.model_type.value,
                        "timestamp": datetime.now().isoformat(),
                        "word_count": len(parsed_result["content"].split())
                    }
                }

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Problematic string: {result_str}")
                return {
                    "content": "Error: Failed to parse content generation response",
                    "metadata": {
                        "topic": topic,
                        "model_used": self.model_type.value,
                        "timestamp": datetime.now().isoformat(),
                        "error": f"JSON parsing error: {str(e)}"
                    }
                }
            except Exception as e:
                logger.error(f"Content generation error: {str(e)}")
                return {
                    "content": "Error: Content generation failed",
                    "metadata": {
                        "topic": topic,
                        "model_used": self.model_type.value,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e)
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