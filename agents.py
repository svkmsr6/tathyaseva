import os
import json
import logging
from datetime import datetime
from enum import Enum
from crewai import Agent, Task, Crew

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TaskStatus(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"

class ResearchCrew:
    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found in environment variables")

    def create_agent(self, role, goal, backstory):
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=True,
            verbose=True,
            api_key=self.api_key,
            model="gpt-4o"  # Using GPT-4o for all tasks
        )

    def _extract_json(self, text: str) -> dict:
        """Extract and parse JSON from text with enhanced error handling."""
        try:
            logger.debug("=== Raw Response for JSON Extraction ===")
            logger.debug(f"Response type: {type(text)}")
            logger.debug(f"Response length: {len(text)}")
            logger.debug(f"First 100 characters: {repr(text[:100])}")
            logger.debug(f"Last 100 characters: {repr(text[-100:])}")

            # Find JSON boundaries
            start_idx = text.find('{')
            end_idx = text.rfind('}')

            logger.debug(f"JSON start index: {start_idx}")
            logger.debug(f"JSON end index: {end_idx}")

            if start_idx == -1 or end_idx == -1:
                logger.error("No JSON object found in response")
                logger.error(f"Raw response: {repr(text)}")
                return None

            # Extract the JSON string
            json_str = text[start_idx:end_idx + 1]
            logger.debug(f"Extracted JSON string: {repr(json_str)}")

            try:
                parsed_result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON: {parsed_result}")
                return parsed_result
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
                logger.error(f"Problematic JSON: {repr(json_str)}")
                return None

        except Exception as e:
            logger.error(f"JSON extraction error: {str(e)}")
            logger.error(f"Problematic text: {repr(text)}")
            return None

    def generate_factual_content(self, topic: str) -> dict:
        """Generate factual, verified content with status tracking."""
        try:
            # Create content writer agent
            writer = self.create_agent(
                role='Content Writer',
                goal='Create engaging and factual content within 500 words',
                backstory='Professional writer specializing in concise, accurate content creation'
            )

            # Create fact checker agent
            fact_checker = self.create_agent(
                role='Fact Checker',
                goal='Verify content accuracy and suggest improvements',
                backstory='Expert fact checker with extensive verification experience'
            )

            # Initial content generation task
            writing_task = Task(
                description=f"""
                Write a professional, structured article about: {topic}

                STRICT REQUIREMENTS:
                1. Maximum 500 words
                2. Include citations or references
                3. Professional tone
                4. Clear structure with sections
                5. Format your response as a valid JSON object with this exact structure:
                {{
                    "content": "your article content here",
                    "structure": "outline of sections",
                    "word_count": number
                }}

                IMPORTANT JSON RULES:
                1. Response must be ONLY the JSON object
                2. No text before or after the JSON
                3. Use double quotes for strings
                4. Properly escape special characters
                """,
                agent=writer,
                expected_output="Valid JSON string containing structured article"
            )

            # Fact checking task
            fact_check_task = Task(
                description="""
                Verify the accuracy of the generated content.
                Format your response as a valid JSON object with this exact structure:
                {
                    "score": number between 0 and 100,
                    "improvements": "suggested improvements if any",
                    "citations": ["list of verified sources"]
                }

                IMPORTANT JSON RULES:
                1. Response must be ONLY the JSON object
                2. No text before or after the JSON
                3. Use double quotes for strings
                4. Properly escape special characters
                """,
                agent=fact_checker,
                expected_output="Valid JSON string with verification results"
            )

            # Execute tasks sequentially
            crew = Crew(
                agents=[writer, fact_checker],
                tasks=[writing_task, fact_check_task],
                verbose=True
            )

            # Process results
            result = crew.kickoff()
            logger.debug(f"Raw factual content generation result: {result}")

            # Extract JSON from each task result
            content_json = self._extract_json(str(writing_task.output))
            if not content_json:
                raise ValueError("Failed to parse content generation output")

            verification_json = self._extract_json(str(fact_check_task.output))
            if not verification_json:
                raise ValueError("Failed to parse verification output")

            # Combine results
            return {
                "status": TaskStatus.COMPLETE.value,
                "content": content_json["content"],
                "structure": content_json.get("structure", ""),
                "word_count": content_json.get("word_count", 0),
                "verification": {
                    "score": verification_json.get("score", 0),
                    "improvements": verification_json.get("improvements", ""),
                    "citations": verification_json.get("citations", [])
                },
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Factual content generation error: {str(e)}")
            return {
                "status": TaskStatus.FAILED.value,
                "error": str(e),
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat()
                }
            }

    def run_fact_check(self, content: str) -> dict:
        """Run the fact checking process using a single agent approach."""
        try:
            # Create a single fact-checking agent for simplicity
            fact_checker = self.create_agent(
                role='Fact Checker',
                goal='Verify the accuracy of information and provide a detailed assessment',
                backstory='Expert fact checker with extensive experience in information verification'
            )

            # Create a single task for fact checking
            fact_check_task = Task(
                description=f"""
                Carefully fact-check the following content:

                {content}

                Format your response as a valid JSON object with this exact structure:
                {{
                    "score": number between 0 and 100,
                    "details": "detailed explanation of the findings"
                }}

                STRICT REQUIREMENTS:
                1. Response must be ONLY the JSON object
                2. No text before or after the JSON
                3. Use double quotes for strings
                4. Properly escape special characters
                """,
                agent=fact_checker,
                expected_output="Valid JSON string containing score and details"
            )

            # Set up the crew with just one agent
            crew = Crew(
                agents=[fact_checker],
                tasks=[fact_check_task],
                verbose=True
            )

            # Execute the task
            result = crew.kickoff()
            result_str = str(result).strip()

            # Super detailed logging
            logger.debug("=== Raw Fact Check Response ===")
            logger.debug(f"Response type: {type(result_str)}")
            logger.debug(f"Response length: {len(result_str)}")
            logger.debug(f"First 100 characters: {repr(result_str[:100])}")
            logger.debug(f"Last 100 characters: {repr(result_str[-100:])}")

            # Simple JSON extraction - exactly like content generation
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')

            logger.debug(f"JSON start index: {start_idx}")
            logger.debug(f"JSON end index: {end_idx}")

            if start_idx == -1 or end_idx == -1:
                logger.error("No JSON object found in response")
                logger.error(f"Raw response: {repr(result_str)}")
                return {"score": 0, "details": "Error: Could not extract JSON from response"}

            # Extract the JSON string
            json_str = result_str[start_idx:end_idx + 1]
            logger.debug(f"Extracted JSON string: {repr(json_str)}")

            try:
                # Parse the JSON
                parsed_result = json.loads(json_str)
                logger.debug(f"Successfully parsed JSON: {parsed_result}")

                # Validate and format the result
                if "score" not in parsed_result or "details" not in parsed_result:
                    raise ValueError("Missing required fields in response")

                score = float(parsed_result["score"])
                score = max(0, min(100, score))  # Ensure score is between 0 and 100

                return {
                    "score": score,
                    "details": f"Veracity Score: {score} - {parsed_result['details']}"
                }

            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"Position: {e.pos}, Line: {e.lineno}, Column: {e.colno}")
                logger.error(f"Problematic JSON: {repr(json_str)}")
                return {"score": 0, "details": f"Error: JSON parsing failed - {str(e)}"}

        except Exception as e:
            logger.error(f"Fact check error: {str(e)}")
            return {"score": 0, "details": f"Error during fact check: {str(e)}"}

    def generate_content(self, topic: str) -> dict:
        try:
            writer = self.create_agent(
                role='Content Writer',
                goal='Create engaging and factual content',
                backstory='Professional writer specializing in accurate content creation'
            )

            writing_task = Task(
                description=f"""
                Write an engaging article about: {topic}

                Format your response as a valid JSON object with this exact structure:
                {{
                    "content": "your article content here"
                }}

                STRICT REQUIREMENTS:
                1. Response must be ONLY the JSON object
                2. No text before or after the JSON
                3. Use double quotes for strings
                4. Properly escape special characters
                """,
                agent=writer,
                expected_output="Valid JSON string containing article"
            )

            crew = Crew(
                agents=[writer],
                tasks=[writing_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug("Raw content generation result:")
            logger.debug(result_str)

            # Extract and parse JSON
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")

            json_str = result_str[start_idx:end_idx + 1]
            parsed_result = json.loads(json_str)

            if "content" not in parsed_result:
                raise ValueError("Missing 'content' field in response")

            return {
                "content": parsed_result["content"],
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "word_count": len(parsed_result["content"].split())
                }
            }

        except Exception as e:
            logger.error(f"Content generation error: {str(e)}")
            return {
                "content": f"Error generating content: {str(e)}",
                "metadata": {
                    "topic": topic,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }