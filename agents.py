import os
import json
import logging
import time
import markdown
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
        self.md = markdown.Markdown(extensions=['extra'])

    def _convert_to_html(self, content):
        """Convert markdown content to HTML with proper escaping."""
        try:
            return self.md.convert(content)
        except Exception as e:
            logger.error(f"Markdown conversion error: {str(e)}")
            return content

    def create_agent(self, role, goal, backstory):
        """Create an agent with proper JSON response format configuration."""
        return Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            allow_delegation=True,
            verbose=True,
            api_key=self.api_key,
            llm_config={
                "config_list": [{
                    "model": "gpt-4o",
                    "api_key": self.api_key,
                    "response_format": {"type": "json_object"}
                }]
            }
        )

    def extract_json_with_retries(self, text, max_retries=3):
        """Extract JSON from text with retries and validation."""
        for attempt in range(max_retries):
            try:
                # Clean the text
                text = text.strip()
                if text.startswith('```json'):
                    text = text[7:]
                if text.endswith('```'):
                    text = text[:-3]

                # Find the outermost JSON object
                bracket_count = 0
                start_idx = -1
                end_idx = -1

                for i, char in enumerate(text):
                    if char == '{':
                        if bracket_count == 0:
                            start_idx = i
                        bracket_count += 1
                    elif char == '}':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            break

                if start_idx != -1 and end_idx != -1:
                    json_str = text[start_idx:end_idx]
                    # Clean up the JSON string
                    json_str = json_str.replace('\n', ' ').strip()
                    # Parse and validate JSON
                    result = json.loads(json_str)
                    return result

            except Exception as e:
                logger.error(f"JSON extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue

        return None

    def generate_factual_content(self, topic: str) -> dict:
        """Generate factual, verified content with enhanced error handling."""
        try:
            # Create writer agent
            writer = self.create_agent(
                role='Content Writer',
                goal='Create engaging and factual content within 500 words',
                backstory='Professional writer specializing in concise, accurate content creation'
            )

            # Content generation task with strict JSON format
            writing_task = Task(
                description=f"Write a well-structured article about: {topic}. Output format: {{\"content\": \"markdown article here\", \"structure\": \"outline\", \"word_count\": number}}",
                agent=writer,
                expected_output="Valid JSON object containing article content, structure, and word count"
            )

            # Execute content generation
            content_crew = Crew(
                agents=[writer],
                tasks=[writing_task],
                verbose=True  # Changed from verbose=2
            )
            content_result = content_crew.kickoff()

            # Extract and validate content JSON
            content_json = self.extract_json_with_retries(str(content_result))
            if not content_json:
                raise ValueError("Failed to generate valid content")

            # Create fact checker agent
            fact_checker = self.create_agent(
                role='Fact Checker',
                goal='Verify content accuracy and suggest improvements',
                backstory='Expert fact checker with extensive verification experience'
            )

            # Fact checking task
            fact_check_task = Task(
                description=f"Verify this content: {content_json.get('content', '')}. Output format: {{\"score\": number, \"improvements\": \"text\", \"citations\": [\"sources\"]}}",
                agent=fact_checker,
                expected_output="Valid JSON object containing verification score, improvements, and citations"
            )

            # Execute fact checking
            fact_crew = Crew(
                agents=[fact_checker],
                tasks=[fact_check_task],
                verbose=True  # Changed from verbose=2
            )
            fact_result = fact_crew.kickoff()

            # Extract and validate verification JSON
            verification_json = self.extract_json_with_retries(str(fact_result))
            if not verification_json:
                raise ValueError("Failed to generate valid verification")

            # Convert markdown content to HTML
            html_content = self._convert_to_html(content_json["content"])
            html_structure = self._convert_to_html(content_json.get("structure", ""))

            # Return combined results
            return {
                "status": TaskStatus.COMPLETE.value,
                "content": html_content,
                "content_markdown": content_json["content"],
                "structure": html_structure,
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
        """Run fact checking with enhanced error handling."""
        try:
            fact_checker = self.create_agent(
                role='Fact Checker',
                goal='Verify the accuracy of information',
                backstory='Expert fact checker with extensive verification experience'
            )

            fact_check_task = Task(
                description=f"Verify this content: {content}. Output format: {{\"score\": number, \"details\": \"explanation\"}}",
                agent=fact_checker,
                expected_output="Valid JSON object containing verification score and detailed explanation"
            )

            crew = Crew(
                agents=[fact_checker],
                tasks=[fact_check_task],
                verbose=True  # Changed from verbose=2
            )
            result = crew.kickoff()

            # Extract and validate JSON
            parsed_result = self.extract_json_with_retries(str(result))
            if not parsed_result:
                raise ValueError("Failed to generate valid fact check result")

            return {
                "score": float(parsed_result["score"]),
                "details": parsed_result["details"]
            }

        except Exception as e:
            logger.error(f"Fact check error: {str(e)}")
            return {
                "score": 0,
                "details": f"Error during fact check: {str(e)}"
            }

    def generate_content(self, topic: str) -> dict:
        try:
            writer = self.create_agent(
                role='Content Writer',
                goal='Create engaging and factual content',
                backstory='Professional writer specializing in accurate content creation'
            )

            writing_task = Task(
                description=f"Write an article about: {topic}. Output format: {{\"content\": \"article here\"}}",
                agent=writer,
                expected_output="Valid JSON string containing article"
            )

            crew = Crew(
                agents=[writer],
                tasks=[writing_task],
                verbose=True  # Changed from verbose=2
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug("Raw content generation result:")
            logger.debug(result_str)

            # Extract and parse JSON
            parsed_result = self.extract_json_with_retries(result_str)

            if not parsed_result:
                raise ValueError("Failed to extract JSON from content generation result")

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