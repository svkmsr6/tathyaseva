import os
import json
import logging
from datetime import datetime
from crewai import Agent, Task, Crew

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

    def run_fact_check(self, content: str) -> dict:
        try:
            researcher = self.create_agent(
                role='Researcher',
                goal='Gather comprehensive information on the given topic',
                backstory='Expert researcher with vast knowledge in multiple domains'
            )
            fact_checker = self.create_agent(
                role='Fact Checker',
                goal='Verify the accuracy of information and provide a veracity score',
                backstory='Experienced fact checker with attention to detail'
            )

            research_task = Task(
                description=f"Research this content carefully:\n{content}",
                agent=researcher,
                expected_output="Detailed research findings"
            )

            fact_check_task = Task(
                description=f"""
                Analyze the content's accuracy based on research findings.

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

            crew = Crew(
                agents=[researcher, fact_checker],
                tasks=[research_task, fact_check_task],
                verbose=True
            )

            result = crew.kickoff()
            result_str = str(result).strip()
            logger.debug("Raw fact check result:")
            logger.debug(result_str)

            # Extract JSON using the same approach as content generation
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")

            json_str = result_str[start_idx:end_idx + 1]
            parsed_result = json.loads(json_str)

            # Basic validation
            if "score" not in parsed_result or "details" not in parsed_result:
                raise ValueError("Missing required fields in response")

            score = float(parsed_result["score"])
            score = max(0, min(100, score))  # Ensure score is between 0 and 100

            return {
                "score": score,
                "details": f"Veracity Score: {score} - {parsed_result['details']}"
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