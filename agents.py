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
            raise ValueError(
                "OpenAI API key not found in environment variables")

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
                description=f"Research the following content and provide evidence:\n{content}",
                agent=researcher,
                expected_output="Detailed research findings with supporting evidence"
            )

            fact_check_task = Task(
                description=f"""
                Based on the research findings, analyze the content's veracity.
                Format your response EXACTLY like this JSON object:
                {{
                    "score": number_between_0_and_100,
                    "details": "Veracity Score: number_between_0_and_100 - detailed explanation"
                }}

                STRICT REQUIREMENTS:
                1. Response must be ONLY the JSON object
                2. No text before or after the JSON
                3. Use double quotes for strings
                4. Properly escape special characters
                5. The score in "details" MUST MATCH the score value
                6. "details" must start with "Veracity Score: X" where X matches the score value
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

            # Extract and parse JSON
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON object found in response")

            json_str = result_str[start_idx:end_idx + 1]
            parsed_result = json.loads(json_str)

            if "score" not in parsed_result:
                raise ValueError("Missing 'score' field in response")
            if "details" not in parsed_result:
                raise ValueError("Missing 'details' field in response")

            score = float(parsed_result["score"])
            details = parsed_result["details"]

            # Ensure score is within valid range
            score = max(0, min(100, score))

            # Ensure details format is correct
            if not details.startswith(f"Veracity Score: {score}"):
                details = f"Veracity Score: {score} - {details.split('-', 1)[1].strip() if '-' in details else details}"

            return {
                "score": score,
                "details": details
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
                expected_output="Valid JSON string containing article")

            crew = Crew(agents=[writer], tasks=[writing_task], verbose=True)

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