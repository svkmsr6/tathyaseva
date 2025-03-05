import os
import json
from datetime import datetime
from crewai import Agent, Task, Crew
from openai import OpenAI
from typing import Dict, Any
from ai_router import ModelType

class ResearchCrew:
    def __init__(self, model_type: ModelType):
        self.model_type = model_type
        self.client = None
        self.initialize_client()

    def initialize_client(self):
        """Initialize the appropriate AI client based on model type and available API keys."""
        try:
            if self.model_type == ModelType.OPENAI:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                self.client = OpenAI(api_key=api_key)

            elif self.model_type == ModelType.GROK:
                api_key = os.environ.get("XAI_API_KEY")
                if not api_key:
                    raise ValueError("X.AI API key not found in environment variables")
                self.client = OpenAI(
                    base_url="https://api.x.ai/v1",
                    api_key=api_key
                )

            elif self.model_type == ModelType.LLAMA:
                # Fallback to OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                self.client = OpenAI(api_key=api_key)

            else:
                # Default to OpenAI
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not found in environment variables")
                self.client = OpenAI(api_key=api_key)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize AI client: {str(e)}")

    def create_researcher_agent(self) -> Agent:
        if not self.client:
            self.initialize_client()
        return Agent(
            role='Researcher',
            goal='Gather comprehensive information on the given topic',
            backstory='Expert researcher with vast knowledge in multiple domains',
            allow_delegation=True,
            verbose=True,
            llm=self.client
        )

    def create_fact_checker_agent(self) -> Agent:
        if not self.client:
            self.initialize_client()
        return Agent(
            role='Fact Checker',
            goal='Verify the accuracy of information and provide a veracity score',
            backstory='Experienced fact checker with attention to detail',
            allow_delegation=True,
            verbose=True,
            llm=self.client
        )

    def create_writer_agent(self) -> Agent:
        if not self.client:
            self.initialize_client()
        return Agent(
            role='Content Writer',
            goal='Create engaging and factual content',
            backstory='Professional writer specializing in accurate content creation',
            allow_delegation=True,
            verbose=True,
            llm=self.client
        )

    def create_editor_agent(self) -> Agent:
        if not self.client:
            self.initialize_client()
        return Agent(
            role='Editor',
            goal='Refine and polish content while maintaining accuracy',
            backstory='Senior editor with years of experience in content optimization',
            allow_delegation=True,
            verbose=True,
            llm=self.client
        )

    def run_fact_check(self, content: str) -> Dict[str, Any]:
        researcher = self.create_researcher_agent()
        fact_checker = self.create_fact_checker_agent()

        research_task = Task(
            description=f"Research and gather evidence about: {content}",
            agent=researcher
        )

        fact_check_task = Task(
            description="Verify the information and provide a veracity score (0-100)",
            agent=fact_checker
        )

        crew = Crew(
            agents=[researcher, fact_checker],
            tasks=[research_task, fact_check_task],
            verbose=True
        )

        result = crew.kickoff()

        # Parse the result to extract score and details
        try:
            score = float(result.split("Score: ")[1].split()[0])
            details = result.split("Details: ")[1]
            return {"score": score, "details": details}
        except Exception:
            return {"score": 0, "details": "Failed to parse result"}

    def generate_content(self, topic: str) -> Dict[str, Any]:
        researcher = self.create_researcher_agent()
        writer = self.create_writer_agent()
        editor = self.create_editor_agent()

        research_task = Task(
            description=f"Research the topic: {topic}",
            agent=researcher
        )

        writing_task = Task(
            description="Write engaging content based on the research",
            agent=writer
        )

        editing_task = Task(
            description="Polish and refine the content while ensuring accuracy",
            agent=editor
        )

        crew = Crew(
            agents=[researcher, writer, editor],
            tasks=[research_task, writing_task, editing_task],
            verbose=True
        )

        result = crew.kickoff()

        return {
            "content": result,
            "metadata": {
                "topic": topic,
                "model_used": self.model_type.value,
                "timestamp": datetime.now().isoformat()
            }
        }