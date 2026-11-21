import logging
from typing import Any
from agents.base_agent import Agent
from schemas import JobDescriptionAnalysis

logger = logging.getLogger(__name__)

PARSER_SYSTEM_PROMPT = """You are an expert Job Description Parser agent.
Your sole job is to analyze the provided job description text and extract structured technical and domain requirements.

Strict Rules:
1. Extract only facts directly stated in the job description. Do not invent or assume requirements not mentioned.
2. Separate must-have technical skills into 'required_skills' and optional/nice-to-have skills into 'preferred_skills'.
3. Extract core day-to-day duties into 'responsibilities'.
4. Identify the exact target role title into 'role_title'.
5. Output MUST be a valid JSON object matching the JobDescriptionAnalysis schema:
   {
     "role_title": "string",
     "required_skills": ["string"],
     "preferred_skills": ["string"],
     "responsibilities": ["string"]
   }
6. Do not include markdown commentary, conversational pleasantries, or text outside the JSON object.
"""


class ParserAgent(Agent):
    def __init__(self, client: Any | None = None, max_steps: int = 5):
        super().__init__(
            name="ParserAgent",
            system_prompt=PARSER_SYSTEM_PROMPT,
            tools=[],
            client=client,
            max_steps=max_steps
        )

    def parse(self, jd_text: str) -> JobDescriptionAnalysis:
        user_prompt = (
            f"Analyze the following job description and return the structured requirements JSON:\n\n"
            f"{jd_text}"
        )
        logger.info("ParserAgent parsing job description (%d chars)", len(jd_text))
        return self.run_structured(user_prompt, JobDescriptionAnalysis)
