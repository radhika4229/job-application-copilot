import json
import logging
from typing import Any
from agents.base_agent import Agent
from schemas import JobDescriptionAnalysis, SkillMatchResult
from tools.profile_tools import get_profile_skills

logger = logging.getLogger(__name__)

MATCHER_SYSTEM_PROMPT = """You are an objective Skill Matcher agent.
Your job is to compare required skills from a job description against verified profile skills and identify matches and gaps.

Strict Rules:
1. You MUST call the 'get_profile_skills' tool to inspect the candidate's verified skills list.
2. ONLY skills returned by 'get_profile_skills' are valid candidate skills.
3. If a required skill is in the candidate's 'learning' list or is not returned by 'get_profile_skills', it MUST be classified as 'missing_skills'. NEVER put it in 'matched_skills'.
4. Do not extrapolate, assume, or fabricate that a candidate knows a tool or language simply because they know a related technology.
5. Calculate 'match_percentage' accurately: (len(matched_skills) / len(required_skills)) * 100.0 if required_skills is non-empty, else 100.0.
6. Output MUST be a valid JSON object matching the SkillMatchResult schema:
   {
     "matched_skills": ["string"],
     "missing_skills": ["string"],
     "match_percentage": 0.0
   }
7. Do not include markdown commentary, conversational pleasantries, or text outside the JSON object.
"""


class MatcherAgent(Agent):
    def __init__(self, client: Any | None = None, max_steps: int = 5):
        super().__init__(
            name="MatcherAgent",
            system_prompt=MATCHER_SYSTEM_PROMPT,
            tools=[get_profile_skills],
            client=client,
            max_steps=max_steps
        )

    def match(self, parsed_jd: JobDescriptionAnalysis) -> SkillMatchResult:
        user_prompt = (
            f"Compare these required skills against the verified profile skills:\n"
            f"Required Skills: {json.dumps(parsed_jd.required_skills)}\n"
            f"Preferred Skills: {json.dumps(parsed_jd.preferred_skills)}\n\n"
            f"Call 'get_profile_skills' to retrieve verified skills, then output the SkillMatchResult JSON."
        )
        logger.info("MatcherAgent matching %d required skills", len(parsed_jd.required_skills))
        return self.run_structured(user_prompt, SkillMatchResult)
