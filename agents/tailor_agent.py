import json
import logging
from typing import Any
from agents.base_agent import Agent
from schemas import ResumeBullets
from tools.profile_tools import get_project_details, search_profile_evidence

logger = logging.getLogger(__name__)

TAILOR_SYSTEM_PROMPT = """You are a specialized Resume Tailor agent.
Your mission is to draft compelling, high-impact resume bullet points aligned to the target role and matched skills.

Strict Honesty Rules:
1. You MAY ONLY use facts, metrics, technologies, and achievements returned by 'get_project_details' and 'search_profile_evidence'.
2. NEVER invent, exaggerate, or assume any metrics, percentages, tools, or facts not explicitly returned by the tools.
3. If no project facts exist for a skill, do NOT invent work for it.
4. Each bullet point must specify:
   - 'bullet_point': Action-oriented bullet text anchored entirely in verified profile facts.
   - 'target_skill': The verified skill highlighted.
   - 'project_source': The exact project name where the facts originate.
5. If rejection feedback from the Reviewer is provided, rewrite ONLY the rejected bullets, fixing the specific critique using verified profile evidence.
6. Output MUST be a valid JSON object matching the ResumeBullets schema:
   {
     "bullets": [
       {
         "bullet_point": "string",
         "target_skill": "string",
         "project_source": "string"
       }
     ]
   }
7. Do not include markdown commentary, conversational pleasantries, or text outside the JSON object.
"""


class TailorAgent(Agent):
    def __init__(self, client: Any | None = None, max_steps: int = 5):
        super().__init__(
            name="TailorAgent",
            system_prompt=TAILOR_SYSTEM_PROMPT,
            tools=[get_project_details, search_profile_evidence],
            client=client,
            max_steps=max_steps
        )

    def tailor(
        self,
        role_title: str,
        matched_skills: list[str],
        rejection_feedback: list[dict[str, Any]] | None = None
    ) -> ResumeBullets:
        feedback_text = ""
        if rejection_feedback:
            feedback_text = (
                f"\n\nReviewer Feedback on previous attempt:\n"
                f"{json.dumps(rejection_feedback, indent=2)}\n"
                f"Rewrite ONLY the rejected bullets to fix the exact critique. "
                f"Call 'get_project_details' or 'search_profile_evidence' to verify your claims."
            )

        user_prompt = (
            f"Draft resume bullet points for the target role: '{role_title}'.\n"
            f"Matched Skills to highlight: {json.dumps(matched_skills)}\n"
            f"Call 'get_project_details' and 'search_profile_evidence' to retrieve verified project facts.\n"
            f"Anchor every bullet strictly in returned facts.{feedback_text}\n\n"
            f"Output the final ResumeBullets JSON."
        )

        logger.info(
            "TailorAgent generating bullets for role '%s' (rejection_feedback=%s)",
            role_title,
            bool(rejection_feedback)
        )
        return self.run_structured(user_prompt, ResumeBullets)
