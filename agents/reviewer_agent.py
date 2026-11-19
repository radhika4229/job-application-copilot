import json
import logging
from typing import Any
from agents.base_agent import Agent
from schemas import ResumeBullet, ReviewVerdict
from tools.profile_tools import get_project_details, search_profile_evidence

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """You are a rigorous, adversarial Resume Bullet Reviewer and Auditor.
Your job is to independently verify every resume bullet point against ground-truth profile evidence.

Strict Verification Rules:
1. You MUST call 'search_profile_evidence' and/or 'get_project_details' YOURSELF to verify each bullet.
2. Do not trust claims made in bullets. Independently check every technology, metric (percentages, latency, throughput), and fact.
3. If ANY skill, tool, metric, number, or achievement in a bullet is NOT found in the verified profile evidence, you MUST REJECT the bullet:
   - 'approved': false
   - 'reason': Clear, explicit explanation of the invented claim or missing evidence.
4. If and only if all claims, metrics, and technologies in the bullet are completely supported by verified profile facts:
   - 'approved': true
   - 'reason': Confirmation of verified evidence.
5. 'is_all_approved' MUST be true only if every single bullet in 'items' has 'approved': true.
6. Output MUST be a valid JSON object matching the ReviewVerdict schema:
   {
     "is_all_approved": true,
     "items": [
       {
         "bullet_point": "string",
         "approved": true,
         "reason": "string"
       }
     ]
   }
7. Do not include markdown commentary, conversational pleasantries, or text outside the JSON object.
"""


class ReviewerAgent(Agent):
    def __init__(self, client: Any | None = None, max_steps: int = 5):
        super().__init__(
            name="ReviewerAgent",
            system_prompt=REVIEWER_SYSTEM_PROMPT,
            tools=[get_project_details, search_profile_evidence],
            client=client,
            max_steps=max_steps
        )

    def review(self, bullets: list[ResumeBullet]) -> ReviewVerdict:
        bullets_payload = [b.model_dump() for b in bullets]
        user_prompt = (
            f"Review and independently audit these resume bullets:\n"
            f"{json.dumps(bullets_payload, indent=2)}\n\n"
            f"For each bullet, call 'search_profile_evidence' and 'get_project_details' to verify every claim.\n"
            f"Reject any bullet with unverified claims, technologies, or metrics.\n"
            f"Output the final ReviewVerdict JSON."
        )

        logger.info("ReviewerAgent reviewing %d candidate bullets", len(bullets))
        return self.run_structured(user_prompt, ReviewVerdict)
