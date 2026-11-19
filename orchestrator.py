import logging
from typing import Any
from agents.matcher_agent import MatcherAgent
from agents.parser_agent import ParserAgent
from agents.reviewer_agent import ReviewerAgent
from agents.tailor_agent import TailorAgent
from schemas import (
    BulletReviewItem,
    PipelineResult,
    RejectedBullet,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    def __init__(self, stage: str, message: str, original_error: Exception | None = None):
        self.stage = stage
        self.message = message
        self.original_error = original_error
        super().__init__(f"Pipeline failed at stage '{stage}': {message}")


def run_pipeline(
    jd_text: str,
    client: Any | None = None,
    parser: ParserAgent | None = None,
    matcher: MatcherAgent | None = None,
    tailor: TailorAgent | None = None,
    reviewer: ReviewerAgent | None = None,
    max_retries: int = 2
) -> PipelineResult:
    parser = parser or ParserAgent(client=client)
    matcher = matcher or MatcherAgent(client=client)
    tailor = tailor or TailorAgent(client=client)
    reviewer = reviewer or ReviewerAgent(client=client)

    logger.info("Starting Job Application Copilot pipeline (%d chars JD)", len(jd_text))

    # Stage 1: Parser
    logger.info("Starting pipeline stage: Parser")
    try:
        parsed_jd = parser.parse(jd_text)
        logger.info("Finished pipeline stage: Parser (role: '%s')", parsed_jd.role_title)
    except Exception as e:
        logger.error("Pipeline failure in stage 'Parser': %s", e)
        raise PipelineError("Parser", str(e), e) from e

    # Stage 2: Matcher
    logger.info("Starting pipeline stage: Matcher")
    try:
        match_result = matcher.match(parsed_jd)
        logger.info(
            "Finished pipeline stage: Matcher (%d matched, %d missing)",
            len(match_result.matched_skills),
            len(match_result.missing_skills)
        )
    except Exception as e:
        logger.error("Pipeline failure in stage 'Matcher': %s", e)
        raise PipelineError("Matcher", str(e), e) from e

    # Stage 3: Tailor (Initial)
    logger.info("Starting pipeline stage: Tailor (initial)")
    try:
        tailored_bullets = tailor.tailor(
            role_title=parsed_jd.role_title,
            matched_skills=match_result.matched_skills
        )
        logger.info("Finished pipeline stage: Tailor (drafted %d bullets)", len(tailored_bullets.bullets))
    except Exception as e:
        logger.error("Pipeline failure in stage 'Tailor': %s", e)
        raise PipelineError("Tailor", str(e), e) from e

    # Stage 4: Reviewer (Initial)
    logger.info("Starting pipeline stage: Reviewer (initial)")
    try:
        verdict = reviewer.review(tailored_bullets.bullets)
        logger.info(
            "Finished pipeline stage: Reviewer (all_approved=%s, total=%d)",
            verdict.is_all_approved,
            len(verdict.items)
        )
    except Exception as e:
        logger.error("Pipeline failure in stage 'Reviewer': %s", e)
        raise PipelineError("Reviewer", str(e), e) from e

    # Separate approved bullets and rejected bullets
    approved_bullets: list[str] = []
    pending_rejected: list[BulletReviewItem] = []

    for item in verdict.items:
        if item.approved:
            approved_bullets.append(item.bullet_point)
        else:
            pending_rejected.append(item)

    retry_count = 0

    # Retry loop: rewrite ONLY rejected bullets, maximum 2 retries
    while pending_rejected and retry_count < max_retries:
        retry_count += 1
        logger.info(
            "Starting retry %d/%d for %d rejected bullets",
            retry_count,
            max_retries,
            len(pending_rejected)
        )

        feedback = [
            {"bullet_point": item.bullet_point, "reason": item.reason}
            for item in pending_rejected
        ]

        logger.info("Starting pipeline stage: Tailor (retry %d)", retry_count)
        try:
            rewritten_bullets = tailor.tailor(
                role_title=parsed_jd.role_title,
                matched_skills=match_result.matched_skills,
                rejection_feedback=feedback
            )
            logger.info("Finished pipeline stage: Tailor (retry %d)", retry_count)
        except Exception as e:
            logger.error("Pipeline failure in stage 'Tailor' on retry %d: %s", retry_count, e)
            raise PipelineError("Tailor", str(e), e) from e

        logger.info("Starting pipeline stage: Reviewer (retry %d)", retry_count)
        try:
            retry_verdict = reviewer.review(rewritten_bullets.bullets)
            logger.info("Finished pipeline stage: Reviewer (retry %d)", retry_count)
        except Exception as e:
            logger.error("Pipeline failure in stage 'Reviewer' on retry %d: %s", retry_count, e)
            raise PipelineError("Reviewer", str(e), e) from e

        new_pending_rejected: list[BulletReviewItem] = []
        for item in retry_verdict.items:
            if item.approved:
                if item.bullet_point not in approved_bullets:
                    approved_bullets.append(item.bullet_point)
            else:
                new_pending_rejected.append(item)

        pending_rejected = new_pending_rejected
        if not pending_rejected:
            logger.info("All bullets successfully approved after retry %d", retry_count)
            break

    # Construct final rejected bullets list with reasons
    final_rejected_bullets = [
        RejectedBullet(bullet_point=item.bullet_point, reason=item.reason)
        for item in pending_rejected
    ]

    logger.info(
        "Pipeline completed. Total retries: %d, Approved bullets: %d, Rejected bullets: %d",
        retry_count,
        len(approved_bullets),
        len(final_rejected_bullets)
    )

    return PipelineResult(
        parsed_jd=parsed_jd,
        matched_skills=match_result.matched_skills,
        missing_skills=match_result.missing_skills,
        approved_bullets=approved_bullets,
        rejected_bullets=final_rejected_bullets,
        retry_count=retry_count
    )
