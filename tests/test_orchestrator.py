from unittest.mock import MagicMock
import pytest
from orchestrator import run_pipeline, PipelineError
from schemas import (
    BulletReviewItem,
    JobDescriptionAnalysis,
    ResumeBullet,
    ResumeBullets,
    ReviewVerdict,
    SkillMatchResult,
)


@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse.return_value = JobDescriptionAnalysis(
        role_title="Backend Engineer",
        required_skills=["Python", "FastAPI"],
        preferred_skills=["Docker"],
        responsibilities=["Develop APIs"]
    )
    return parser


@pytest.fixture
def mock_matcher():
    matcher = MagicMock()
    matcher.match.return_value = SkillMatchResult(
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Kubernetes"],
        match_percentage=66.7
    )
    return matcher


def test_pipeline_all_bullets_approved_first_try(mock_parser, mock_matcher):
    bullet1 = ResumeBullet(
        bullet_point="Built 14 endpoints using FastAPI.",
        target_skill="FastAPI",
        project_source="Smart Expense Tracker"
    )
    bullet2 = ResumeBullet(
        bullet_point="Engineered database layer with SQLite.",
        target_skill="SQLite",
        project_source="Smart Expense Tracker"
    )

    tailor = MagicMock()
    tailor.tailor.return_value = ResumeBullets(bullets=[bullet1, bullet2])

    reviewer = MagicMock()
    reviewer.review.return_value = ReviewVerdict(
        is_all_approved=True,
        items=[
            BulletReviewItem(bullet_point=bullet1.bullet_point, approved=True, reason="Verified."),
            BulletReviewItem(bullet_point=bullet2.bullet_point, approved=True, reason="Verified.")
        ]
    )

    result = run_pipeline(
        jd_text="Job description text...",
        parser=mock_parser,
        matcher=mock_matcher,
        tailor=tailor,
        reviewer=reviewer
    )

    assert result.retry_count == 0
    assert len(result.approved_bullets) == 2
    assert len(result.rejected_bullets) == 0
    assert bullet1.bullet_point in result.approved_bullets
    assert bullet2.bullet_point in result.approved_bullets
    assert tailor.tailor.call_count == 1
    assert reviewer.review.call_count == 1


def test_pipeline_rejected_then_approved_on_retry_1(mock_parser, mock_matcher):
    bullet_valid = ResumeBullet(
        bullet_point="Built 14 endpoints using FastAPI.",
        target_skill="FastAPI",
        project_source="Smart Expense Tracker"
    )
    bullet_invalid = ResumeBullet(
        bullet_point="Sped up database by 40%.",
        target_skill="SQL",
        project_source="Smart Expense Tracker"
    )
    bullet_rewritten = ResumeBullet(
        bullet_point="Reduced query latency by 30% with indexes.",
        target_skill="SQL",
        project_source="Smart Expense Tracker"
    )

    tailor = MagicMock()
    tailor.tailor.side_effect = [
        ResumeBullets(bullets=[bullet_valid, bullet_invalid]),  # Initial pass
        ResumeBullets(bullets=[bullet_rewritten])              # Retry 1
    ]

    reviewer = MagicMock()
    reviewer.review.side_effect = [
        ReviewVerdict(
            is_all_approved=False,
            items=[
                BulletReviewItem(bullet_point=bullet_valid.bullet_point, approved=True, reason="Verified."),
                BulletReviewItem(bullet_point=bullet_invalid.bullet_point, approved=False, reason="Invented 40%.")
            ]
        ),
        ReviewVerdict(
            is_all_approved=True,
            items=[
                BulletReviewItem(bullet_point=bullet_rewritten.bullet_point, approved=True, reason="Verified 30%.")
            ]
        )
    ]

    result = run_pipeline(
        jd_text="Job description text...",
        parser=mock_parser,
        matcher=mock_matcher,
        tailor=tailor,
        reviewer=reviewer
    )

    assert result.retry_count == 1
    assert len(result.approved_bullets) == 2
    assert len(result.rejected_bullets) == 0
    assert bullet_valid.bullet_point in result.approved_bullets
    assert bullet_rewritten.bullet_point in result.approved_bullets

    # Tailor was called with rejection feedback on retry
    retry_call_args = tailor.tailor.call_args_list[1][1]
    assert retry_call_args["rejection_feedback"] == [
        {"bullet_point": bullet_invalid.bullet_point, "reason": "Invented 40%."}
    ]


def test_pipeline_still_rejected_after_2_retries(mock_parser, mock_matcher):
    bullet_good = ResumeBullet(
        bullet_point="Built 14 endpoints using FastAPI.",
        target_skill="FastAPI",
        project_source="Smart Expense Tracker"
    )
    bullet_bad = ResumeBullet(
        bullet_point="Trained GPT-4 from scratch.",
        target_skill="AI",
        project_source="Classifier"
    )

    tailor = MagicMock()
    tailor.tailor.side_effect = [
        ResumeBullets(bullets=[bullet_good, bullet_bad]),  # Initial
        ResumeBullets(bullets=[bullet_bad]),               # Retry 1
        ResumeBullets(bullets=[bullet_bad])                # Retry 2
    ]

    reviewer = MagicMock()
    reviewer.review.side_effect = [
        ReviewVerdict(
            is_all_approved=False,
            items=[
                BulletReviewItem(bullet_point=bullet_good.bullet_point, approved=True, reason="OK"),
                BulletReviewItem(bullet_point=bullet_bad.bullet_point, approved=False, reason="Invented GPT-4")
            ]
        ),
        ReviewVerdict(
            is_all_approved=False,
            items=[
                BulletReviewItem(bullet_point=bullet_bad.bullet_point, approved=False, reason="Still invented GPT-4")
            ]
        ),
        ReviewVerdict(
            is_all_approved=False,
            items=[
                BulletReviewItem(bullet_point=bullet_bad.bullet_point, approved=False, reason="Still unverified")
            ]
        )
    ]

    result = run_pipeline(
        jd_text="Job description text...",
        parser=mock_parser,
        matcher=mock_matcher,
        tailor=tailor,
        reviewer=reviewer,
        max_retries=2
    )

    assert result.retry_count == 2
    assert len(result.approved_bullets) == 1
    assert result.approved_bullets[0] == bullet_good.bullet_point

    # Bad bullet is relegated to rejected_bullets and NEVER in approved_bullets
    assert len(result.rejected_bullets) == 1
    assert result.rejected_bullets[0].bullet_point == bullet_bad.bullet_point
    assert "Still unverified" in result.rejected_bullets[0].reason
    assert bullet_bad.bullet_point not in result.approved_bullets


def test_pipeline_approved_bullets_preserved_across_retries(mock_parser, mock_matcher):
    bullet_a = ResumeBullet(bullet_point="Bullet A", target_skill="Python", project_source="P1")
    bullet_b = ResumeBullet(bullet_point="Bullet B", target_skill="SQL", project_source="P1")
    bullet_b_prime = ResumeBullet(bullet_point="Bullet B Prime", target_skill="SQL", project_source="P1")

    tailor = MagicMock()
    tailor.tailor.side_effect = [
        ResumeBullets(bullets=[bullet_a, bullet_b]),
        ResumeBullets(bullets=[bullet_b_prime])
    ]

    reviewer = MagicMock()
    reviewer.review.side_effect = [
        ReviewVerdict(
            is_all_approved=False,
            items=[
                BulletReviewItem(bullet_point=bullet_a.bullet_point, approved=True, reason="OK"),
                BulletReviewItem(bullet_point=bullet_b.bullet_point, approved=False, reason="Needs fix")
            ]
        ),
        ReviewVerdict(
            is_all_approved=True,
            items=[
                BulletReviewItem(bullet_point=bullet_b_prime.bullet_point, approved=True, reason="OK now")
            ]
        )
    ]

    result = run_pipeline(
        jd_text="Job description text...",
        parser=mock_parser,
        matcher=mock_matcher,
        tailor=tailor,
        reviewer=reviewer
    )

    # Bullet A must be retained, and Bullet B Prime added
    assert "Bullet A" in result.approved_bullets
    assert "Bullet B Prime" in result.approved_bullets
    assert len(result.approved_bullets) == 2


@pytest.mark.parametrize("failing_stage", ["Parser", "Matcher", "Tailor", "Reviewer"])
def test_pipeline_agent_failure_raises_pipeline_error(failing_stage):
    parser = MagicMock()
    matcher = MagicMock()
    tailor = MagicMock()
    reviewer = MagicMock()

    if failing_stage == "Parser":
        parser.parse.side_effect = RuntimeError("Parser LLM connection died")
    elif failing_stage == "Matcher":
        parser.parse.return_value = JobDescriptionAnalysis(
            role_title="Dev", required_skills=[], preferred_skills=[], responsibilities=[]
        )
        matcher.match.side_effect = ValueError("Matcher invalid format")
    elif failing_stage == "Tailor":
        parser.parse.return_value = JobDescriptionAnalysis(
            role_title="Dev", required_skills=[], preferred_skills=[], responsibilities=[]
        )
        matcher.match.return_value = SkillMatchResult(
            matched_skills=[], missing_skills=[], match_percentage=100.0
        )
        tailor.tailor.side_effect = TimeoutError("Tailor timed out")
    elif failing_stage == "Reviewer":
        parser.parse.return_value = JobDescriptionAnalysis(
            role_title="Dev", required_skills=[], preferred_skills=[], responsibilities=[]
        )
        matcher.match.return_value = SkillMatchResult(
            matched_skills=[], missing_skills=[], match_percentage=100.0
        )
        tailor.tailor.return_value = ResumeBullets(bullets=[])
        reviewer.review.side_effect = RuntimeError("Reviewer crashed")

    with pytest.raises(PipelineError) as exc_info:
        run_pipeline(
            jd_text="Some text",
            parser=parser,
            matcher=matcher,
            tailor=tailor,
            reviewer=reviewer
        )

    assert exc_info.value.stage == failing_stage
    assert f"Pipeline failed at stage '{failing_stage}'" in str(exc_info.value)
