import pytest
from pydantic import ValidationError
from schemas import (
    JobDescriptionAnalysis,
    SkillMatchResult,
    ResumeBullet,
    ResumeBullets,
    BulletReviewItem,
    ReviewVerdict,
    RejectedBullet,
    AnalysisReport,
    AnalyzeRequest,
)


def test_job_description_analysis_valid():
    jd = JobDescriptionAnalysis(
        role_title="Backend Software Engineer",
        required_skills=["Python", "FastAPI"],
        preferred_skills=["Docker"],
        responsibilities=["Develop REST APIs", "Write tests"]
    )
    assert jd.role_title == "Backend Software Engineer"
    assert len(jd.required_skills) == 2


def test_job_description_analysis_missing_field():
    with pytest.raises(ValidationError):
        # role_title is required
        JobDescriptionAnalysis(required_skills=["Python"])


def test_skill_match_result_validation():
    match = SkillMatchResult(
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Kubernetes"],
        match_percentage=66.7
    )
    assert match.match_percentage == 66.7

    # Percentage over 100 should raise ValidationError
    with pytest.raises(ValidationError):
        SkillMatchResult(
            matched_skills=["Python"],
            missing_skills=[],
            match_percentage=150.0
        )


def test_resume_bullets_schema():
    bullet = ResumeBullet(
        bullet_point="Engineered high-throughput REST API using FastAPI and SQLite.",
        target_skill="FastAPI",
        project_source="Smart Expense Tracker"
    )
    container = ResumeBullets(bullets=[bullet])
    assert len(container.bullets) == 1
    assert container.bullets[0].target_skill == "FastAPI"


def test_review_verdict_schema():
    item = BulletReviewItem(
        bullet_point="Engineered high-throughput REST API using FastAPI.",
        approved=True,
        reason="Claim verified in project facts."
    )
    verdict = ReviewVerdict(is_all_approved=True, items=[item])
    assert verdict.is_all_approved is True
    assert len(verdict.items) == 1


def test_analysis_report_with_rejected_bullets():
    report = AnalysisReport(
        role_title="Python Developer",
        required_skills=["Python", "Docker"],
        matched_skills=["Python"],
        missing_skills=["Docker"],
        tailored_bullets=["Developed APIs in Python."],
        rejected_bullets=[
            RejectedBullet(
                bullet_point="Deployed Kubernetes cluster.",
                reason="Kubernetes is in learning list and has no verified project facts."
            )
        ]
    )
    assert len(report.tailored_bullets) == 1
    assert len(report.rejected_bullets) == 1
    assert report.rejected_bullets[0].bullet_point.startswith("Deployed Kubernetes")


def test_analyze_request_min_length():
    valid = AnalyzeRequest(jd_text="Looking for a Python Developer with experience in FastAPI and SQL.")
    assert valid.jd_text is not None

    with pytest.raises(ValidationError):
        AnalyzeRequest(jd_text="Short text")
