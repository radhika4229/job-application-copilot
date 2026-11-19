import json
import pytest
from agents.base_agent import ToolCall, LLMResponse, AgentValidationError
from agents.parser_agent import ParserAgent
from agents.matcher_agent import MatcherAgent
from agents.tailor_agent import TailorAgent
from agents.reviewer_agent import ReviewerAgent
from schemas import (
    JobDescriptionAnalysis,
    ResumeBullet,
    ResumeBullets,
    ReviewVerdict,
    BulletReviewItem,
    SkillMatchResult,
)


class MockLLMClient:
    def __init__(self, responses: list[LLMResponse]):
        self.responses = list(responses)
        self.call_count = 0
        self.recorded_messages: list[list[dict]] = []

    def generate(self, messages, system_instruction, tools=None) -> LLMResponse:
        self.call_count += 1
        self.recorded_messages.append([dict(m) for m in messages])
        if self.responses:
            return self.responses.pop(0)
        return LLMResponse(text="Fallback response")


def test_parser_agent_valid_output():
    valid_analysis = {
        "role_title": "Backend Python Engineer",
        "required_skills": ["Python", "FastAPI", "SQL"],
        "preferred_skills": ["Docker"],
        "responsibilities": ["Design REST APIs", "Write clean tests"]
    }
    client = MockLLMClient([
        LLMResponse(text=json.dumps(valid_analysis))
    ])

    parser = ParserAgent(client=client)
    result = parser.parse("Looking for a Backend Python Engineer with FastAPI and SQL experience.")

    assert isinstance(result, JobDescriptionAnalysis)
    assert result.role_title == "Backend Python Engineer"
    assert "FastAPI" in result.required_skills
    assert client.call_count == 1


def test_parser_agent_retry_on_invalid_output_recovers():
    # Attempt 1: Invalid schema (missing required role_title)
    invalid_json = json.dumps({"required_skills": ["Python"]})
    # Attempt 2: Correct schema
    valid_json = json.dumps({
        "role_title": "Python Developer",
        "required_skills": ["Python"],
        "preferred_skills": [],
        "responsibilities": ["Write code"]
    })

    client = MockLLMClient([
        LLMResponse(text=invalid_json),
        LLMResponse(text=valid_json)
    ])

    parser = ParserAgent(client=client)
    result = parser.parse("We are hiring Python developers.")

    assert isinstance(result, JobDescriptionAnalysis)
    assert result.role_title == "Python Developer"
    assert client.call_count == 2


def test_parser_agent_retry_exhaustion_raises_error():
    # Both attempts return malformed / invalid content
    client = MockLLMClient([
        LLMResponse(text="Not valid JSON at all"),
        LLMResponse(text="Still completely invalid JSON")
    ])

    parser = ParserAgent(client=client)
    with pytest.raises(AgentValidationError) as exc:
        parser.parse("Job posting text...")

    assert "failed to produce valid JobDescriptionAnalysis after retry" in str(exc.value)
    assert client.call_count == 2


def test_matcher_agent_learning_skills_classified_as_missing():
    parsed_jd = JobDescriptionAnalysis(
        role_title="Backend Engineer",
        required_skills=["Python", "FastAPI", "Kubernetes", "Rust"],
        preferred_skills=[],
        responsibilities=[]
    )

    # Step 1: Model asks for get_profile_skills
    # Step 2: Model returns match result classifying Kubernetes & Rust as missing
    expected_match = {
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Kubernetes", "Rust"],
        "match_percentage": 50.0
    }

    client = MockLLMClient([
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="get_profile_skills", args={})]
        ),
        LLMResponse(
            text=json.dumps(expected_match)
        )
    ])

    matcher = MatcherAgent(client=client)
    match_result = matcher.match(parsed_jd)

    assert isinstance(match_result, SkillMatchResult)
    assert "Python" in match_result.matched_skills
    assert "FastAPI" in match_result.matched_skills

    # Verified: Skills in learning list must be in missing_skills, NOT in matched_skills
    assert "Kubernetes" in match_result.missing_skills
    assert "Rust" in match_result.missing_skills
    assert "Kubernetes" not in match_result.matched_skills
    assert "Rust" not in match_result.matched_skills
    assert match_result.match_percentage == 50.0
    assert client.call_count == 2


def test_tailor_agent_uses_tool_facts_and_handles_feedback():
    # Step 1: Tool call to inspect project facts
    # Step 2: Bullets returned
    bullets_payload = {
        "bullets": [
            {
                "bullet_point": "Engineered 14 RESTful API endpoints using FastAPI and SQLite.",
                "target_skill": "FastAPI",
                "project_source": "Smart Expense Tracker"
            }
        ]
    }

    client = MockLLMClient([
        LLMResponse(
            text=None,
            tool_calls=[ToolCall(name="get_project_details", args={"project_name": "Smart Expense Tracker"})]
        ),
        LLMResponse(
            text=json.dumps(bullets_payload)
        )
    ])

    tailor = TailorAgent(client=client)
    result = tailor.tailor(
        role_title="API Engineer",
        matched_skills=["FastAPI"],
        rejection_feedback=[{"bullet_point": "Old bullet", "reason": "Unverified metric"}]
    )

    assert isinstance(result, ResumeBullets)
    assert len(result.bullets) == 1
    assert result.bullets[0].project_source == "Smart Expense Tracker"
    assert "14 RESTful API endpoints" in result.bullets[0].bullet_point

    # Ensure rejection feedback was injected into the prompt
    first_turn = client.recorded_messages[0]
    prompt_text = first_turn[0]["content"]
    assert "Reviewer Feedback on previous attempt" in prompt_text
    assert "Unverified metric" in prompt_text


def test_reviewer_agent_calls_tools_and_rejects_invented_claim():
    bullets = [
        ResumeBullet(
            bullet_point="Designed and implemented RESTful API with 14 endpoints using FastAPI and SQLite.",
            target_skill="FastAPI",
            project_source="Smart Expense Tracker"
        ),
        ResumeBullet(
            bullet_point="Improved database performance by 40% and deployed Redis caching clusters.",
            target_skill="Redis",
            project_source="Smart Expense Tracker"
        )
    ]

    # Step 1: Reviewer calls search_profile_evidence itself for Redis & 40%
    # Step 2: Reviewer returns verdict rejecting the invented 40% / Redis bullet
    verdict_payload = {
        "is_all_approved": False,
        "items": [
            {
                "bullet_point": bullets[0].bullet_point,
                "approved": True,
                "reason": "Verified in Smart Expense Tracker project facts."
            },
            {
                "bullet_point": bullets[1].bullet_point,
                "approved": False,
                "reason": "Invented claim: 40% performance and Redis are not present in verified profile evidence."
            }
        ]
    }

    client = MockLLMClient([
        LLMResponse(
            text=None,
            tool_calls=[
                ToolCall(name="search_profile_evidence", args={"query": "FastAPI"}),
                ToolCall(name="search_profile_evidence", args={"query": "Redis"})
            ]
        ),
        LLMResponse(
            text=json.dumps(verdict_payload)
        )
    ])

    reviewer = ReviewerAgent(client=client)
    verdict = reviewer.review(bullets)

    assert isinstance(verdict, ReviewVerdict)
    assert verdict.is_all_approved is False
    assert len(verdict.items) == 2

    # Grounded bullet is approved
    assert verdict.items[0].approved is True

    # Invented claim (40% / Redis) is rejected
    assert verdict.items[1].approved is False
    assert "Invented claim" in verdict.items[1].reason

    # Reviewer executed tool calls itself
    first_step_model_msg = client.recorded_messages[1]
    tool_turns = [m for m in first_step_model_msg if m.get("role") == "tool"]
    assert len(tool_turns) == 2
    assert any("FastAPI" in str(t.get("content")) for t in tool_turns)
