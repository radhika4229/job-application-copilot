from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from db import init_db, save_run
from main import app
from orchestrator import PipelineError
from schemas import JobDescriptionAnalysis, PipelineResult


@pytest.fixture(autouse=True)
def isolated_test_db(tmp_path, monkeypatch):
    db_file = tmp_path / "test_copilot.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    init_db(str(db_file))
    yield str(db_file)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_successful_analyze(client, monkeypatch):
    fake_result = PipelineResult(
        parsed_jd=JobDescriptionAnalysis(
            role_title="Backend Developer",
            required_skills=["Python", "FastAPI"],
            preferred_skills=["Docker"],
            responsibilities=["Build APIs"]
        ),
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Kubernetes"],
        approved_bullets=["Built 14 endpoints using FastAPI."],
        rejected_bullets=[],
        retry_count=0
    )

    monkeypatch.setattr("main.run_pipeline", MagicMock(return_value=fake_result))

    jd_payload = {
        "jd_text": "We are seeking a Backend Developer proficient in Python and FastAPI to build reliable services."
    }
    response = client.post("/analyze", json=jd_payload)

    assert response.status_code == 200
    data = response.json()
    assert "run_id" in data
    assert data["role_title"] == "Backend Developer"
    assert "FastAPI" in data["matched_skills"]
    assert len(data["tailored_bullets"]) == 1

    # Verify run was saved to DB and can be retrieved
    run_response = client.get(f"/runs/{data['run_id']}")
    assert run_response.status_code == 200
    run_data = run_response.json()
    assert run_data["status"] == "success"
    assert run_data["result"]["role_title"] == "Backend Developer"


def test_analyze_validation_empty_and_too_long(client):
    # Too short (< 20 chars)
    response_short = client.post("/analyze", json={"jd_text": "Short JD"})
    assert response_short.status_code == 422

    # Too long (> 20000 chars)
    response_long = client.post("/analyze", json={"jd_text": "A" * 20001})
    assert response_long.status_code == 422


def test_pipeline_failure_returns_502_and_saved_as_failed(client, monkeypatch):
    def mock_failing_pipeline(jd_text: str):
        raise PipelineError(stage="Reviewer", message="Reviewer agent LLM connection timed out.")

    monkeypatch.setattr("main.run_pipeline", mock_failing_pipeline)

    jd_payload = {
        "jd_text": "Looking for an experienced Python developer to build web services."
    }
    response = client.post("/analyze", json=jd_payload)

    assert response.status_code == 502
    data = response.json()
    assert data["stage"] == "Reviewer"
    assert "Reviewer agent LLM connection timed out" in data["detail"]
    assert "run_id" in data

    # Verify run is recorded in database as 'failed'
    run_response = client.get(f"/runs/{data['run_id']}")
    assert run_response.status_code == 200
    run_data = run_response.json()
    assert run_data["status"] == "failed"
    assert "Stage 'Reviewer'" in run_data["error_message"]


def test_get_runs_ordering_newest_first(client):
    # Insert 3 runs with distinct timestamps
    save_run(
        run_id="run-1",
        jd_text="Job 1 description text...",
        status="success",
        created_at="2026-01-01T10:00:00Z"
    )
    save_run(
        run_id="run-2",
        jd_text="Job 2 description text...",
        status="success",
        created_at="2026-01-03T10:00:00Z"
    )
    save_run(
        run_id="run-3",
        jd_text="Job 3 description text...",
        status="failed",
        created_at="2026-01-02T10:00:00Z"
    )

    response = client.get("/runs?limit=10")
    assert response.status_code == 200
    runs = response.json()

    assert len(runs) == 3
    # Ordered newest first: run-2 (Jan 3), run-3 (Jan 2), run-1 (Jan 1)
    assert runs[0]["id"] == "run-2"
    assert runs[1]["id"] == "run-3"
    assert runs[2]["id"] == "run-1"


def test_get_run_by_id_found_and_not_found(client):
    # Existing run
    save_run(
        run_id="test-run-id-123",
        jd_text="Software engineer role text...",
        status="success",
        result_json='{"role_title": "Software Engineer"}'
    )

    response_found = client.get("/runs/test-run-id-123")
    assert response_found.status_code == 200
    data = response_found.json()
    assert data["id"] == "test-run-id-123"
    assert data["status"] == "success"
    assert data["result"]["role_title"] == "Software Engineer"

    # Non-existent run
    response_not_found = client.get("/runs/unknown-uuid-999")
    assert response_not_found.status_code == 404
    assert "not found" in response_not_found.json()["detail"].lower()


def test_export_run_markdown_success_and_errors(client):
    # Non-existent run export
    resp_404 = client.get("/runs/non-existent-id/export")
    assert resp_404.status_code == 404

    # Run with no result
    save_run(
        run_id="run-no-result",
        jd_text="Some JD",
        status="failed",
        error_message="Something failed"
    )
    resp_400 = client.get("/runs/run-no-result/export")
    assert resp_400.status_code == 400

    # Successful export
    save_run(
        run_id="run-with-result",
        jd_text="Backend Engineer JD",
        status="success",
        result_json='{"role_title": "Backend Engineer", "matched_skills": ["Python", "FastAPI"], "tailored_bullets": ["Built high-speed REST APIs using FastAPI."]}'
    )
    resp_export = client.get("/runs/run-with-result/export")
    assert resp_export.status_code == 200
    assert resp_export.headers["content-type"].startswith("text/markdown")
    content = resp_export.text
    assert "# Tailored Resume Highlights: Backend Engineer" in content
    assert "- Built high-speed REST APIs using FastAPI." in content
    assert "Python, FastAPI" in content

