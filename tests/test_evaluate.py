import json
from eval.evaluate import evaluate_runs
from db import init_db, save_run


def test_evaluate_runs_with_metrics(tmp_path, monkeypatch):
    db_file = tmp_path / "test_eval.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    init_db(str(db_file))

    # Run 1: 0 retries, 2 approved bullets, 1 missing skill
    report_1 = {
        "role_title": "Backend Dev",
        "required_skills": ["Python", "FastAPI", "Kafka"],
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Kafka"],
        "tailored_bullets": ["Bullet 1", "Bullet 2"],
        "rejected_bullets": [],
        "retry_count": 0
    }
    save_run(
        run_id="run-1",
        jd_text="Job 1 text",
        status="success",
        result_json=json.dumps(report_1),
        db_path=str(db_file)
    )

    # Run 2: 1 retry, 1 approved bullet, 1 rejected bullet, 2 missing skills
    report_2 = {
        "role_title": "ML Engineer",
        "required_skills": ["Python", "PyTorch", "Kafka", "Kubernetes"],
        "matched_skills": ["Python", "PyTorch"],
        "missing_skills": ["Kafka", "Kubernetes"],
        "tailored_bullets": ["Bullet 3"],
        "rejected_bullets": [{"bullet_point": "Fake Bullet", "reason": "Unverified claim"}],
        "retry_count": 1
    }
    save_run(
        run_id="run-2",
        jd_text="Job 2 text",
        status="success",
        result_json=json.dumps(report_2),
        db_path=str(db_file)
    )

    metrics = evaluate_runs(db_path=str(db_file))

    assert metrics["total_runs"] == 2
    assert metrics["first_try_approved_bullets"] == 2
    assert metrics["retry_needed_approved_bullets"] == 1
    assert metrics["total_rejected_bullets"] == 1

    # Kafka appeared twice in missing_skills
    skills_dict = dict(metrics["top_missing_skills"])
    assert skills_dict.get("Kafka") == 2
    assert skills_dict.get("Kubernetes") == 1
