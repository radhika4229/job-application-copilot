from tools.profile_tools import (
    get_profile_skills,
    get_project_details,
    search_profile_evidence,
    load_profile,
)


def test_get_profile_skills_excludes_learning(sample_profile_path):
    skills = get_profile_skills(sample_profile_path)
    # Verified skills must be present
    assert "Python" in skills
    assert "FastAPI" in skills
    assert "Docker" in skills

    # 'learning' skills must NEVER be included as claimable profile skills
    assert "Rust" not in skills
    assert "Kubernetes" not in skills


def test_get_project_details_found_and_not_found(sample_profile_path):
    project = get_project_details("Inventory Manager", sample_profile_path)
    assert project is not None
    assert project["name"] == "Inventory Manager"
    assert "FastAPI" in project["tech"]
    assert len(project["facts"]) == 2

    missing_project = get_project_details("Nonexistent Blockchain App", sample_profile_path)
    assert missing_project is None


def test_search_profile_evidence_verified_vs_learning(sample_profile_path):
    # Searching for verified skill
    python_evidence = search_profile_evidence("Python", sample_profile_path)
    assert python_evidence["found"] is True
    assert "Python" in python_evidence["matched_skills"]
    assert len(python_evidence["matched_projects"]) >= 1

    # Searching for project fact metric
    latency_evidence = search_profile_evidence("latency", sample_profile_path)
    assert latency_evidence["found"] is True
    assert any("25%" in fact for proj in latency_evidence["matched_projects"] for fact in proj["matched_facts"])

    # Searching for education evidence
    edu_evidence = search_profile_evidence("Artificial Intelligence", sample_profile_path)
    assert edu_evidence["found"] is True
    assert len(edu_evidence["matched_education"]) >= 1

    # Items only in 'learning' must NOT be counted as verified evidence
    learning_evidence = search_profile_evidence("Kubernetes", sample_profile_path)
    assert learning_evidence["found"] is False
    assert learning_evidence["matched_skills"] == []
    assert learning_evidence["matched_projects"] == []

    # Completely unmentioned technology
    fake_evidence = search_profile_evidence("Solidity", sample_profile_path)
    assert fake_evidence["found"] is False


def test_default_profile_loads_validly():
    profile = load_profile("profile.json")
    assert "skills" in profile
    assert "learning" in profile
    assert "projects" in profile
    assert len(profile["skills"]) > 0
    assert len(profile["learning"]) > 0

    skills = get_profile_skills("profile.json")
    for item in profile["learning"]:
        assert item not in skills
