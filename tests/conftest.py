import json
import sys
from pathlib import Path
import pytest

# Ensure root directory is on sys.path for test runner
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def sample_profile_path(tmp_path):
    """Creates a temporary isolated profile.json for unit tests."""
    profile_data = {
        "name": "Jane Doe",
        "education": [
            {
                "degree": "B.Tech CSE",
                "institution": "Tech University",
                "graduation_year": 2025,
                "details": "Specialized in Artificial Intelligence and Web Systems"
            }
        ],
        "skills": ["Python", "FastAPI", "SQL", "Docker"],
        "learning": ["Rust", "Kubernetes"],
        "projects": [
            {
                "name": "Inventory Manager",
                "tech": ["Python", "FastAPI", "SQL"],
                "facts": [
                    "Engineered CRUD API for warehouse inventory with 10 endpoints.",
                    "Improved query latency by 25% with indexed joins."
                ]
            }
        ]
    }
    path = tmp_path / "test_profile.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile_data, f)
    return str(path)
