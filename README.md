# 🎯 Job Application Copilot

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Gemini](https://img.shields.io/badge/Google%20GenAI-Gemini%20API-4285F4.svg?logo=google&logoColor=white)](https://ai.google.dev/)
[![Pydantic v2](https://img.shields.io/badge/pydantic-v2-E92063.svg?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An agentic, multi-stage AI workflow backend that tailors resume bullet points against job descriptions while strictly enforcing ground-truth verification and eliminating hallucinations.**

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Multi-Agent Architecture](#-multi-agent-architecture)
- [Workflow Pipeline](#-workflow-pipeline)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
  - [Candidate Profile Configuration](#candidate-profile-configuration)
- [API Reference](#-api-reference)
  - [Health Check](#1-health-check)
  - [Analyze Job Description](#2-analyze-job-description)
  - [List Analysis Runs](#3-list-analysis-runs)
  - [Get Run Details](#4-get-run-details)
  - [Export Run to Markdown](#5-export-run-to-markdown)
- [Docker Deployment](#-docker-deployment)
- [Evaluation & Analytics](#-evaluation--analytics)
- [Testing](#-testing)
- [License](#-license)

---

## 🚀 Overview

Most AI resume builders suffer from two major flaws:
1. **Blind Keyword Stuffing:** Extrapolating skills the applicant does not possess.
2. **Hallucination:** Fabricating metrics, tools, and achievements that never occurred in the candidate's actual history.

**Job Application Copilot** solves this by implementing an **adversarial, multi-agent architecture** powered by Google Gemini and Pydantic. It extracts requirements from a Job Description (JD), matches them against a verified profile, generates STAR-format bullet points, and subjects every bullet point to an independent auditor agent that verifies every claim against ground-truth profile evidence. If a bullet contains an unverified claim, it is rejected and sent through an automated feedback retry loop.

---

## ✨ Key Features

- **Multi-Agent Orchestration:** Specialized agents (`Parser`, `Matcher`, `Tailor`, `Reviewer`) collaborate via a structured pipeline.
- **Zero-Hallucination Policy:** Agents use explicit tool calling (`get_profile_skills`, `get_project_details`, `search_profile_evidence`) to ground every output in verified facts.
- **Adversarial Verification:** An auditor agent independently reviews each bullet point, rejecting unverified metrics, exaggerated claims, or learning-phase skills.
- **Self-Healing Feedback Loop:** When bullets are rejected, critique feedback is routed back to the tailor agent for up to two automated correction passes.
- **Persistent Run History:** Built-in SQLite database tracking all run prompts, results, retry counts, and status logs.
- **Evaluation & Gap Analytics:** Built-in analytics engine (`eval/evaluate.py`) tracking first-pass approval rates, retry success rates, and the most common missing skill gaps across applications.
- **Production-Ready API:** FastAPI backend with Pydantic v2 schemas, strict type safety, sensitive token redaction, and error boundary handling.

---

## 🤖 Multi-Agent Architecture

```
                       +-------------------------+
                       | Raw Job Description (JD)|
                       +------------+------------+
                                    |
                                    v
                     +-----------------------------+
                     |        ParserAgent          |
                     | - Extracts role title       |
                     | - Identifies required skills|
                     | - Extracts responsibilities |
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     |        MatcherAgent         |  <--- [Tool: get_profile_skills]
                     | - Matches candidate skills  |
                     | - Separates known vs missing|
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     |         TailorAgent         |  <--- [Tools: get_project_details,
                     | - Drafts STAR bullet points |          search_profile_evidence]
                     | - Anchors in verified facts |
                     +--------------+--------------+
                                    |
                                    v
                     +-----------------------------+
                     |        ReviewerAgent        |  <--- [Tools: search_profile_evidence,
                     | - Adversarial audit         |          get_project_details]
                     | - Verifies numbers & tech   |
                     +--------------+--------------+
                                    |
                    +---------------+---------------+
                    |                               |
          [Has Rejections & Retries < 2]     [All Approved / Max Retries]
                    |                               |
                    v                               v
         (Feedback Retry Loop)            +--------------------+
         Returns to TailorAgent           | Final Report & DB  |
         with targeted critiques          | - Matched Skills   |
                                          | - Missing Skills   |
                                          | - Approved Bullets |
                                          | - Rejected Bullets |
                                          +--------------------+
```

| Agent | Responsibility | Available Tools |
| :--- | :--- | :--- |
| **`ParserAgent`** | Analyzes raw job descriptions and extracts structured technical/domain requirements, preferred qualifications, and core duties. | *None (Direct LLM)* |
| **`MatcherAgent`** | Compares required skills against candidate skills. Explicitly ignores skills listed under `learning` in candidate profile. | `get_profile_skills` |
| **`TailorAgent`** | Generates high-impact resume bullet points aligned to the target role using only verified achievements and metrics. | `get_project_details`<br>`search_profile_evidence` |
| **`ReviewerAgent`** | Adversarially checks each generated bullet. Rejects bullets containing unverified technologies, fabricated statistics, or claims without ground truth. | `get_project_details`<br>`search_profile_evidence` |

---

## 📁 Project Structure

```bash
job-application-copilot/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Base Agent class, Gemini client wrapper, tool executor
│   ├── parser_agent.py        # Job description parser
│   ├── matcher_agent.py       # Skill gap & qualification matcher
│   ├── tailor_agent.py        # Resume bullet generator
│   └── reviewer_agent.py      # Adversarial bullet verification agent
├── eval/
│   ├── __init__.py
│   └── evaluate.py            # SQLite analytics & reporting script
├── tests/
│   ├── conftest.py            # Shared fixtures & test mock objects
│   ├── test_agents.py         # Unit tests for individual agents
│   ├── test_api.py            # FastAPI endpoint tests
│   ├── test_base_agent.py     # Base agent loop & retry tests
│   ├── test_evaluate.py       # Evaluation logic tests
│   ├── test_orchestrator.py   # Full pipeline & retry loop tests
│   ├── test_profile_tools.py  # Profile retrieval & tool search tests
│   └── test_schemas.py        # Pydantic schema validation tests
├── tools/
│   ├── __init__.py
│   └── profile_tools.py       # Ground-truth profile search & inspection tools
├── .env.example               # Template environment configuration
├── .gitignore
├── db.py                      # SQLite database operations & run persistence
├── main.py                    # FastAPI application & route declarations
├── orchestrator.py            # Pipeline coordinator & self-healing retry logic
├── profile.json               # Candidate master profile (ground truth)
├── pytest.ini                 # Pytest test suite configuration
├── requirements.txt           # Python dependencies
└── schemas.py                 # Pydantic models for inputs, outputs, and validation
```

---

## 🛠️ Getting Started

### Prerequisites

- **Python 3.10+** (Python 3.11 or 3.12 recommended)
- **Google Gemini API Key** ([Get one here](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/job-application-copilot.git
   cd job-application-copilot
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Environment Configuration

Copy `.env.example` to `.env` and set your Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Gemini API Key and Model configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
DB_PATH=job_copilot.db
```

### Candidate Profile Configuration

The system uses `profile.json` as the **ground truth**. Modify this file with your own background, skills, education, and real project facts:

```json
{
  "name": "Jane Developer",
  "education": [
    {
      "degree": "B.S. in Computer Science",
      "institution": "Tech University",
      "graduation_year": 2024,
      "details": "Coursework: Distributed Systems, Database Internals, Machine Learning"
    }
  ],
  "skills": [
    "Python", "FastAPI", "SQL", "PostgreSQL", "Docker", "Git", "pytest"
  ],
  "learning": [
    "Kubernetes", "Rust", "Apache Kafka"
  ],
  "projects": [
    {
      "name": "Distributed Task Queue",
      "tech": ["Python", "FastAPI", "PostgreSQL", "pytest"],
      "facts": [
        "Architected an asynchronous worker queue processing 10,000 tasks per hour.",
        "Authored 45 unit and integration tests using pytest, reaching 90% branch coverage.",
        "Reduced database connection overhead by 40% through async connection pooling."
      ]
    }
  ]
}
```

> ⚠️ **Note:** Any skill placed in the `"learning"` array will **never** be matched as a known skill and cannot be claimed in bullets.

---

## 🌐 API Reference

Start the FastAPI application:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Interactive API documentation will be available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

### 1. Health Check
Checks service readiness.

- **URL:** `/health`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

---

### 2. Analyze Job Description
Runs the full multi-agent workflow: parses requirements, matches skills, drafts tailored bullets, and verifies against profile evidence.

- **URL:** `/analyze`
- **Method:** `POST`
- **Request Body:**
  ```json
  {
    "jd_text": "We are looking for a Backend Engineer with strong Python and FastAPI experience to build scalable REST APIs. Experience with Docker and PostgreSQL required. Familiarity with Kubernetes is a plus."
  }
  ```
- **Response (`200 OK`):**
  ```json
  {
    "run_id": "8b9e6fa4-8025-45ad-bd86-63f524bc10e9",
    "role_title": "Backend Engineer",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "REST APIs"],
    "matched_skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
    "missing_skills": ["Kubernetes"],
    "tailored_bullets": [
      "Designed and implemented RESTful API with 14 endpoints using FastAPI and SQLite.",
      "Containerized inference pipeline with Docker, standardizing deployment across Linux environments."
    ],
    "rejected_bullets": [],
    "retry_count": 0
  }
  ```

---

### 3. List Analysis Runs
Retrieves recent pipeline runs with timestamps and statuses.

- **URL:** `/runs?limit=10`
- **Method:** `GET`
- **Response:**
  ```json
  [
    {
      "id": "8b9e6fa4-8025-45ad-bd86-63f524bc10e9",
      "created_at": "2026-11-20T07:45:00+00:00",
      "status": "success"
    }
  ]
  ```

---

### 4. Get Run Details
Retrieves complete execution details, input JD, outputs, and any error traces for a specific run.

- **URL:** `/runs/{run_id}`
- **Method:** `GET`

---

### 5. Export Run to Markdown
Exports the verified tailored bullets, target role, and matched skills as a clean Markdown snippet.

- **URL:** `/runs/{run_id}/export`
- **Method:** `GET`
- **Response Format:** `text/markdown`

---

## 🐳 Docker Deployment

To launch the backend using Docker:

```bash
# Build and start using Docker Compose
docker compose up -d --build
```

The service will start on `http://localhost:8000` with the SQLite database volume mapped for persistence.


---

## 📊 Evaluation & Analytics

The copilot records each run into an SQLite database (`job_copilot.db`). To inspect application analytics and skill gap distributions:

```bash
python eval/evaluate.py
```

**Sample Output:**
```text
============================================================
JOB APPLICATION COPILOT - EVALUATION SUMMARY
============================================================
Total Pipeline Runs:         18
  - Successful Runs:         18
  - Failed Runs:             0
------------------------------------------------------------
BULLET VERIFICATION METRICS:
  - Bullets approved on first pass (0 retries): 32
  - Bullets approved after retries (1-2):       10
  - Bullets permanently rejected:               2
  - Runs with 0 retries:                        12
  - Runs needing retries:                       6
------------------------------------------------------------
MOST COMMON MISSING SKILLS:
   1. Kubernetes                (14 occurrences)
   2. Apache Kafka              (9 occurrences)
   3. AWS                       (7 occurrences)
   4. GraphQL                   (5 occurrences)
============================================================
```

---

## 🧪 Testing

The repository includes a comprehensive `pytest` test suite with unit tests, schema verifications, and mock pipeline tests.

To run tests:

```bash
pytest
```

To run with verbose output and test coverage:
```bash
pytest -v
```

---

## 🔒 Security & Privacy

- **API Key Sanitization:** Error logs and API responses automatically redact Gemini API keys and sensitive URL tokens.
- **Local Data Storage:** Candidate profiles and analysis history remain on your machine in local SQLite and JSON files.

---

## 📄 License

This project is open-source and distributed under the [MIT License](LICENSE).
