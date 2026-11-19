import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_profile(profile_path: str = "profile.json") -> dict[str, Any]:
    file_path = Path(profile_path)
    if not file_path.is_file():
        logger.error("Profile file not found at %s", profile_path)
        raise FileNotFoundError(f"Profile file not found at {profile_path}")

    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_profile_skills(profile_path: str = "profile.json") -> list[str]:
    profile = load_profile(profile_path)
    # Only 'skills' is claimable. 'learning' must not be returned here.
    verified_skills = profile.get("skills", [])
    logger.info("Retrieved %d verified skills from profile", len(verified_skills))
    return list(verified_skills)


def get_project_details(project_name: str, profile_path: str = "profile.json") -> dict[str, Any] | None:
    profile = load_profile(profile_path)
    projects = profile.get("projects", [])
    target = project_name.strip().lower()

    for project in projects:
        name = project.get("name", "")
        if name.strip().lower() == target or target in name.strip().lower():
            logger.info("Found project details for '%s'", name)
            return {
                "name": name,
                "tech": list(project.get("tech", [])),
                "facts": list(project.get("facts", []))
            }

    logger.warning("No project matching '%s' found in profile", project_name)
    return None


def search_profile_evidence(query: str, profile_path: str = "profile.json") -> dict[str, Any]:
    profile = load_profile(profile_path)
    normalized_query = query.strip().lower()

    matched_skills: list[str] = []
    # Search verified skills only (exclude 'learning')
    for skill in profile.get("skills", []):
        if normalized_query in skill.lower() or skill.lower() in normalized_query:
            matched_skills.append(skill)

    matched_projects: list[dict[str, Any]] = []
    for project in profile.get("projects", []):
        p_name = project.get("name", "")
        p_tech = [t for t in project.get("tech", []) if normalized_query in t.lower() or t.lower() in normalized_query]
        p_facts = [f for f in project.get("facts", []) if normalized_query in f.lower()]

        if p_tech or p_facts or normalized_query in p_name.lower():
            matched_projects.append({
                "project": p_name,
                "matched_tech": p_tech,
                "matched_facts": p_facts
            })

    matched_education: list[str] = []
    for edu in profile.get("education", []):
        text = f"{edu.get('degree', '')} {edu.get('institution', '')} {edu.get('details', '')}"
        if normalized_query in text.lower():
            matched_education.append(text.strip())

    found = bool(matched_skills or matched_projects or matched_education)

    logger.info("Search evidence for '%s': found=%s (skills=%d, projects=%d, edu=%d)",
                query, found, len(matched_skills), len(matched_projects), len(matched_education))

    return {
        "query": query,
        "found": found,
        "matched_skills": matched_skills,
        "matched_projects": matched_projects,
        "matched_education": matched_education
    }
