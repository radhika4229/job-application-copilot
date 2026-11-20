import json
import logging
import os
import sqlite3
from collections import Counter
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("evaluator")


def load_runs_dataframe(db_path: str = "job_copilot.db") -> pd.DataFrame:
    path = os.getenv("DB_PATH", db_path)
    if not Path(path).is_file():
        logger.warning("Database not found at '%s'. Run an analysis first to populate data.", path)
        return pd.DataFrame()

    with sqlite3.connect(path) as conn:
        df = pd.read_sql_query("SELECT id, created_at, status, result_json FROM runs;", conn)
    return df


def evaluate_runs(db_path: str = "job_copilot.db") -> dict[str, int | list]:
    df = load_runs_dataframe(db_path)
    if df.empty:
        logger.info("No run data available to evaluate.")
        return {
            "total_runs": 0,
            "first_try_approved_bullets": 0,
            "retry_needed_approved_bullets": 0,
            "total_rejected_bullets": 0,
            "top_missing_skills": []
        }

    total_runs = len(df)
    success_runs = df[df["status"] == "success"]
    failed_runs = df[df["status"] == "failed"]

    first_try_approved_bullets = 0
    retry_needed_approved_bullets = 0
    total_rejected_bullets = 0
    all_missing_skills: list[str] = []

    runs_zero_retries = 0
    runs_with_retries = 0

    for _, row in success_runs.iterrows():
        if not row["result_json"]:
            continue
        try:
            report = json.loads(row["result_json"])
        except json.JSONDecodeError:
            continue

        retries = report.get("retry_count", 0)
        approved = report.get("tailored_bullets", [])
        rejected = report.get("rejected_bullets", [])
        missing = report.get("missing_skills", [])

        if retries == 0:
            runs_zero_retries += 1
            first_try_approved_bullets += len(approved)
        else:
            runs_with_retries += 1
            retry_needed_approved_bullets += len(approved)

        total_rejected_bullets += len(rejected)
        all_missing_skills.extend(missing)

    skill_counter = Counter(all_missing_skills)
    top_missing = skill_counter.most_common(10)

    # Print clean tabular evaluation summary
    logger.info("============================================================")
    logger.info("JOB APPLICATION COPILOT - EVALUATION SUMMARY")
    logger.info("============================================================")
    logger.info("Total Pipeline Runs:         %d", total_runs)
    logger.info("  - Successful Runs:         %d", len(success_runs))
    logger.info("  - Failed Runs:             %d", len(failed_runs))
    logger.info("------------------------------------------------------------")
    logger.info("BULLET VERIFICATION METRICS:")
    logger.info("  - Bullets approved on first pass (0 retries): %d", first_try_approved_bullets)
    logger.info("  - Bullets approved after retries (1-2):       %d", retry_needed_approved_bullets)
    logger.info("  - Bullets permanently rejected:               %d", total_rejected_bullets)
    logger.info("  - Runs with 0 retries:                        %d", runs_zero_retries)
    logger.info("  - Runs needing retries:                       %d", runs_with_retries)
    logger.info("------------------------------------------------------------")
    logger.info("MOST COMMON MISSING SKILLS:")
    if top_missing:
        missing_df = pd.DataFrame(top_missing, columns=["Skill", "Count"])
        for idx, row in missing_df.iterrows():
            logger.info("  %2d. %-25s (%d occurrences)", idx + 1, row["Skill"], row["Count"])
    else:
        logger.info("  No missing skills recorded.")
    logger.info("============================================================")

    return {
        "total_runs": total_runs,
        "first_try_approved_bullets": first_try_approved_bullets,
        "retry_needed_approved_bullets": retry_needed_approved_bullets,
        "total_rejected_bullets": total_rejected_bullets,
        "top_missing_skills": top_missing
    }


if __name__ == "__main__":
    evaluate_runs()
