import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse, PlainTextResponse
from db import get_run, init_db, list_runs, save_run
from orchestrator import PipelineError, run_pipeline
from schemas import AnalyzeRequest, AnalyzeResponse, RunSummary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("job_application_copilot")


def sanitize_error_message(message: str) -> str:
    """Masks potential API keys, query params, or sensitive tokens from error messages."""
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key and len(api_key) > 4:
        message = message.replace(api_key, "[REDACTED_API_KEY]")
    message = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", message)
    message = re.sub(r"key=[0-9A-Za-z-_]+", "key=[REDACTED]", message)
    return message


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database on startup
    init_db()
    logger.info("Application startup: database ready")
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title="Job Application Copilot API",
    version="1.0.0",
    description="Agentic workflow backend for resume tailoring and job requirement verification.",
    lifespan=lifespan
)


@app.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, status_code=status.HTTP_200_OK, tags=["Pipeline"])
def analyze_job_description(request: AnalyzeRequest) -> Any:
    run_id = str(uuid.uuid4())
    logger.info("Received /analyze request (run_id=%s, length=%d)", run_id, len(request.jd_text))

    try:
        pipeline_result = run_pipeline(jd_text=request.jd_text)
        report = pipeline_result.to_analysis_report()
        save_run(
            run_id=run_id,
            jd_text=request.jd_text,
            status="success",
            result_json=report.model_dump_json()
        )
        return AnalyzeResponse(
            run_id=run_id,
            role_title=report.role_title,
            required_skills=report.required_skills,
            matched_skills=report.matched_skills,
            missing_skills=report.missing_skills,
            tailored_bullets=report.tailored_bullets,
            rejected_bullets=report.rejected_bullets,
            retry_count=pipeline_result.retry_count
        )

    except PipelineError as e:
        safe_msg = sanitize_error_message(e.message)
        logger.error("Pipeline failed for run %s at stage '%s': %s", run_id, e.stage, safe_msg)
        save_run(
            run_id=run_id,
            jd_text=request.jd_text,
            status="failed",
            error_message=f"Stage '{e.stage}': {safe_msg}"
        )
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={
                "detail": f"Pipeline failed at stage '{e.stage}': {safe_msg}",
                "stage": e.stage,
                "run_id": run_id
            }
        )
    except Exception as e:
        logger.exception("Unexpected error occurred during /analyze for run %s: %s", run_id, e)
        save_run(
            run_id=run_id,
            jd_text=request.jd_text,
            status="failed",
            error_message="Internal server error"
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "An internal server error occurred.",
                "run_id": run_id
            }
        )


@app.get("/runs", response_model=list[RunSummary], tags=["History"])
def get_recent_runs(limit: int = Query(default=50, ge=1, le=100)) -> list[dict[str, Any]]:
    return list_runs(limit=limit)


@app.get("/runs/{run_id}", tags=["History"])
def get_run_details(run_id: str) -> dict[str, Any]:
    run = get_run(run_id=run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found."
        )
    return run


@app.get("/runs/{run_id}/export", tags=["Export"])
def export_run_markdown(run_id: str) -> PlainTextResponse:
    run = get_run(run_id=run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found."
        )
    result = run.get("result")
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Run '{run_id}' has no analysis result to export."
        )

    role = result.get("role_title", "Candidate")
    matched = ", ".join(result.get("matched_skills", []))
    bullets = result.get("tailored_bullets", [])

    lines = [
        f"# Tailored Resume Highlights: {role}",
        "",
        f"**Target Role:** {role}",
        f"**Matched Skills:** {matched}",
        "",
        "## Verified Bullet Points",
        *[f"- {bullet}" for bullet in bullets]
    ]
    return PlainTextResponse(content="\n".join(lines), media_type="text/markdown")

