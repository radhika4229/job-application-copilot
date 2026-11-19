from pydantic import BaseModel, Field


class JobDescriptionAnalysis(BaseModel):
    role_title: str = Field(description="Target role title identified from the job description")
    required_skills: list[str] = Field(default_factory=list, description="Must-have technical and domain skills")
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have skills or qualifications")
    responsibilities: list[str] = Field(default_factory=list, description="Core day-to-day responsibilities")


class SkillMatchResult(BaseModel):
    matched_skills: list[str] = Field(default_factory=list, description="Skills present in both JD and verified profile")
    missing_skills: list[str] = Field(default_factory=list, description="Required skills from JD not found in verified profile")
    match_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of required skills matched")


class ResumeBullet(BaseModel):
    bullet_point: str = Field(description="Action-oriented resume bullet grounded in profile facts")
    target_skill: str = Field(description="Skill demonstrated by this bullet point")
    project_source: str = Field(description="Project name from which facts were drawn")


class ResumeBullets(BaseModel):
    bullets: list[ResumeBullet] = Field(default_factory=list, description="Generated list of candidate resume bullets")


class BulletReviewItem(BaseModel):
    bullet_point: str = Field(description="The bullet point text being reviewed")
    approved: bool = Field(description="True if every claim, metric, and skill is verified in profile")
    reason: str = Field(description="Approval confirmation or explicit rejection critique")


class ReviewVerdict(BaseModel):
    is_all_approved: bool = Field(description="True if all reviewed bullets were approved")
    items: list[BulletReviewItem] = Field(default_factory=list, description="Review feedback per bullet")


class RejectedBullet(BaseModel):
    bullet_point: str
    reason: str


class AnalysisReport(BaseModel):
    role_title: str
    required_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    tailored_bullets: list[str]
    rejected_bullets: list[RejectedBullet] = Field(default_factory=list)


class PipelineResult(BaseModel):
    parsed_jd: JobDescriptionAnalysis
    matched_skills: list[str]
    missing_skills: list[str]
    approved_bullets: list[str]
    rejected_bullets: list[RejectedBullet] = Field(default_factory=list)
    retry_count: int = 0

    def to_analysis_report(self) -> AnalysisReport:
        return AnalysisReport(
            role_title=self.parsed_jd.role_title,
            required_skills=self.parsed_jd.required_skills,
            matched_skills=self.matched_skills,
            missing_skills=self.missing_skills,
            tailored_bullets=self.approved_bullets,
            rejected_bullets=self.rejected_bullets,
        )



class AnalyzeRequest(BaseModel):
    jd_text: str = Field(min_length=20, description="Raw text of the job description to analyze")


class RunRecord(BaseModel):
    id: str
    created_at: str
    jd_text: str
    status: str
    result: AnalysisReport | None = None
