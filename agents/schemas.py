from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# ==========================================
# 1. RESUME AGENT SCHEMAS (Agent 2)
# ==========================================

class PersonalInfo(BaseModel):
    name: str = Field(default="Not Provided", description="Candidate's full name")
    email: str = Field(default="Not Provided", description="Candidate's email address")
    phone: str = Field(default="Not Provided", description="Candidate's phone number")
    location: str = Field(default="Not Provided", description="Candidate's location/city/country")
    linkedin: str = Field(default="Not Provided", description="LinkedIn or GitHub profile URL")


class EducationItem(BaseModel):
    degree: str = Field(description="Degree name or certification title")
    institution: str = Field(description="University, college, or platform name")
    graduation_year: str = Field(default="N/A", description="Graduation year or completion date")


class ExperienceItem(BaseModel):
    company: str = Field(description="Company or organization name")
    role: str = Field(description="Job title or designation")
    duration: str = Field(description="Time period or duration, e.g., '1 year', 'Jan 2023 - Present'")
    responsibilities: List[str] = Field(default_factory=list, description="List of key responsibilities and achievements")


class ProjectItem(BaseModel):
    name: str = Field(description="Project title")
    technologies_used: List[str] = Field(default_factory=list, description="Technologies, tools, or frameworks used")
    description: str = Field(description="Brief overview of project scope and responsibilities")


class ResumeData(BaseModel):
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list, description="All technical, framework, database, cloud, and soft skills identified")
    projects: List[ProjectItem] = Field(default_factory=list)


# ==========================================
# 2. JOB DESCRIPTION AGENT SCHEMAS (Agent 3)
# ==========================================

class JobData(BaseModel):
    target_role: str = Field(default="Target Role", description="Job title or role name extracted from job description")
    required_skills: List[str] = Field(default_factory=list, description="Mandatory/must-have technical and professional skills")
    preferred_skills: List[str] = Field(default_factory=list, description="Desirable or nice-to-have skills")
    responsibilities: List[str] = Field(default_factory=list, description="Core responsibilities and day-to-day duties")
    experience_requirements: str = Field(default="Not Specified", description="Required years of experience or seniority level")
    education_requirements: str = Field(default="Not Specified", description="Minimum or preferred educational qualifications")
    other_requirements: List[str] = Field(default_factory=list, description="Certifications, domain knowledge, or soft skills")


# ==========================================
# 3. MATCHING AGENT SCHEMAS (Agent 4)
# ==========================================

class MatchItem(BaseModel):
    requirement: str = Field(description="The specific job requirement or skill being evaluated")
    category: str = Field(description="Requirement category, e.g., 'Required Skill', 'Experience', 'Education'")
    status: Literal["Strong Match", "Partial Match", "Weak Evidence", "Missing"] = Field(
        description="Match evaluation rating"
    )
    evidence_from_resume: str = Field(
        description="Direct proof or specific text extracted from resume supporting this evaluation"
    )
    rag_context_used: Optional[str] = Field(
        default=None, 
        description="Technical relationship or context retrieved from ChromaDB (e.g., 'Flask maps to Python Backend Framework')"
    )


class MatchingAnalysis(BaseModel):
    matches: List[MatchItem] = Field(default_factory=list, description="List of requirement evaluations")


# ==========================================
# 4. GAP ANALYSIS AGENT SCHEMAS (Agent 5)
# ==========================================

class GapItem(BaseModel):
    skill: str = Field(description="Name of the missing or partially matched skill/requirement")
    priority: Literal["High Priority", "Medium Priority", "Low Priority"] = Field(
        description="Gap priority level based on job description criticality"
    )
    status: str = Field(description="Current status, e.g., 'Missing', 'Partially Met'")
    reason: str = Field(description="Explanation of why this is considered a gap")
    recommendation: str = Field(description="Actionable step the candidate can take to bridge this gap")


class GapAnalysis(BaseModel):
    skill_gaps: List[GapItem] = Field(default_factory=list)
    overall_recommendations: List[str] = Field(default_factory=list, description="High-level practical advice steps")


# ==========================================
# 5. MANAGER / FINAL REPORT SCHEMA (Agent 1)
# ==========================================

class FinalReport(BaseModel):
    candidate_name: str = Field(default="Candidate", description="Full name of candidate")
    target_role: str = Field(default="Target Role", description="Role being analyzed for")
    overall_fit_summary: str = Field(description="Concise 3-4 sentence narrative summarizing overall candidate alignment")
    fit_category: Literal["Strong Match", "Partial Match", "Weak Match"] = Field(description="Overall fit classification")
    strengths: List[str] = Field(default_factory=list, description="Top 2-4 candidate strengths")
    top_gaps: List[str] = Field(default_factory=list, description="Top primary gaps or risk factors")