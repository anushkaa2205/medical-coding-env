from pydantic import BaseModel, Field
from typing import List, Optional


class Observation(BaseModel):
    patient_age: int = Field(description="Age of patient")
    patient_sex: str = Field(description="Sex of patient")
    clinical_note: str = Field(description="Doctor's notes")
    feedback: str = Field(description="Grader feedback")
    remaining_tasks: int = Field(description="Tasks left")


class Action(BaseModel):
    primary_icd10: str = Field(description="Primary diagnosis code")
    secondary_icd10s: List[str] = Field(description="Secondary codes")
    cpt_codes: List[str] = Field(description="Procedure codes")


class StepResult(BaseModel):
    observation: Optional[Observation]
    reward: float
    done: bool