from pydantic import BaseModel, Field
from enum import IntEnum
from datetime import datetime
from typing import Optional

class GeneticsEnum(IntEnum):
    """
    Strict molecular genetic mutation biomarker status bounds:
    0: Wild-Type (WT) / Negative
    1: Mutant / Positive
    2: Unknown / Untested
    """
    WT = 0
    MUTANT = 1
    UNKNOWN = 2

class ClinicalPatientProfile(BaseModel):
    """
    Patient Clinical & Demographics Data Profile with Pydantic V2 bounds.
    Matches strict clinical guidelines for lung cancer risk stratification.
    """
    age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Patient age in years at date of assessment."
    )
    smoking_pack_years: float = Field(
        ..., 
        ge=0.0, 
        le=150.0, 
        description="Cumulative smoking pack-years (packs per day * years smoked)."
    )
    egfr: GeneticsEnum = Field(
        GeneticsEnum.UNKNOWN,
        description="EGFR gene mutation status (0=WT, 1=Mutant, 2=Unknown)."
    )
    kras: GeneticsEnum = Field(
        GeneticsEnum.UNKNOWN,
        description="KRAS gene mutation status (0=WT, 1=Mutant, 2=Unknown)."
    )
    alk: GeneticsEnum = Field(
        GeneticsEnum.UNKNOWN,
        description="ALK gene rearrangement status (0=WT, 1=Mutant, 2=Unknown)."
    )

class InferenceMetricsOutput(BaseModel):
    """
    Structured model output containing calibrated risk score, diagnostic categories,
    and computational latencies for analytical audit logging.
    """
    patient_id: Optional[str] = Field("DEMO-ID-001", description="De-identified patient UUID identifier.")
    risk_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Fused AI risk probability score for lung cancer nodule malignancy."
    )
    risk_category: str = Field(
        ..., 
        description="Risk classification category: 'LOW RISK', 'MODERATE RISK', or 'HIGH RISK'."
    )
    latency_ms: float = Field(
        ..., 
        description="End-to-end model pipeline execution latency in milliseconds."
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC time record of inference event execution."
    )
