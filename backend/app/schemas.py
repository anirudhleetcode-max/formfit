import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ml.features import FAULT_PENALTY, FEATURES

Exercise = Literal["squat", "pushup", "curl", "press", "lunge"]


class SessionCreate(BaseModel):
    exercise: Exercise
    source: Literal["live", "upload"] = "live"


class RepIn(BaseModel):
    set: int = Field(ge=1, le=50)
    # None = the browser counted the rep but abstained from scoring it (low tracking confidence)
    score: int | None = Field(default=None, ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0, le=1, description="rep tracking confidence")
    scored: bool | None = None
    abstain_reason: str | None = Field(default=None, max_length=120)
    faults: list[str] = Field(default_factory=list, max_length=12)
    ecc_s: float = Field(ge=0, le=30)
    con_s: float = Field(ge=0, le=30)
    rom: float = Field(ge=0, le=180)
    t: float = Field(ge=0, le=6 * 3600, description="seconds since session start")
    features: dict[str, float] = Field(default_factory=dict)

    @field_validator("faults")
    @classmethod
    def known_faults(cls, v: list[str]) -> list[str]:
        bad = [f for f in v if f not in FAULT_PENALTY]
        if bad:
            raise ValueError(f"unknown fault codes: {bad[:3]}")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def consistent(self):
        if self.scored is None:
            self.scored = self.score is not None
        if not self.scored:
            self.score, self.faults = None, []
        elif self.score is None:
            raise ValueError("a scored rep needs a score")
        return self

    @field_validator("features")
    @classmethod
    def known_features(cls, v: dict[str, float]) -> dict[str, float]:
        return {k: round(float(x), 4) for k, x in v.items() if k in FEATURES and math.isfinite(x) and abs(x) < 1e4}


class SessionFinish(BaseModel):
    reps: list[RepIn] = Field(max_length=500)
    duration_s: float = Field(ge=0, le=6 * 3600)


class SessionSummary(BaseModel):
    id: str
    exercise: Exercise
    source: str
    status: str
    started_at: datetime
    finished_at: datetime | None = None
    duration_s: float | None = None
    summary: dict | None = None
    fatigue: dict | None = None
