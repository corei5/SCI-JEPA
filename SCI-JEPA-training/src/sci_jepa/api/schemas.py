"""Request bodies (JSON Schema in the OpenAPI file). Unknown fields are rejected; text is NFC-normalised
and control characters are removed before the length limits apply (specification, section 7a)."""
from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sci_jepa.api.textparts import normalise

Goal = Literal["forward", "inverse", "abduction", "consistency", "edit_check", "intervention"]
RankGoal = Literal["forward", "inverse", "abduction"]
Scoring = Literal["hybrid", "jepa", "text", "bm25"]
FieldName = Literal["Physics", "Computer Science", "Mathematics", "Astrophysics", "EESS & Engineering",
                    "Quantitative Biology & Medicine", "Statistics", "Economics & Finance", "Chemistry", "Other"]
ModelPin = Optional[str]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _clean(v):
    return normalise(v) if isinstance(v, str) else v


class Filters(Strict):
    fields: Optional[List[FieldName]] = None
    year_from: Optional[int] = Field(None, ge=1990, le=2026)
    year_to: Optional[int] = Field(None, ge=1990, le=2026)

    @model_validator(mode="after")
    def _order(self):
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("out_of_range: year_from must be <= year_to")
        return self


class Options(Strict):
    top_k: int = Field(10, ge=1, le=50)
    scoring: Scoring = "hybrid"
    filters: Optional[Filters] = None


class Parts(Strict):
    problem: Optional[str] = Field(None, max_length=1500)
    method: Optional[str] = Field(None, max_length=1500)
    conditions: Optional[str] = Field(None, max_length=1500)
    result: Optional[str] = Field(None, max_length=1500)
    claim: Optional[str] = Field(None, max_length=1000)
    evidence: Optional[str] = Field(None, max_length=1500)

    _n = field_validator("*", mode="before")(_clean)

    def given(self):
        return {k: v for k, v in self.model_dump().items() if v}


class Change(Strict):
    part: str
    text: str = Field(..., min_length=1, max_length=1500)

    _n = field_validator("text", mode="before")(_clean)


class AskRequest(Strict):
    text: str = Field(..., min_length=40, max_length=5000)
    goal: Optional[Goal] = None
    options: Options = Options()
    model: ModelPin = None

    _n = field_validator("text", mode="before")(_clean)


class ParseRequest(Strict):
    text: str = Field(..., min_length=1, max_length=5000)
    goal: Optional[Goal] = None

    _n = field_validator("text", mode="before")(_clean)


class ValidateParts(Parts):
    original: Optional[str] = Field(None, max_length=1500)
    edited: Optional[str] = Field(None, max_length=1500)
    change: Optional[Change] = None


class ValidateRequest(Strict):
    task: Goal
    parts: ValidateParts


class TaskRequest(Strict):
    parts: Parts
    options: Options = Options()
    model: ModelPin = None


class ConsistencyRequest(Strict):
    parts: Parts
    related: int = Field(5, ge=0, le=10)
    options: Options = Options()
    model: ModelPin = None


class EditCheckRequest(Strict):
    original: str = Field(..., min_length=1, max_length=1500)
    edited: str = Field(..., min_length=1, max_length=1500)
    context: Optional[Parts] = None
    model: ModelPin = None

    _n = field_validator("original", "edited", mode="before")(_clean)


class InterventionRequest(Strict):
    parts: Parts
    change: Change
    options: Options = Options()
    model: ModelPin = None


PartText = Annotated[str, Field(max_length=1500)]


class SessionAnswers(Strict):
    answers: Dict[str, Union[PartText, Change]] = Field(default_factory=dict, max_length=10)
    corrections: Dict[str, Union[PartText, Change]] = Field(default_factory=dict, max_length=10)
    options: Optional[Options] = None


class AskPaperRequest(Strict):
    paper_id: str = Field(..., min_length=1, max_length=64)
    goal: RankGoal = "forward"
    reveal_truth: bool = False
    options: Options = Options()
    model: ModelPin = None
