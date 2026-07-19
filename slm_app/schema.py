"""Pydantic schema every model response must satisfy."""
from pydantic import BaseModel, Field, field_validator


class SLMResponse(BaseModel):
    """The one JSON shape the assistant is allowed to emit."""

    answer: str = Field(min_length=1, description="Direct answer to the user's question")
    confidence: float = Field(ge=0.0, le=1.0, description="Self-reported confidence, 0..1")
    tags: list[str] = Field(default_factory=list, description="1-5 topic tags")

    @field_validator("tags")
    @classmethod
    def cap_tags(cls, v: list[str]) -> list[str]:
        if len(v) > 5:
            raise ValueError("at most 5 tags")
        return v


JSON_SCHEMA_INSTRUCTION = (
    "You must reply with ONLY a single JSON object, no markdown fences, no prose, "
    "matching exactly this schema:\n"
    '{"answer": "<string, the direct answer>", '
    '"confidence": <number between 0 and 1>, '
    '"tags": ["<1 to 5 short topic strings>"]}\n'
    "Do not add any keys. Do not wrap the JSON in backticks."
)
