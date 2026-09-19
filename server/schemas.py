from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # The question must contain at least one character.
    query: str = Field(min_length=1)

    # Limit retrieval size so a client cannot request an unbounded result set.
    k: int = Field(default=5, ge=1, le=20)
