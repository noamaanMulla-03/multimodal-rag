from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # The question must contain at least one character.
    query: str = Field(min_length=1)
