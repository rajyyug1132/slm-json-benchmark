"""FastAPI wrapper: uvicorn slm_app.api:app --port 8000"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .client import ask
from .schema import SLMResponse

app = FastAPI(title="Local SLM Assistant", version="1.0.0")


class AskRequest(BaseModel):
    question: str
    model: str = "llama3.2"
    temperature: float = 0.0


class AskReply(BaseModel):
    result: SLMResponse
    attempts: int
    first_try_valid: bool
    total_latency_s: float


@app.post("/ask", response_model=AskReply)
def ask_endpoint(req: AskRequest) -> AskReply:
    r = ask(req.model, req.question, req.temperature)
    if not r.ok:
        raise HTTPException(502, detail=f"model failed schema after {len(r.attempts)} attempts")
    return AskReply(
        result=r.response,
        attempts=len(r.attempts),
        first_try_valid=r.first_try_valid,
        total_latency_s=round(r.total_latency_s, 3),
    )
