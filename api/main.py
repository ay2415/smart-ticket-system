from fastapi import FastAPI
from pydantic import BaseModel

from src.inference_pipeline import analyze_ticket

app = FastAPI(title="Smart Support Ticket Prioritization API")

class TicketRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze(request: TicketRequest):
    return analyze_ticket(request.text)
