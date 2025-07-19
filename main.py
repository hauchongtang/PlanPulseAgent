from typing import Union
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.main import api_router

app = FastAPI(
  title="PlanPulseAgent",
  description="A Notion and Calendar integration agent",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Add a root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to PlanPulseAgent API",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PlanPulseAgent"}

app.include_router(api_router, prefix="/v1")