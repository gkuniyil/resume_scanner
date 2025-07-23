# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from matcher import rank_resumes

app = FastAPI(title="Resume Matcher API")

# Enable CORS (optional, but important if you're adding a frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobDescription(BaseModel):
    job_description: str
    top_k: int = 5

@app.get("/", tags=["Health Check"])
def home():
    """Returns health status of API"""
    return {"message": "✅ Resume Matcher API is live!"}

@app.post("/match_resumes", tags=["Matching"])
def match_resumes(input: JobDescription):
    """
    Match resumes based on a job description.
    Returns top_k resumes ranked by cosine similarity.
    """
    matches = rank_resumes(input.job_description, input.top_k)
    return {"matches": [{"name": name, "score": score} for name, score in matches]}
