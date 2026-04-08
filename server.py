"""
FastAPI server — OpenEnv HTTP API
POST /reset   → initial observation
POST /step    → step result
GET  /state   → current state snapshot
GET  /health  → 200 OK
"""
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env import DataCleaningEnv, Action, Observation, StepResult
from tasks import TASKS

app = FastAPI(
    title="DataCleaningEnv",
    description="OpenEnv-compliant data cleaning environment",
    version="1.0.0",
)

# One env instance per task_id, keyed by task_id
_envs: dict[str, DataCleaningEnv] = {}


# ── Request schemas ──────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


# ── Endpoints ────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "tasks": list(TASKS.keys())}


@app.post("/reset", response_model=Observation)
def reset(req: Optional[ResetRequest] = None):
    # OpenEnv checker sometimes sends empty/null body — default to "easy"
    if req is None:
        req = ResetRequest()
    if req.task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{req.task_id}'")
    env = DataCleaningEnv(task_id=req.task_id)
    _envs[req.task_id] = env
    return env.reset()


@app.post("/step", response_model=StepResult)
def step(req: StepRequest):
    if req.task_id not in _envs:
        env = DataCleaningEnv(task_id=req.task_id)
        env.reset()
        _envs[req.task_id] = env
    return _envs[req.task_id].step(req.action)


@app.get("/state")
def state(task_id: str = "easy"):
    if task_id not in _envs:
        raise HTTPException(
            status_code=404,
            detail=f"No active environment for '{task_id}'. Call POST /reset first.",
        )
    return _envs[task_id].state()


@app.get("/tasks")
def list_tasks():
    return [
        {
            "task_id": tid,
            "difficulty": t["difficulty"],
            "description": t["description"],
            "max_steps": t["max_steps"],
        }
        for tid, t in TASKS.items()
    ]