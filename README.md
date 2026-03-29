---
title: DataCleaningEnv
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - data-cleaning
  - reinforcement-learning
  - agent
short_description: OpenEnv environment for AI data cleaning agents
---

# DataCleaningEnv

An OpenEnv-compliant environment where an AI agent cleans messy real-world CSV datasets using structured actions.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/reset` | POST | Reset environment, get initial observation |
| `/step` | POST | Take a cleaning action |
| `/state` | GET | Current environment state |
| `/tasks` | GET | List all tasks |

## Quick Start

```bash
# Reset
curl -X POST /reset -d '{"task_id": "easy"}'

# Step
curl -X POST /step -d '{
  "task_id": "easy",
  "action": {"action_type": "fix_null", "column": "salary", "strategy": "mean"}
}'

# State
curl /state?task_id=easy
```

## Tasks

| Task | Difficulty | Goal |
|---|---|---|
| easy | Easy | Fix nulls + type errors in employee CSV |
| medium | Medium | Deduplicate + standardize phone/date/email |
| hard | Hard | Resolve conflicts from two merged data sources |
