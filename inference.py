"""
Inference Script — DataCleaningEnv
===================================
MANDATORY env vars:
  API_BASE_URL   LLM API endpoint
  MODEL_NAME     Model identifier
  HF_TOKEN       Hugging Face / API key
  ENV_URL        Environment server URL (default: http://localhost:7860)
"""

import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 25
TEMPERATURE = 0.1
MAX_TOKENS = 300

SYSTEM_PROMPT = """You are a data cleaning agent operating on a pandas DataFrame.
You must clean the dataset by issuing structured JSON actions one at a time.

Available action_type values:
  fix_null      → fix nulls in a column. Required: column, strategy (mean|median|mode|drop|value). Optional: value (if strategy=value)
  fix_dtype     → convert column type. Required: column, dtype (int|float|str)
  deduplicate   → remove duplicate rows. Optional: column (to deduplicate by a specific key, e.g. customer_id)
  standardize   → standardize format. Required: column, fmt (date|phone|lowercase|uppercase)
  replace_value → replace a specific value. Required: column, value (old), new_value (new)
  drop_column   → drop a column. Required: column
  noop          → do nothing

Respond ONLY with a single valid JSON object. No explanation, no markdown, no extra text.

Examples:
{"action_type": "fix_null", "column": "salary", "strategy": "mean"}
{"action_type": "fix_dtype", "column": "age", "dtype": "int"}
{"action_type": "deduplicate"}
{"action_type": "deduplicate", "column": "customer_id"}
{"action_type": "standardize", "column": "email", "fmt": "lowercase"}
{"action_type": "standardize", "column": "join_date", "fmt": "date"}
{"action_type": "replace_value", "column": "city", "value": "LA", "new_value": "Los Angeles"}
"""


def get_action(client: OpenAI, obs: dict) -> dict:
    user_msg = f"""
Task ({obs['difficulty']}): {obs['task_description']}

Step: {obs['step']} / {obs['max_steps']}
Columns: {obs['columns']}
Shape: {obs['shape']}
Null counts: {json.dumps(obs['null_counts'])}
Last action result: {obs['last_action_result']}

Current DataFrame sample (up to 20 rows):
{obs['dataframe_json'][:3000]}

Issue your next cleaning action as JSON.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = completion.choices[0].message.content or ""
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"  [model error] {exc} → using noop", flush=True)
        return {"action_type": "noop"}


def run_task(client: OpenAI, task_id: str) -> float:
    print(f"[START] task={task_id}", flush=True)

    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs = resp.json()

    final_reward = 0.0
    step_num = 0

    for step_num in range(1, MAX_STEPS + 1):
        if obs.get("done"):
            break

        action = get_action(client, obs)

        resp = requests.post(
            f"{ENV_URL}/step",
            json={"task_id": task_id, "action": action},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]

        print(f"[STEP] step={step_num} reward={reward:.4f}", flush=True)
        final_reward = reward

        if done:
            break

    print(f"[END] task={task_id} score={final_reward:.4f} steps={step_num}", flush=True)
    return final_reward


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    scores: dict[str, float] = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(client, task_id)
        except Exception as exc:
            print(f"  [task error] {task_id}: {exc}", flush=True)
            scores[task_id] = 0.0

    avg = sum(scores.values()) / len(scores)
    print(f"\nFinal average score: {avg:.4f}", flush=True)


if __name__ == "__main__":
    main()