"""
Pre-Submission Validator — DataCleaningEnv
==========================================
Run this before submitting to catch any disqualification issues.

Usage:
    # With server running locally:
    python validate.py --url http://localhost:7860

    # Against HF Space:
    python validate.py --url https://your-username-data-cleaning-env.hf.space
"""

import sys
import json
import argparse
import requests

TASKS = ["easy", "medium", "hard"]
PASS = "✅ PASS"
FAIL = "❌ FAIL"
WARN = "⚠️  WARN"


def check(label: str, passed: bool, detail: str = "") -> bool:
    status = PASS if passed else FAIL
    msg = f"  {status}  {label}"
    if detail:
        msg += f"\n         {detail}"
    print(msg)
    return passed


def validate(base_url: str) -> bool:
    base_url = base_url.rstrip("/")
    all_passed = True

    print(f"\n{'='*60}")
    print(f"  OpenEnv Pre-Submission Validator")
    print(f"  Target: {base_url}")
    print(f"{'='*60}\n")

    # ── 1. Health / Deploy check ──────────────────
    print("[ 1 ] HF Space deploys and responds")
    try:
        r = requests.get(f"{base_url}/health", timeout=15)
        ok = r.status_code == 200
        all_passed &= check("GET /health returns 200", ok, f"status={r.status_code}")
        if ok:
            data = r.json()
            check("health response has 'status' field", "status" in data)
    except Exception as e:
        check("GET /health reachable", False, str(e))
        all_passed = False
        print(f"\n  ⛔ Server unreachable. Start with:\n"
              f"     uvicorn server:app --host 0.0.0.0 --port 7860\n")
        return False

    print()

    # ── 2. OpenEnv spec compliance ────────────────
    print("[ 2 ] OpenEnv spec compliance — reset() / step() / state()")

    for task_id in TASKS:
        print(f"\n  Task: {task_id}")

        # reset()
        try:
            r = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=15)
            ok = r.status_code == 200
            all_passed &= check(f"POST /reset ({task_id}) → 200", ok, f"status={r.status_code}")
            if not ok:
                continue
            obs = r.json()
        except Exception as e:
            check(f"POST /reset ({task_id})", False, str(e))
            all_passed = False
            continue

        # observation fields
        required_obs_fields = [
            "task_id", "task_description", "difficulty",
            "dataframe_json", "columns", "shape",
            "null_counts", "step", "max_steps",
            "last_action_result", "done"
        ]
        missing = [f for f in required_obs_fields if f not in obs]
        all_passed &= check(
            "Observation has all required fields",
            len(missing) == 0,
            f"missing: {missing}" if missing else ""
        )

        # step()
        try:
            action = {"action_type": "noop"}
            r = requests.post(
                f"{base_url}/step",
                json={"task_id": task_id, "action": action},
                timeout=15
            )
            ok = r.status_code == 200
            all_passed &= check(f"POST /step ({task_id}) → 200", ok, f"status={r.status_code}")
            if ok:
                result = r.json()
                required_step_fields = ["observation", "reward", "done", "info"]
                missing = [f for f in required_step_fields if f not in result]
                all_passed &= check(
                    "StepResult has all required fields",
                    len(missing) == 0,
                    f"missing: {missing}" if missing else ""
                )
                reward = result.get("reward", -1)
                all_passed &= check(
                    f"Reward in [0.0, 1.0] — got {reward:.4f}",
                    0.0 <= reward <= 1.0
                )
        except Exception as e:
            check(f"POST /step ({task_id})", False, str(e))
            all_passed = False

        # state()
        try:
            r = requests.get(f"{base_url}/state", params={"task_id": task_id}, timeout=15)
            ok = r.status_code == 200
            all_passed &= check(f"GET /state ({task_id}) → 200", ok, f"status={r.status_code}")
        except Exception as e:
            check(f"GET /state ({task_id})", False, str(e))
            all_passed = False

    print()

    # ── 3. Grader quality ─────────────────────────
    print("[ 3 ] Grader quality — scores must be non-constant")

    for task_id in TASKS:
        requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=15)

        # Get initial reward
        r1 = requests.post(
            f"{base_url}/step",
            json={"task_id": task_id, "action": {"action_type": "noop"}},
            timeout=15
        )
        r1_reward = r1.json().get("reward", 0)

        # Take a meaningful action
        meaningful_actions = {
            "easy": {"action_type": "fix_null", "column": "salary", "strategy": "mean"},
            "medium": {"action_type": "deduplicate"},
            "hard": {"action_type": "replace_value", "column": "city", "value": "LA", "new_value": "Los Angeles"},
        }
        r2 = requests.post(
            f"{base_url}/step",
            json={"task_id": task_id, "action": meaningful_actions[task_id]},
            timeout=15
        )
        r2_reward = r2.json().get("reward", 0)

        # Reward should change (not always return same score)
        changed = r1_reward != r2_reward
        all_passed &= check(
            f"Grader produces varying rewards ({task_id})",
            changed,
            f"before={r1_reward:.4f}, after={r2_reward:.4f}"
        )

    print()

    # ── 4. 3+ tasks confirmed ────────────────────
    print("[ 4 ] 3+ tasks with graders")
    try:
        r = requests.get(f"{base_url}/tasks", timeout=15)
        tasks_list = r.json()
        all_passed &= check(
            f"At least 3 tasks registered — found {len(tasks_list)}",
            len(tasks_list) >= 3
        )
        difficulties = [t.get("difficulty") for t in tasks_list]
        has_easy = "easy" in difficulties
        has_hard = "hard" in difficulties
        all_passed &= check(
            "Tasks span difficulty range (easy → hard)",
            has_easy and has_hard,
            f"difficulties found: {difficulties}"
        )
    except Exception as e:
        check("GET /tasks", False, str(e))
        all_passed = False

    print()

    # ── 5. Summary ───────────────────────────────
    print(f"{'='*60}")
    if all_passed:
        print("  🎉 ALL CHECKS PASSED — Ready to submit!")
    else:
        print("  ⛔ SOME CHECKS FAILED — Fix before submitting.")
    print(f"{'='*60}\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the environment server"
    )
    args = parser.parse_args()
    success = validate(args.url)
    sys.exit(0 if success else 1)
