"""
Local smoke test — no server needed.
Runs all 3 tasks with a rule-based agent to verify the env works end-to-end.

Usage:
    python test_local.py
"""

from env import DataCleaningEnv, Action


def run_rule_based_agent(task_id: str, actions: list[dict]) -> float:
    env = DataCleaningEnv(task_id=task_id)
    obs = env.reset()
    print(f"\n  [{task_id}] Goal: {obs.task_description[:80]}...")
    print(f"  [{task_id}] Initial reward: {env._grade():.4f}")

    reward = 0.0
    for action_dict in actions:
        result = env.step(Action(**action_dict))
        reward = result.reward
        print(f"  [{task_id}] {action_dict['action_type']:15s} → reward: {reward:.4f} | {result.info.get('action_result', '')[:60]}")
        if result.done:
            print(f"  [{task_id}] Done!")
            break

    return reward


def main():
    print("=" * 60)
    print("  DataCleaningEnv — Local Smoke Test")
    print("=" * 60)

    scores = {}

    # ── EASY ──────────────────────────────────────
    scores["easy"] = run_rule_based_agent("easy", [
        {"action_type": "fix_null", "column": "salary", "strategy": "mean"},
        {"action_type": "fix_null", "column": "department", "strategy": "mode"},
        {"action_type": "fix_null", "column": "name", "strategy": "drop"},
        {"action_type": "fix_dtype", "column": "age", "dtype": "float"},
        {"action_type": "fix_null", "column": "age", "strategy": "median"},
        {"action_type": "fix_dtype", "column": "age", "dtype": "int"},
    ])

    # ── MEDIUM ────────────────────────────────────
    scores["medium"] = run_rule_based_agent("medium", [
        {"action_type": "deduplicate"},
        {"action_type": "standardize", "column": "phone", "fmt": "phone"},
        {"action_type": "standardize", "column": "join_date", "fmt": "date"},
        {"action_type": "standardize", "column": "email", "fmt": "lowercase"},
    ])

    # ── HARD ──────────────────────────────────────
    scores["hard"] = run_rule_based_agent("hard", [
        {"action_type": "replace_value", "column": "city", "value": "LA", "new_value": "Los Angeles"},
        {"action_type": "replace_value", "column": "name", "value": "Robert Wilson", "new_value": "Bob Wilson"},
        {"action_type": "drop_column", "column": "source"},
        {"action_type": "deduplicate"},
    ])

    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    for task_id, score in scores.items():
        bar = "█" * int(score * 30)
        status = "✅" if score >= 0.7 else "⚠️ "
        print(f"  {status} {task_id:8s}: {score:.4f}  {bar}")

    avg = sum(scores.values()) / len(scores)
    print(f"\n  Average: {avg:.4f}")
    all_ok = all(s > 0 for s in scores.values())
    print(f"\n  {'🎉 All tasks produce non-zero rewards!' if all_ok else '⛔ Some tasks returned 0 — check env logic.'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
