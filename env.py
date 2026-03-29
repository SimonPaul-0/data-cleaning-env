"""
DataCleaningEnv — OpenEnv-compliant environment
step() / reset() / state() API
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from tasks import TASKS


# ─────────────────────────────────────────────
# Typed Pydantic models (OpenEnv spec)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    task_description: str
    difficulty: str
    dataframe_json: str = Field(description="Current DataFrame as JSON string")
    columns: List[str]
    shape: List[int]
    null_counts: Dict[str, int]
    step: int
    max_steps: int
    last_action_result: str
    done: bool


class Action(BaseModel):
    action_type: str = Field(
        description=(
            "One of: fix_null | fix_dtype | deduplicate | standardize | "
            "drop_column | replace_value | noop"
        )
    )
    column: Optional[str] = Field(default=None, description="Target column name. For deduplicate, use as subset key (e.g. 'customer_id')")
    strategy: Optional[str] = Field(
        default=None,
        description="For fix_null: mean | median | mode | drop | value",
    )
    value: Optional[str] = Field(
        default=None,
        description="For fix_null with strategy='value', or replace_value old value",
    )
    new_value: Optional[str] = Field(
        default=None, description="For replace_value: new value to set"
    )
    dtype: Optional[str] = Field(
        default=None, description="For fix_dtype: int | float | str"
    )
    fmt: Optional[str] = Field(
        default=None,
        description="For standardize: date | phone | lowercase | uppercase",
    )


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

class DataCleaningEnv:
    """
    OpenEnv-compliant data cleaning environment.

    Usage:
        env = DataCleaningEnv(task_id="easy")
        obs = env.reset()
        result = env.step(Action(action_type="deduplicate"))
        state = env.state()
    """

    def __init__(self, task_id: str = "easy") -> None:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(TASKS)}")
        self.task_id = task_id
        self.task = TASKS[task_id]
        self._df: Optional[pd.DataFrame] = None
        self._step_count: int = 0
        self._done: bool = False
        self._last_action_result: str = "Not started."

    # ── Public API ──────────────────────────────

    def reset(self) -> Observation:
        """Reset environment to initial state and return first observation."""
        self._df = self.task["get_data"]().copy()
        self._step_count = 0
        self._done = False
        self._last_action_result = "Environment reset. DataFrame loaded."
        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Apply an action, compute reward, return StepResult."""
        if self._df is None:
            raise RuntimeError("Call reset() before step().")

        if self._done:
            return StepResult(
                observation=self._observe(),
                reward=self._grade(),
                done=True,
                info={"warning": "Episode already finished."},
            )

        self._step_count += 1
        result_msg = self._apply_action(action)
        self._last_action_result = result_msg

        reward = self._grade()
        max_steps = self.task["max_steps"]
        if reward >= 1.0 or self._step_count >= max_steps:
            self._done = True

        return StepResult(
            observation=self._observe(),
            reward=reward,
            done=self._done,
            info={"action_result": result_msg, "reward": reward},
        )

    def state(self) -> Dict[str, Any]:
        """Return serializable snapshot of current environment state."""
        import math
        def safe_dict(d):
            """Recursively replace NaN/Inf with None for JSON safety."""
            if isinstance(d, dict):
                return {k: safe_dict(v) for k, v in d.items()}
            elif isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
                return None
            elif isinstance(d, list):
                return [safe_dict(i) for i in d]
            return d

        return {
            "task_id": self.task_id,
            "difficulty": self.task["difficulty"],
            "step": self._step_count,
            "max_steps": self.task["max_steps"],
            "done": self._done,
            "current_reward": self._grade() if self._df is not None else 0.0,
            "dataframe": safe_dict(self._df.where(self._df.notna(), None).to_dict()) if self._df is not None else {},
        }

    # ── Internal helpers ─────────────────────────

    def _grade(self) -> float:
        if self._df is None:
            return 0.0
        return self.task["grader"](self._df)

    def _observe(self) -> Observation:
        df = self._df
        null_counts: Dict[str, int] = {}
        df_json = "{}"
        columns: List[str] = []
        shape: List[int] = [0, 0]

        if df is not None:
            null_counts = {k: int(v) for k, v in df.isnull().sum().items()}
            df_json = df.head(20).to_json()  # limit payload size
            columns = list(df.columns)
            shape = list(df.shape)

        return Observation(
            task_id=self.task_id,
            task_description=self.task["description"],
            difficulty=self.task["difficulty"],
            dataframe_json=df_json,
            columns=columns,
            shape=shape,
            null_counts=null_counts,
            step=self._step_count,
            max_steps=self.task["max_steps"],
            last_action_result=self._last_action_result,
            done=self._done,
        )

    def _apply_action(self, action: Action) -> str:  # noqa: C901
        df = self._df
        try:
            atype = action.action_type.lower().strip()

            if atype == "noop":
                return "No operation."

            elif atype == "fix_null":
                col = action.column
                if col not in df.columns:
                    return f"ERROR: column '{col}' not found."
                before = int(df[col].isnull().sum())
                if before == 0:
                    return f"No nulls in '{col}'."
                strat = (action.strategy or "").lower()
                if strat == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strat == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif strat == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                elif strat == "drop":
                    df.dropna(subset=[col], inplace=True)
                    df.reset_index(drop=True, inplace=True)
                elif strat == "value" and action.value is not None:
                    df[col] = df[col].fillna(action.value)
                else:
                    return f"ERROR: unknown strategy '{strat}'."
                after = int(df[col].isnull().sum())
                return f"fix_null '{col}' ({strat}): {before} → {after} nulls."

            elif atype == "fix_dtype":
                col = action.column
                if col not in df.columns:
                    return f"ERROR: column '{col}' not found."
                dtype_map = {"int": "Int64", "float": "float64", "str": "string"}
                target = dtype_map.get((action.dtype or "").lower(), action.dtype)
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce") if target in ("Int64", "float64") else df[col].astype(target)
                    if target == "Int64":
                        df[col] = df[col].round().astype("Int64")
                    return f"fix_dtype '{col}' → {target}."
                except Exception as e:
                    return f"ERROR converting '{col}': {e}"

            elif atype == "deduplicate":
                before = len(df)
                subset = [action.column] if action.column and action.column in df.columns else None
                df.drop_duplicates(subset=subset, inplace=True)
                df.reset_index(drop=True, inplace=True)
                self._df = df
                subset_label = f" by '{action.column}'" if subset else ""
                return f"deduplicate{subset_label}: {before} → {len(df)} rows (removed {before - len(df)})."

            elif atype == "standardize":
                col = action.column
                if col not in df.columns:
                    return f"ERROR: column '{col}' not found."
                fmt = (action.fmt or "").lower()
                if fmt == "date":
                    df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
                    return f"standardize '{col}' → date YYYY-MM-DD."
                elif fmt == "phone":
                    df[col] = df[col].astype(str).str.replace(r"\D", "", regex=True)
                    return f"standardize '{col}' → digits only."
                elif fmt == "lowercase":
                    df[col] = df[col].astype(str).str.lower().str.strip()
                    return f"standardize '{col}' → lowercase."
                elif fmt == "uppercase":
                    df[col] = df[col].astype(str).str.upper().str.strip()
                    return f"standardize '{col}' → uppercase."
                else:
                    return f"ERROR: unknown fmt '{fmt}'."

            elif atype == "drop_column":
                col = action.column
                if col not in df.columns:
                    return f"ERROR: column '{col}' not found."
                df.drop(columns=[col], inplace=True)
                return f"drop_column '{col}'."

            elif atype == "replace_value":
                col = action.column
                if col not in df.columns:
                    return f"ERROR: column '{col}' not found."
                old_val = action.value
                new_val = action.new_value
                count = int((df[col] == old_val).sum())
                df[col] = df[col].replace(old_val, new_val)
                return f"replace_value '{col}': '{old_val}' → '{new_val}' ({count} replaced)."

            else:
                return f"ERROR: unknown action_type '{atype}'."

        except Exception as exc:
            return f"ERROR: {exc}"
