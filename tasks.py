"""
Tasks for Data Cleaning OpenEnv
3 tasks: easy → medium → hard
Each has a deterministic grader returning 0.0–1.0
"""
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any


# ─────────────────────────────────────────────
# EASY: Fix nulls + wrong data types
# ─────────────────────────────────────────────

def get_easy_data() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name": ["Alice", "Bob", None, "David", "Eve",
                 "Frank", None, "Grace", "Hank", "Ivy"],
        "age": ["25", "30", "22", "thirty", "28",
                "35", "27", None, "31", "24"],
        "salary": [50000, None, 45000, 60000, None,
                   75000, 55000, 48000, None, 52000],
        "department": ["Engineering", "Marketing", "Engineering", None, "HR",
                       "Engineering", "Marketing", "HR", "Engineering", None],
    }
    return pd.DataFrame(data)


def grade_easy(df: pd.DataFrame) -> float:
    score = 0.0

    # 50%: No nulls remain (name:2, age:1, salary:3, department:2 = 8 total)
    total_nulls = int(df.isnull().sum().sum())
    original_nulls = 8
    null_score = max(0.0, (original_nulls - total_nulls) / original_nulls)
    score += 0.50 * null_score

    # 25%: 'age' column is fully numeric
    if "age" in df.columns:
        numeric_age = pd.to_numeric(df["age"], errors="coerce")
        valid = int(numeric_age.notna().sum())
        score += 0.25 * (valid / len(df))

    # 25%: 'salary' column is fully numeric
    if "salary" in df.columns:
        numeric_sal = pd.to_numeric(df["salary"], errors="coerce")
        valid = int(numeric_sal.notna().sum())
        score += 0.25 * (valid / len(df))

    return round(min(1.0, score), 4)


# ─────────────────────────────────────────────
# MEDIUM: Deduplicate + standardize formats
# ─────────────────────────────────────────────

def get_medium_data() -> pd.DataFrame:
    data = {
        "id": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5],
        "name": [
            "Alice Smith", "Bob Jones", "Carol White", "David Brown", "Eve Davis",
            "Frank Miller", "Grace Wilson", "Hank Moore", "Bob Jones", "Eve Davis",
        ],
        "phone": [
            "(555) 123-4567", "555.234.5678", "5553456789", "+1-555-456-7890",
            "(555)567-8901", "555 678 9012", "5557890123", "(555) 890-1234",
            "555.234.5678", "(555)567-8901",
        ],
        "join_date": [
            "2023-01-15", "15/02/2023", "March 3, 2023", "2023-04-20",
            "05-05-2023", "2023-06-10", "2023/07/25", "2023-08-30",
            "15/02/2023", "05-05-2023",
        ],
        "email": [
            "Alice@Company.COM", "bob@company.com", "CAROL@COMPANY.COM", "david@company.com",
            "EVE@COMPANY.COM", "frank@company.com", "Grace@Company.Com", "hank@company.com",
            "bob@company.com", "EVE@COMPANY.COM",
        ],
    }
    return pd.DataFrame(data)


def grade_medium(df: pd.DataFrame) -> float:
    score = 0.0

    # 25%: Deduplication — should be 8 rows
    target_rows = 8
    if len(df) == target_rows:
        score += 0.25
    elif len(df) < 10:
        score += 0.25 * (10 - len(df)) / 2.0

    # 25%: Phone — all digits count >= 10
    if "phone" in df.columns:
        digits = df["phone"].astype(str).str.replace(r"\D", "", regex=True)
        valid = int((digits.str.len() >= 10).sum())
        score += 0.25 * (valid / len(df))

    # 25%: join_date in YYYY-MM-DD format
    if "join_date" in df.columns:
        parsed = pd.to_datetime(df["join_date"], format="%Y-%m-%d", errors="coerce")
        valid = int(parsed.notna().sum())
        score += 0.25 * (valid / len(df))

    # 25%: email all lowercase
    if "email" in df.columns:
        correct = int((df["email"] == df["email"].str.lower()).sum())
        score += 0.25 * (correct / len(df))

    return round(min(1.0, score), 4)


# ─────────────────────────────────────────────
# HARD: Resolve conflicts from two merged sources
# ─────────────────────────────────────────────

def get_hard_data() -> pd.DataFrame:
    data = {
        "customer_id": [101, 102, 103, 104, 105, 101, 103, 106, 102, 105],
        "name": [
            "John Doe", "Jane Smith", "Bob Wilson", "Alice Brown", "Charlie Davis",
            "John Doe", "Robert Wilson", "Mary Johnson", "Jane Smith", "Charles Davis",
        ],
        "email": [
            "john@email.com", "jane@email.com", "bob@email.com",
            "alice@email.com", "charlie@email.com",
            "johndoe@email.com", "bob.wilson@email.com",
            "mary@email.com", "jane@email.com", "charlie@email.com",
        ],
        "age": [35, 28, 42, 31, 45, 35, 42, 29, 28, 45],
        "city": [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "New York", "Chicago", "Philadelphia", "LA", "Phoenix",
        ],
        "source": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
    }
    return pd.DataFrame(data)


def grade_hard(df: pd.DataFrame) -> float:
    score = 0.0

    # 40%: Exactly 6 unique customers
    if "customer_id" in df.columns:
        unique = df["customer_id"].nunique()
        if unique == 6:
            score += 0.40
        else:
            score += 0.40 * (unique / 6)

    # 30%: No duplicate customer_ids
    if "customer_id" in df.columns:
        if not df["customer_id"].duplicated().any():
            score += 0.30

    # 15%: 'LA' normalized to 'Los Angeles'
    if "city" in df.columns:
        if "LA" not in df["city"].values:
            score += 0.15

    # 15%: 'Robert Wilson' resolved to 'Bob Wilson'
    if "name" in df.columns:
        if "Robert Wilson" not in df["name"].values:
            score += 0.15

    return round(min(1.0, score), 4)


# ─────────────────────────────────────────────
# Task registry
# ─────────────────────────────────────────────

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "description": (
            "Clean this employee dataset. Fix all null values using appropriate strategies "
            "(mean for salary, mode for department, drop rows where name is null). "
            "Convert 'age' to integer — some values are text strings or invalid. "
            "Goal: zero nulls, all columns correctly typed."
        ),
        "difficulty": "easy",
        "max_steps": 20,
        "get_data": get_easy_data,
        "grader": grade_easy,
    },
    "medium": {
        "description": (
            "Clean this customer contact dataset. "
            "Remove all duplicate rows (same id appears twice). "
            "Standardize phone numbers so all digits are extractable (>=10 digits). "
            "Parse all join_date values to YYYY-MM-DD format. "
            "Normalize all email addresses to lowercase."
        ),
        "difficulty": "medium",
        "max_steps": 20,
        "get_data": get_medium_data,
        "grader": grade_medium,
    },
    "hard": {
        "description": (
            "This dataset was merged from two sources (A and B) and contains conflicting records "
            "for the same customer_id. Resolve all conflicts so each customer_id appears exactly once. "
            "Normalize city names ('LA' → 'Los Angeles'). "
            "Resolve name discrepancies ('Robert Wilson' is the same person as 'Bob Wilson'). "
            "Final dataset must have exactly 6 unique customers, no duplicate customer_ids."
        ),
        "difficulty": "hard",
        "max_steps": 25,
        "get_data": get_hard_data,
        "grader": grade_hard,
    },
}
