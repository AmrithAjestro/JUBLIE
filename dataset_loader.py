import pandas as pd
import re
from pathlib import Path

# ✅ Use a safe cross-platform dataset path
DATA_DIR = Path(r"C:\jublie\dataset").resolve()

def clean_text(text: str) -> str:
    """Normalize and clean text for consistency."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces/newlines/tabs
    return text


def load_dataset(filename: str, drop_columns=None) -> pd.DataFrame:
    """
    Load and clean CSV dataset safely.
    :param filename: CSV file name inside dataset folder
    :param drop_columns: optional list of columns to drop
    """
    file_path = DATA_DIR / filename

    if not file_path.exists():
        print(f"[⚠️ WARN] File not found: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception as e:
        print(f"[❌ ERROR] Failed to read {file_path}: {e}")
        return pd.DataFrame()

    # Drop unwanted columns if present
    if drop_columns:
        df.drop(columns=drop_columns, errors="ignore", inplace=True)

    # Clean all text columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(clean_text)

    return df


def get_past_cases(limit: int = 10):
    """
    Returns a list of dicts with case title, issues, and decision.
    Uses fj.csv (Final Judgments dataset)
    """
    fj_df = load_dataset(
        "fj.csv",
        drop_columns=["judges name(s)", "cited cases", "Unnamed: 0"]
    )
    if fj_df.empty:
        return []

    cases = []
    for _, row in fj_df.head(limit).iterrows():
        cases.append({
            "case_title": row.get("case title", ""),
            "issues": row.get("issues", ""),
            "decision": row.get("decision", "")
        })
    return cases


def get_law_provisions(limit: int = 10):
    """
    Returns a list of dicts with article, title, and description
    from the Constitution of India dataset (coi.csv)
    """
    coi_df = load_dataset("coi.csv")
    if coi_df.empty:
        return []

    laws = []
    for _, row in coi_df.head(limit).iterrows():
        laws.append({
            "article": row.get("article", ""),
            "title": row.get("title", ""),
            "description": row.get("description", "")
        })
    return laws


if __name__ == "__main__":
    # Quick manual test
    print("\n[INFO] Sample past cases:")
    for case in get_past_cases(3):
        print(case)

    print("\n[INFO] Sample law provisions:")
    for law in get_law_provisions(3):
        print(law)
