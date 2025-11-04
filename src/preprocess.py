# src/preprocess.py
import os
import pandas as pd
from tqdm import tqdm
from src.utils import clean_text
tqdm.pandas()

RAW_PATH = os.path.join("data", "biznet_tweets_raw.csv")
CLEAN_PATH = os.path.join("data", "biznet_tweets_clean.csv")

def run_preprocess():
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"{RAW_PATH} not found.")
    df = pd.read_csv(RAW_PATH, dtype=str)
    # Expected header check (flexible)
    expected = ["Tweet ID","URL","Content","Likes","Retweets","Replies","Quotes","Views","Date"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print("Warning: missing columns:", missing)
    # drop duplicates by Tweet ID
    if "Tweet ID" in df.columns:
        df = df.drop_duplicates(subset=["Tweet ID"]).reset_index(drop=True)
    # normalize numeric columns
    for col in ["Likes","Retweets","Replies","Quotes","Views"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].fillna("0").astype(str).str.replace(r"[^\d]", "", regex=True).replace("", "0"), errors="coerce").fillna(0).astype(int)
    print("Cleaning text. This may take a while...")
    df["clean_text"] = df.get("Content", "").progress_apply(clean_text)
    # parse date
        # parse date with explicit format, fallback if needed
    if "Date" in df.columns:
        # first try: known format like "October 31, 2025 at 10:11 PM"
        fmt = "%B %d, %Y at %I:%M %p"
        try:
            parsed = pd.to_datetime(df["Date"], format=fmt, errors="coerce")
        except Exception:
            parsed = pd.to_datetime(df["Date"], errors="coerce")

        df["Date"] = parsed

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved cleaned data: {CLEAN_PATH} (rows={len(df)})")

if __name__ == "__main__":
    run_preprocess()
