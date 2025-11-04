# src/sentiment.py
import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

CLEAN_PATH = os.path.join("data", "biznet_tweets_clean.csv")
OUT_PATH = os.path.join("data", "biznet_tweets_sentiment.csv")

MODEL_NAME = "w11wo/indonesian-roberta-base-sentiment-classifier"

# if GPU available, use it (device=0), else -1 for CPU
DEVICE = 0 if torch.cuda.is_available() else -1
BATCH_SIZE = 32  # adjust according to memory

def normalize_label(label: str) -> str:
    if label is None:
        return "neutral"
    label_l = label.lower()
    if label_l in ["positive","pos","positif","positif"]:
        return "positive"
    if label_l in ["negative","neg","negatif"]:
        return "negative"
    if label_l in ["neutral","netral","none"]:
        return "neutral"
    # handle LABEL_0/LABEL_1 cases
    if label_l.startswith("label_"):
        # common mapping assumption: LABEL_0 neg, LABEL_1 neu, LABEL_2 pos
        mapping = {"label_0":"negative","label_1":"neutral","label_2":"positive"}
        return mapping.get(label_l, "neutral")
    return label_l

def run_inference(batch_size=BATCH_SIZE):
    if not os.path.exists(CLEAN_PATH):
        raise FileNotFoundError(f"{CLEAN_PATH} not found. Run preprocess first.")
    df = pd.read_csv(CLEAN_PATH, dtype=str)
    texts = df["clean_text"].fillna("").tolist()
    # setup pipeline
    nlp = pipeline("sentiment-analysis", model=MODEL_NAME, tokenizer=MODEL_NAME, device=DEVICE)
    sentiments = []
    scores = []
    print("Running sentiment inference ...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        preds = nlp(batch)
        for p in preds:
            lbl = p.get("label")
            scr = p.get("score", None)
            sentiments.append(normalize_label(lbl))
            scores.append(float(scr) if scr is not None else None)
    df["sentiment"] = sentiments
    df["sentiment_score"] = scores
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved sentiment results to {OUT_PATH} (rows={len(df)})")

if __name__ == "__main__":
    run_inference()
