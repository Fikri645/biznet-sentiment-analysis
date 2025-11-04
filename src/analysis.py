# src/analysis.py
"""
Improved analysis for sentiment keywords:
- filter stopwords (Sastrawi + extras)
- compute filtered unigrams
- compute TF-IDF top terms (1-2 grams) per sentiment
- compute bigram "PMI-like" scores
- save basic plots (distribution + trend)
Usage:
    python -m src.analysis
"""
import os
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Sastrawi stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Paths
DATA_PATH = os.path.join("data", "biznet_tweets_sentiment.csv")
OUT_DIR = os.path.join("data", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# prepare stopwords
factory = StopWordRemoverFactory()
sastrawi_stopwords = set(factory.get_stop_words())

# additional stopwords commonly non-informative for keyword extraction
EXTRA_STOPWORDS = set([

])

STOPWORDS = sastrawi_stopwords.union(EXTRA_STOPWORDS)

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, dtype=str)
    # ensure types for engagement columns if present
    for c in ["Likes", "Retweets", "Replies", "Quotes", "Views"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).astype(int)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # ensure clean_text column exists
    if "clean_text" not in df.columns:
        raise ValueError("clean_text column not found. Run preprocessing first.")
    # ensure sentiment column exists
    if "sentiment" not in df.columns:
        raise ValueError("sentiment column not found. Run sentiment inference first.")
    return df

def tokenize_simple(text):
    """Split on whitespace, keep tokens longer than 2 chars."""
    if pd.isna(text):
        return []
    toks = [t for t in re.split(r"\s+", str(text)) if t and len(t) > 2]
    # filter tokens that are purely numeric
    toks = [t for t in toks if not t.isdigit()]
    return toks

def top_unigrams_filtered(df, sentiment='negative', top_n=20, stopwords=STOPWORDS):
    texts = df[df['sentiment'] == sentiment]['clean_text'].dropna().tolist()
    words = []
    for t in texts:
        toks = tokenize_simple(t)
        toks = [w for w in toks if w not in stopwords]
        words.extend(toks)
    return Counter(words).most_common(top_n)

def top_tfidf_terms(df, top_n=30, max_features=5000, stopwords=STOPWORDS):
    """
    Compute TF-IDF on whole corpus (1-2 grams), then return top terms per sentiment
    as average TF-IDF score across documents in that sentiment.
    Returns: dict sentiment -> list[(term, avg_score)]
    """
    corpus = df['clean_text'].fillna("").tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=max_features, stop_words=list(stopwords))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # map docs to original index
    idx_map = df.reset_index()[['index','sentiment']].groupby('sentiment')['index'].apply(list).to_dict()
    results = {}
    for sentiment, idxs in idx_map.items():
        if not idxs:
            results[sentiment] = []
            continue
        # compute mean tfidf vector for the group
        group_vec = X[idxs].mean(axis=0)
        arr = group_vec.A1  # convert to 1d
        top_idx = arr.argsort()[::-1][:top_n]
        results[sentiment] = [(feature_names[i], float(arr[i])) for i in top_idx]
    return results

def top_bigrams_pmi_like(df, sentiment='negative', top_n=20, min_df=3, stopwords=STOPWORDS):
    """
    Estimate bigrams that are more characteristic for a sentiment.
    Approach:
      - Count bigram frequencies overall and inside the sentiment
      - Compute (p_bigram_in_sentiment / p_bigram_overall) as a PMI-like score
    """
    corpus = df['clean_text'].fillna("").tolist()
    cv = CountVectorizer(ngram_range=(2,2), min_df=min_df, stop_words=list(stopwords))
    X = cv.fit_transform(corpus)
    features = cv.get_feature_names_out()
    freqs = X.sum(axis=0).A1
    total_freq = freqs.sum() if freqs.sum() > 0 else 1

    # frequency in group
    mask = (df['sentiment'] == sentiment)
    if mask.sum() == 0:
        return []
    X_group = cv.transform(df[mask]['clean_text'].fillna("").tolist())
    freqs_group = X_group.sum(axis=0).A1
    group_total = freqs_group.sum() if freqs_group.sum() > 0 else 1

    pmi_scores = {}
    for i, term in enumerate(features):
        p_xy = freqs_group[i] / group_total if group_total > 0 else 0
        p_x = freqs[i] / total_freq if total_freq > 0 else 0
        score = (p_xy / (p_x + 1e-12)) if p_x > 0 else 0
        pmi_scores[term] = score
    top = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top

def plot_distribution(df, out_path=os.path.join(OUT_DIR, "sentiment_distribution.png")):
    order = ["positive", "neutral", "negative"]
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x="sentiment", order=[o for o in order if o in df['sentiment'].unique()])
    plt.title("Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

def plot_trend(df, out_path=os.path.join(OUT_DIR, "sentiment_trend.png"), freq='W'):
    if "Date" not in df.columns:
        print("No Date column; skipping trend plot.")
        return
    df2 = df.dropna(subset=['Date'])
    if df2.empty:
        print("No valid dates; skipping trend.")
        return
    trend = df2.groupby([pd.Grouper(key='Date', freq=freq), 'sentiment']).size().reset_index(name='count')
    plt.figure(figsize=(10,5))
    sns.lineplot(data=trend, x='Date', y='count', hue='sentiment')
    plt.title("Sentiment Trend")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved", out_path)

def sample_examples(df, sentiment='negative', n=10):
    sub = df[df['sentiment'] == sentiment]
    if sub.empty:
        return pd.DataFrame()
    return sub.sample(min(n, len(sub)))[['Date','Content','clean_text','sentiment','Likes','Retweets']].reset_index(drop=True)

def main():
    print("Loading data...")
    df = load_data()
    total = len(df)
    dist = df['sentiment'].value_counts(normalize=True).mul(100).round(2)
    print(f"Total tweets: {total}")
    print("Sentiment distribution (%):")
    print(dist)

    # Save basic plots
    plot_distribution(df)
    plot_trend(df)

    # Top unigrams (filtered)
    print("\nTop unigram (filtered) per sentiment (sample):")
    for s in ['negative', 'positive']:
        top_uni = top_unigrams_filtered(df, sentiment=s, top_n=30)
        print(f"\n{s.upper()} top unigrams (filtered):")
        print(top_uni[:30])

    # TF-IDF top terms (1-2 grams) per sentiment
    print("\nComputing TF-IDF top terms (this may take a bit)...")
    tfidf_results = top_tfidf_terms(df, top_n=30)
    for s, terms in tfidf_results.items():
        print(f"\nTF-IDF top for {s} (top 15):")
        for term,score in terms[:15]:
            print(f"  {term} ({score:.4f})")

    # Bigrams PMI-like
    print("\nTop bigrams (PMI-like) per sentiment:")
    for s in ['negative','positive']:
        bigrams = top_bigrams_pmi_like(df, sentiment=s, top_n=20)
        print(f"\n{s} bigrams:")
        print(bigrams[:20])

    # Show sample tweets per sentiment to validate
    print("\nSample tweets (negative):")
    print(sample_examples(df, 'negative', n=8).to_string(index=False))

    print("\nSample tweets (positive):")
    print(sample_examples(df, 'positive', n=8).to_string(index=False))

if __name__ == "__main__":
    main()
