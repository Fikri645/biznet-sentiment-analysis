# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import os
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# --- Paths & page setup ---
DATA_PATH = "data/biznet_tweets_sentiment_public.csv"
st.set_page_config(page_title="Biznet Sentiment Dashboard", layout="wide")
st.title("Biznet Sentiment Analysis (Twitter)")

# --- Utility: load data ---
@st.cache_data
def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    # convert numeric columns if present
    for c in ["Likes", "Retweets", "Replies", "Quotes", "Views"]:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c].fillna("0").astype(str).str.replace(r"[^\d]", "", regex=True).replace("", "0"),
                errors="coerce",
            ).fillna(0).astype(int)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()
if df.empty:
    st.warning("Data file not available. Please run preprocess + sentiment pipeline first.")
    st.stop()

# --- Stopwords setup: try Sastrawi else fallback manual ---
@st.cache_data
def get_stopwords(extra_list=None):
    try:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

        factory = StopWordRemoverFactory()
        sw = set(factory.get_stop_words())
    except Exception:
        # fallback minimal list
        sw = set(
            """yang dan di ke dari ini itu untuk dengan tidak sudah sangat banget 
            lebih bisa adalah pada oleh dalam jadi juga lagi karena namun tapi kalau ketika agar supaya 
            enggak gak ga ya yah dong deh sih gue gw aku saya kami kita lo lu mereka dia kau mu 
            nya kamu""".split()
        )
    extra = set(extra_list or [])
    brand_terms = {
        "biznet",
        "indihome",
        "indihomo",
        "myrep",
        "myrepublic",
        "wifi",
        "internet",
        "provider",
        "paket",
    }
    stopwords = sw.union(extra).union(brand_terms)
    return stopwords


# sidebar: user can add extra stopwords (pre-filled with defaults)
st.sidebar.header("Keyword Extraction Settings")

# default extra stopwords displayed in textarea (space-separated)
DEFAULT_EXTRA_SW = [
    # common uninformative words to exclude from keyword extraction
    "gue",
    "gw",
    "saya",
    "aku",
    "kita",
    "kami",
    "lo",
    "lu",
    "ya",
    "yah",
    "nih",
    "deh",
    "dong",
    "sih",
    "aja",
    "si",
    "apa",
    "sama",
    "ini",
    "itu",
    "yang",
    "ada",
    "udah",
    "sudah",
    "lagi",
    "dari",
    "ke",
    "untuk",
    "dengan",
    "di",
    "dan",
    "atau",
    "nya",
    "banget",
    "sangat",
    "lebih",
    "boleh",
    "bisa",
    "gak",
    "enggak",
    "ga",
    "pakai",
    "kalo",
    # brand/generic tokens
    "biznet",
    "indihome",
    "indihomo",
    "myrepublic",
    "myrep",
    "wifi",
    "internet",
    "provider",
    "paket",
]

# show default list as initial textarea content (user can edit)
default_sw_text = " ".join(DEFAULT_EXTRA_SW)
extra_sw_text = st.sidebar.text_area(
    "Add stopwords (space-separated)",
    value=default_sw_text,
    help="Tokens to exclude from keyword extraction, e.g. 'customer support'",
)

# button to reset to default if user modifies textarea
if st.sidebar.button("Reset to default stopwords"):
    extra_sw_text = default_sw_text
    # Streamlit will rerun

# create list of extra stopwords from textarea (space-separated)
extra_sw = [t.strip() for t in extra_sw_text.split() if t.strip()]

# build final STOPWORDS (combination of Sastrawi/fallback + extra + brand terms)
STOPWORDS = get_stopwords(extra_sw)

# sidebar filters
max_likes_default = int(df["Likes"].max()) if "Likes" in df.columns and not df["Likes"].isna().all() else 100
min_likes = st.sidebar.slider("Min Likes", 0, max(max_likes_default, 0), 0)

sentiment_options = sorted(df["sentiment"].unique()) if "sentiment" in df.columns else []
sentiment_choice = st.sidebar.multiselect("Sentiment", options=sentiment_options, default=sentiment_options)

# Toggle: use original Content tokens (minimal cleaning) instead of clean_text (normalized)
st.sidebar.markdown("---")
use_original = st.sidebar.checkbox(
    "Use ORIGINAL words from Content column (without normalization/slang/stemming)", value=False
)
st.sidebar.markdown("---")
st.sidebar.write("Tip: toggle above to see original user words (more useful for human insight).")

# apply filters
df_filtered = df.copy()
if "Likes" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["Likes"].astype(float).fillna(0) >= min_likes]
if sentiment_choice:
    df_filtered = df_filtered[df_filtered["sentiment"].isin(sentiment_choice)]

# --------------------
# Helper functions (new)
# --------------------
def raw_preprocess(text):
    """
    Minimal cleaning for original text tokenization:
      - remove URLs and mentions
      - remove leading '#' only (keep hashtag text)
      - reduce repeated chars (3+ -> 2)
      - remove punctuation except word chars and spaces
      - lowercase
    """
    if pd.isna(text) or text == "":
        return ""
    txt = str(text)
    txt = re.sub(r"http\S+", " ", txt)  # remove URLs
    txt = re.sub(r"@\w+", " ", txt)  # remove mentions
    txt = re.sub(r"#", "", txt)  # remove hashtag symbol but keep text
    txt = re.sub(r"(.)\1{2,}", r"\1\1", txt)  # reduce repeated chars
    txt = re.sub(r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF]", " ", txt)  # keep letters/digits/underscore/space
    txt = txt.lower()
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


@st.cache_data
def tokenize_and_filter(text, stopwords, use_original=False):
    """
    If use_original=True -> runs raw_preprocess and tokenizes preserving original tokens (no slang normalization).
    Else -> expects text already normalized/stemmed (clean_text) and tokenizes.
    Filters out tokens in stopwords and numeric-only tokens; keeps tokens length >2.
    """
    if pd.isna(text) or text == "":
        return []
    if use_original:
        txt = raw_preprocess(text)
    else:
        txt = str(text).lower()
    toks = [t for t in re.split(r"\s+", txt) if t and len(t) > 2]
    toks = [t for t in toks if not t.isdigit() and t not in stopwords]
    return toks


@st.cache_data
def top_unigrams(df_local, sentiment, top_n=20, stopwords=None, source_col="clean_text", use_original=False):
    stopwords = stopwords or set()
    texts = df_local[df_local["sentiment"] == sentiment][source_col].dropna().tolist()
    words = []
    for t in texts:
        words.extend(tokenize_and_filter(t, stopwords, use_original=use_original))
    return Counter(words).most_common(top_n)


@st.cache_data
def tfidf_top_terms(
    df_local, top_n=25, stopwords=None, max_features=5000, source_col="clean_text", use_original=False
):
    """
    TF-IDF computed on chosen source_col. If use_original=True we preprocess Content minimally (raw_preprocess)
    before vectorizing, so TF-IDF terms reflect original tokens.
    """
    stopwords = stopwords or set()
    # prepare corpus according to source
    if source_col == "Content" and use_original:
        corpus = df_local["Content"].fillna("").apply(raw_preprocess).tolist()
    else:
        corpus = df_local[source_col].fillna("").tolist()

    if len(corpus) == 0:
        return {}

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features, stop_words=list(stopwords))
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    # build mapping sentiment -> doc positions 0..N-1
    idx_map = {}
    for pos, sent in enumerate(df_local["sentiment"].fillna("").tolist()):
        idx_map.setdefault(sent, []).append(pos)

    results = {}
    for sentiment, positions in idx_map.items():
        pos_list = [int(p) for p in positions if 0 <= int(p) < X.shape[0]]
        if not pos_list:
            results[sentiment] = []
            continue
        group_vec = X[pos_list].mean(axis=0)
        arr = group_vec.A1
        top_idx = arr.argsort()[::-1][:top_n]
        results[sentiment] = [(feature_names[i], float(arr[i])) for i in top_idx]
    return results


@st.cache_data
def bigram_pmi_like(
    df_local, sentiment, top_n=20, min_df=2, stopwords=None, source_col="clean_text", use_original=False
):
    """
    Build bigrams from source_col.
    If use_original=True, build vocabulary from raw_preprocess(Content) (stopwords not used during fit).
    After building vocabulary, filter out bigrams where BOTH tokens are uninformative (stopwords/brand).
    """
    stopwords = stopwords or set()
    brand_terms = {"biznet", "indihome", "wifi", "internet", "provider", "paket"}
    stop_or_brand = set(stopwords).union(brand_terms)

    # build corpus according to source
    if source_col == "Content" and use_original:
        corpus = df_local["Content"].fillna("").apply(raw_preprocess).tolist()
    else:
        corpus = df_local[source_col].fillna("").tolist()

    if len(corpus) == 0:
        return []

    requested_min_df = int(max(1, min_df))
    effective_min_df = 1 if len(corpus) < 50 else min(requested_min_df, max(1, len(corpus) // 2))

    try:
        # if use_original=True -> DO NOT pass stop_words to CountVectorizer; else we can pass stopwords
        if source_col == "Content" and use_original:
            cv = CountVectorizer(ngram_range=(2, 2), min_df=effective_min_df)
        else:
            cv = CountVectorizer(ngram_range=(2, 2), min_df=effective_min_df, stop_words=list(stopwords))
        X = cv.fit_transform(corpus)
    except ValueError:
        return []
    except Exception:
        return []

    features = cv.get_feature_names_out()
    freqs = X.sum(axis=0).A1
    total_freq = freqs.sum() if freqs.sum() > 0 else 1

    mask = df_local["sentiment"] == sentiment
    if mask.sum() == 0:
        return []

    # transform group's docs
    if source_col == "Content" and use_original:
        X_group = cv.transform(df_local[mask]["Content"].fillna("").apply(raw_preprocess).tolist())
    else:
        X_group = cv.transform(df_local[mask][source_col].fillna("").tolist())

    freqs_group = X_group.sum(axis=0).A1
    group_total = freqs_group.sum() if freqs_group.sum() > 0 else 1

    pmi_scores = {}
    for i, term in enumerate(features):
        p_xy = freqs_group[i] / group_total if group_total > 0 else 0
        p_x = freqs[i] / total_freq if total_freq > 0 else 0
        score = (p_xy / (p_x + 1e-12)) if p_x > 0 else 0
        pmi_scores[term] = score

    sorted_terms = sorted(pmi_scores.items(), key=lambda x: x[1], reverse=True)

    # filter out bigrams where BOTH tokens are stop/brand
    filtered = []
    for term, score in sorted_terms:
        parts = term.split()
        if len(parts) != 2:
            filtered.append((term, score))
            continue
        t0, t1 = parts[0].lower(), parts[1].lower()
        if (t0 in stop_or_brand) and (t1 in stop_or_brand):
            continue
        filtered.append((term, score))
        if len(filtered) >= top_n:
            break

    return filtered[:top_n]


# --- Build main tabs ---
tab1, tab2, tab3 = st.tabs(["Dashboard", "Keyword Extraction", "Interpretation"])

# -----------------------
# Tab 1: Dashboard
# -----------------------
with tab1:
    st.header("Main Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Tweets (filtered)", len(df_filtered))
    pos_pct = (
        (df_filtered["sentiment"] == "positive").mean() * 100
        if "sentiment" in df_filtered.columns and len(df_filtered) > 0
        else 0.0
    )
    neg_pct = (
        (df_filtered["sentiment"] == "negative").mean() * 100
        if "sentiment" in df_filtered.columns and len(df_filtered) > 0
        else 0.0
    )
    col2.metric("Positive %", f"{pos_pct:.1f}%")
    col3.metric("Negative %", f"{neg_pct:.1f}%")

    # Pie chart
    if "sentiment" in df_filtered.columns and not df_filtered.empty:
        fig_pie = px.pie(df_filtered, names="sentiment", title="Sentiment Distribution", hole=0.4)
        st.plotly_chart(fig_pie, width="stretch")
    else:
        st.info("No data available to create pie chart with current filters.")

    # Trend
    if "Date" in df_filtered.columns and not df_filtered["Date"].isna().all():
        df_trend = df_filtered.dropna(subset=["Date"])
        if not df_trend.empty:
            trend = df_trend.groupby([pd.Grouper(key="Date", freq="W"), "sentiment"]).size().reset_index(name="count")
            fig_line = px.line(trend, x="Date", y="count", color="sentiment", title="Weekly Sentiment Trend")
            st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No data with valid dates for trend analysis.")
    else:
        st.info("'Date' column not found or invalid - skipping trend plot.")

    st.subheader("Sample Tweets (Safe sampling)")
    max_sample = min(20, max(1, len(df_filtered)))
    if max_sample <= 1:
        n = 1
    else:
        default_val = min(5, max_sample)
        n = st.slider("Number of sample tweets", 1, max_sample, default_val)
    if len(df_filtered) == 0:
        st.info("No tweets to display based on current filters.")
    else:
        k = min(n, len(df_filtered))
        sample_df = df_filtered[["Date", "Content", "clean_text", "sentiment", "Likes", "Retweets"]].sample(k).reset_index(drop=True)
        st.table(sample_df)
        
    # Wordclouds (filtered tokens)
    st.header("Wordcloud per Sentiment (after stopwords filter)")
    wc_cols = st.columns(2)
    for i, label in enumerate(["positive", "negative"]):
        with wc_cols[i]:
            st.subheader(label.capitalize())
            source_col = "Content" if use_original else "clean_text"
            if source_col not in df_filtered.columns:
                st.write(f"Column '{source_col}' not available. Run preprocessing.")
                continue
            if use_original:
                texts_list = df_filtered[df_filtered["sentiment"] == label]["Content"].dropna().apply(raw_preprocess).tolist()
                full_text = " ".join(texts_list)
                tokens = tokenize_and_filter(full_text, STOPWORDS, use_original=True)
            else:
                texts_list = df_filtered[df_filtered["sentiment"] == label]["clean_text"].dropna().tolist()
                full_text = " ".join(texts_list)
                tokens = tokenize_and_filter(full_text, STOPWORDS, use_original=False)

            filtered_text = " ".join(tokens)
            if not filtered_text.strip():
                st.write("No data available for wordcloud after filtering.")
                continue
            wc = WordCloud(width=800, height=400).generate(filtered_text)
            buf = io.BytesIO()
            plt.figure(figsize=(8, 4))
            plt.imshow(wc.to_array(), interpolation="bilinear")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            plt.close()
            buf.seek(0)
            st.image(buf)

# -----------------------
# Tab 2: Keyword Extraction
# -----------------------
with tab2:
    st.header("Keyword Extraction (Unigram, TF-IDF, Bigrams)")

    cols = st.columns([1, 1])
    with cols[0]:
        st.subheader("Top Unigram (filtered)")
        sel_sent = st.selectbox(
            "Select Sentiment for Unigram",
            options=sentiment_options,
            index=sentiment_options.index("negative") if "negative" in sentiment_options else 0,
        )
        source_col = "Content" if use_original else "clean_text"
        top_uni = top_unigrams(df_filtered, sel_sent, top_n=30, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
        if top_uni:
            uni_df = pd.DataFrame(top_uni, columns=["term", "count"])
            st.table(uni_df.head(25))
            fig_bar = px.bar(uni_df.head(15), x="count", y="term", orientation="h", title=f"Top Unigram ({sel_sent})")
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No unigrams available for selected sentiment.")

    with cols[1]:
        st.subheader("Top TF-IDF Terms (1-2 gram)")
        st.write("TF-IDF shows characteristic words/phrases for each sentiment (more robust than frequency).")
        tfidf_res = tfidf_top_terms(df_filtered, top_n=30, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
        for s in tfidf_res.keys():
            st.markdown(f"**{s.upper()} (top 6)**")
            items = tfidf_res[s][:6]
            if items:
                st.write(pd.DataFrame(items, columns=["term", "avg_tfidf"]))
            else:
                st.write("—")

    st.subheader("Top Bigrams (PMI-like)")
    sel_sent_bigram = st.selectbox(
        "Select Sentiment for Bigrams",
        options=sentiment_options,
        index=sentiment_options.index("negative") if "negative" in sentiment_options else 0,
    )
    bigrams = bigram_pmi_like(df_filtered, sel_sent_bigram, top_n=25, min_df=2, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
    if bigrams:
        bigram_df = pd.DataFrame(bigrams, columns=["bigram", "score"])
        st.table(bigram_df.head(20))
        fig_bi = px.bar(bigram_df.head(15), x="score", y="bigram", orientation="h", title=f"Top Bigrams (PMI-like) - {sel_sent_bigram}")
        st.plotly_chart(fig_bi, width="stretch")
    else:
        st.info("No bigrams meet the criteria (try lowering min_df or removing filters).")


# -----------------------
# Tab 3: Interpretation
# -----------------------
with tab3:
    st.header("Automatic Interpretation")
    # Compute or reuse keyword outputs
    source_col = "Content" if use_original else "clean_text"
    tfidf_res = tfidf_top_terms(df_filtered, top_n=30, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
    uni_neg = top_unigrams(df_filtered, "negative", top_n=25, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
    uni_pos = top_unigrams(df_filtered, "positive", top_n=25, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
    bigrams_neg = bigram_pmi_like(df_filtered, "negative", top_n=20, min_df=2, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)
    bigrams_pos = bigram_pmi_like(df_filtered, "positive", top_n=20, min_df=2, stopwords=STOPWORDS, source_col=source_col, use_original=use_original)

    # Basic metrics
    total = len(df_filtered)
    pct_neg = (df_filtered["sentiment"] == "negative").mean() * 100 if total > 0 else 0.0
    pct_pos = (df_filtered["sentiment"] == "positive").mean() * 100 if total > 0 else 0.0
    pct_neu = (df_filtered["sentiment"] == "neutral").mean() * 100 if total > 0 else 0.0

    st.subheader("Quantitative Summary")
    st.write(f"- Total tweets (filtered): **{total}**")
    st.write(f"- Percentages: **Negative {pct_neg:.1f}%**, **Neutral {pct_neu:.1f}%**, **Positive {pct_pos:.1f}%**")

    # Build interpretative paragraphs
    def top_terms_list(lst, n=5):
        return ", ".join([t for t, *_ in lst[:n]])

    # Identify themes
    neg_themes = [t for t, _ in uni_neg[:8]]
    pos_themes = [t for t, _ in uni_pos[:8]]
    neg_bi = [t for t, _ in bigrams_neg[:6]]
    pos_bi = [t for t, _ in bigrams_pos[:6]]

    # Compose interpretation text (Markdown)
    lines = []
    lines.append(f"# Automatic Interpretation — Biznet (Twitter) — {pd.Timestamp.now().strftime('%Y-%m-%d')}\n")
    lines.append("## Main Summary")
    lines.append(
        f"- From **{total}** analyzed tweets (after filtering), **{pct_neg:.1f}%** are potentially negative, **{pct_pos:.1f}%** positive, and **{pct_neu:.1f}%** neutral."
    )
    lines.append("\n## Main Negative Themes")
    if uni_neg:
        lines.append(f"- Top negative words: **{top_terms_list(uni_neg, 8)}**.")
    if neg_bi:
        lines.append(f"- Characteristic negative phrases: **{', '.join(neg_bi)}**.")
    # Add quick insight sentences
    if "mati" in [w for w, _ in uni_neg[:50]] or any("putus" in b for b, _ in bigrams_neg[:50]):
        lines.append("- Insight: many complaints about **ganggu mulu / jelek sinyal** — indicating network stability issues.")
    if "ganggu" in [w for w, _ in uni_neg[:50]] or any("gangguan" in w for w, _ in uni_neg[:50]):
        lines.append("- Insight: pattern of **ganggu / gangguan** words — indicates high frequency of service disruptions.")
    # Positive
    lines.append("\n## Main Positive Themes")
    if uni_pos:
        lines.append(f"- Top positive words: **{top_terms_list(uni_pos, 8)}**.")
    if pos_bi:
        lines.append(f"- Characteristic positive phrases: **{', '.join(pos_bi)}**.")
    if "cepat" in [w for w, _ in uni_pos[:50]] or "stabil" in [w for w, _ in uni_pos[:50]]:
        lines.append("- Insight: satisfied users frequently mention **cepat** and **aman** — these are service strengths that can be promoted.")
    # Trend insight
    lines.append("\n## Trends & Timing")
    try:
        if "Date" in df_filtered.columns and not df_filtered["Date"].isna().all():
            df_dates = df_filtered.dropna(subset=["Date"]).copy()
            df_week = df_dates.groupby([pd.Grouper(key="Date", freq="W"), "sentiment"]).size().reset_index(name="count")
            neg_week = df_week[df_week["sentiment"] == "negative"].sort_values("count", ascending=False).head(1)
            if not neg_week.empty:
                wk = neg_week.iloc[0]
                lines.append(f"- Detected complaint spike in week **{pd.to_datetime(wk['Date']).date()}** with **{int(wk['count'])}** negative tweets.")
    except Exception:
        pass

    # Recommendations
    lines.append("\n## Brief Recommendations")
    lines.append("1. Prioritize investigation of top negative terms/phrases (e.g. gangguan, jelek bgt).")
    lines.append("2. Enhance speed & uptime monitoring; publish status/maintenance schedules when needed.")
    lines.append("3. Use positive comments (fast, stable) as promotional material; strengthen customer service for quick complaint handling.")

    interpretation_md = "\n".join(lines)

    st.markdown("### Automatic Summary (preview)")
    st.markdown(interpretation_md)

    # Download button (markdown)
    st.download_button("Download Interpretation (Markdown)", data=interpretation_md, file_name="biznet_interpretation.md", mime="text/markdown")

    # Also show quick tables for reference
    with st.expander("View keyword output (unigram TF-IDF bigrams)"):
        st.subheader("Top Unigram Negative (filtered)")
        if uni_neg:
            st.table(pd.DataFrame(uni_neg, columns=["term", "count"]).head(20))
        else:
            st.write("No data available.")
        st.subheader("Top Unigram Positive (filtered)")
        if uni_pos:
            st.table(pd.DataFrame(uni_pos, columns=["term", "count"]).head(20))
        else:
            st.write("No data available.")
        st.subheader("TF-IDF (sample top 8 per sentiment)")
        for s, items in tfidf_res.items():
            st.markdown(f"**{s.upper()}**")
            st.write(pd.DataFrame(items[:8], columns=["term", "avg_tfidf"]))

        st.subheader("Top Bigrams Negative (PMI-like)")
        if bigrams_neg:
            st.table(pd.DataFrame(bigrams_neg, columns=["bigram", "score"]).head(15))
        else:
            st.write("No bigrams available.")
        st.subheader("Top Bigrams Positive (PMI-like)")
        if bigrams_pos:
            st.table(pd.DataFrame(bigrams_pos, columns=["bigram", "score"]).head(15))
        else:
            st.write("No bigrams available.")
