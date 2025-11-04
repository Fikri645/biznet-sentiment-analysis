# src/utils.py
import re
import pandas as pd
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

# load slang map from CSV at runtime
def load_slang_map(path="data/slang.csv"):
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str).fillna("")
    # expect header slang,formal
    pairs = dict(zip(df['slang'].str.lower().str.strip(), df['formal'].str.lower().str.strip()))
    return pairs

SLANG_MAP = load_slang_map()

def normalize_slang_tokens(tokens):
    """Given list of tokens, map slang -> formal using SLANG_MAP."""
    return [SLANG_MAP.get(t, t) for t in tokens]

def reduce_repetition(text):
    # reduce 3+ repeated chars to two (so heyyy -> heyy)
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text(raw: str) -> str:
    """
    Cleaning pipeline:
      - remove urls, mentions, hashtags (decision: remove hashtags)
      - remove RT
      - reduce excessive repetition
      - remove non-letter unicode (keep basic latin + latin extended)
      - lower-case
      - normalize slang via SLANG_MAP
      - stemming (Sastrawi)
    """
    if pd.isna(raw):
        return ""
    text = str(raw)
    text = re.sub(r"http\S+", " ", text)        # remove urls
    text = re.sub(r"@\w+", " ", text)           # remove mentions
    text = re.sub(r"#\w+", " ", text)           # remove hashtags entirely
    text = re.sub(r"\bRT\b", " ", text)         # remove RT tokens
    text = reduce_repetition(text)
    # remove unwanted unicode symbols/emojis (we keep letters and spaces)
    text = re.sub(r"[^0-9A-Za-z\u00C0-\u024F\u1E00-\u1EFF\s]", " ", text)
    text = text.lower().strip()
    # split tokens and normalize slang
    tokens = [t for t in re.split(r"\s+", text) if t]
    tokens = normalize_slang_tokens(tokens)
    text = " ".join(tokens)
    # stemming
    try:
        text = stemmer.stem(text)
    except Exception:
        pass
    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
