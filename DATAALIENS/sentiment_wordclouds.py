#!/usr/bin/env python3
"""Generate separate word clouds for positive, neutral, and negative comments."""

from pathlib import Path
import re

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
INPUT_CSV = ROOT / "analysis_outputs" / "sentiment_analysis.csv"
OUTPUT_DIR = ROOT / "analysis_outputs" / "sentiment_wordclouds"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentiment thresholds (compound VADER scores)
# Sentiment thresholds (compound VADER scores)
# User requested: positive 0.5 to 1, neutral between -0.5 and 0.5, negative -1 to -0.5
POSITIVE_THRESHOLD = 0.5
NEGATIVE_THRESHOLD = -0.5


def clean_text(text: str) -> str:
    """Basic text cleaning for word cloud generation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_wordcloud(texts: list[str], title: str, filename: Path) -> None:
    combined = " ".join(filter(None, texts))
    if not combined.strip():
        print(f"[WARN] No text available for {title}; skipping.")
        return

    wc = WordCloud(
        width=1600,
        height=1000,
        background_color="white",
        colormap="viridis",
        max_words=300,
        random_state=42,
    ).generate(combined)

    plt.figure(figsize=(14, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved {title} word cloud to {filename}")


def main() -> None:
    print(f"Reading comment sentiment data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"text", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in sentiment CSV: {missing}")

    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["sentiment"])
    df["clean_text"] = df["text"].apply(clean_text)

    categories = {
        "positive": df[df["sentiment"] >= POSITIVE_THRESHOLD]["clean_text"].tolist(),
        "neutral": df[
            (df["sentiment"] > NEGATIVE_THRESHOLD) & (df["sentiment"] < POSITIVE_THRESHOLD)
        ]["clean_text"].tolist(),
        "negative": df[df["sentiment"] <= NEGATIVE_THRESHOLD]["clean_text"].tolist(),
    }

    for label, texts in categories.items():
        title = f"{label.capitalize()} Comments Word Cloud"
        filename = OUTPUT_DIR / f"{label}_wordcloud.png"
        build_wordcloud(texts, title, filename)


if __name__ == "__main__":
    main()

