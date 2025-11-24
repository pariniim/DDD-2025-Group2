#!/usr/bin/env python3
"""Generate a word cloud from all TikTok comments (no sentiment filtering)."""

from pathlib import Path
import re

import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
COMMENTS_CSV = ROOT / "analysis_outputs" / "sentiment_analysis.csv"
OUTPUT_DIR = ROOT / "analysis_outputs" / "comment_wordclouds_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """Basic text cleaning for word cloud generation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main() -> None:
    print(f"Reading all comments from {COMMENTS_CSV}")
    df = pd.read_csv(COMMENTS_CSV, encoding="utf-8-sig")

    if "text" not in df.columns:
        raise SystemExit("Input CSV must contain a 'text' column.")

    all_text = " ".join(df["text"].astype(str).apply(clean_text))
    if not all_text.strip():
        raise SystemExit("No comment text available to generate word cloud.")

    print("Generating overall word cloud...")
    wordcloud = WordCloud(
        width=1800,
        height=1100,
        background_color="white",
        max_words=400,
        colormap="viridis",
        random_state=42,
    ).generate(all_text)

    plt.figure(figsize=(16, 9))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud from All TikTok Comments", fontsize=22, pad=20)
    plt.tight_layout()

    output_png = OUTPUT_DIR / "all_comments_wordcloud.png"
    plt.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved overall word cloud to {output_png}")


if __name__ == "__main__":
    main()

