#!/usr/bin/env python3
"""Render plays-vs-sentiment plot using emoji faces instead of markers."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
COMMENTS_PATH = ROOT / "dataset_tiktok-comments-scraper_2025-11-11_15-21-56-961.csv"
VIDEOS_PATH = ROOT / "dataset_tiktok-hashtag-scraper_2025-11-11_13-24-55-129.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "video_sentiment_faces.png"
OUTPUT_SVG = OUTPUT_DIR / "video_sentiment_faces.svg"
OUTPUT_DIR.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def normalize(series: pd.Series, min_val: float = 0.1, max_val: float = 1.0) -> pd.Series:
    """Min-max normalize to [min_val, max_val]."""
    if series.max() == series.min():
        return pd.Series(np.full(len(series), (min_val + max_val) / 2), index=series.index)
    scaled = (series - series.min()) / (series.max() - series.min())
    return min_val + scaled * (max_val - min_val)


def main() -> None:
    comments = pd.read_csv(COMMENTS_PATH, encoding="utf-8-sig")
    videos = pd.read_csv(VIDEOS_PATH, encoding="utf-8-sig")

    for col in ["diggCount", "shareCount", "playCount", "commentCount"]:
        videos[col] = pd.to_numeric(videos[col], errors="coerce")

    analyzer = SentimentIntensityAnalyzer()
    comments["sentiment"] = comments["text"].fillna("").astype(str).apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    agg = (
        comments.groupby("videoWebUrl")
        .agg(
            mean_comment_sentiment=("sentiment", "mean"),
            comment_volume=("cid", "count"),
        )
        .reset_index()
    )
    data = (
        videos.merge(
            agg,
            left_on="webVideoUrl",
            right_on="videoWebUrl",
            how="left",
        )
        .dropna(subset=["mean_comment_sentiment", "playCount"])
        .copy()
    )

    if data.empty:
        raise SystemExit("No merged sentiment data available.")

    data["comment_volume"] = data["comment_volume"].fillna(0)
    sentiment = data["mean_comment_sentiment"].clip(-1, 1)
    abs_size = normalize(sentiment.abs(), 0.1, 1.0)
    volume_alpha = normalize(data["comment_volume"], 0.25, 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Video Plays vs. Sentiment (Emoji Edition)")
    ax.set_xlabel("Avg Comment Sentiment (per video)")
    ax.set_ylabel("Play Count")
    ax.set_xlim(sentiment.min() - 0.05, sentiment.max() + 0.05)
    ax.set_ylim(0, data["playCount"].max() * 1.05)

    base_font = 22
    for (_, row), sent, alpha, size in zip(
        data.iterrows(), sentiment, volume_alpha, abs_size
    ):
        color = "green" if sent >= 0 else "red"
        dot_size = (base_font * size * 1.2) ** 2  # match circle area to emoji size
        ax.scatter(
            row["mean_comment_sentiment"],
            row["playCount"],
            s=dot_size,
            color=color,
            alpha=float(alpha) * 0.4,
            edgecolors="none",
        )
        emoji = "😊" if sent >= 0 else "☹️"
        ax.text(
            row["mean_comment_sentiment"],
            row["playCount"],
            emoji,
            fontsize=base_font * size,
            ha="center",
            va="center",
            alpha=float(alpha),
        )

    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.savefig(OUTPUT_SVG, dpi=200)
    plt.close()
    print(f"Saved emoji sentiment plot to {OUTPUT_PNG} and {OUTPUT_SVG}")


if __name__ == "__main__":
    main()

