#!/usr/bin/env python3
"""Generate per-video sentiment highlights (top positive/negative/neutral comments)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
INPUT_CSV = ROOT / "analysis_outputs" / "sentiment_analysis.csv"
OUTPUT_CSV = ROOT / "analysis_outputs" / "comment_sentiment_highlights.csv"


def pick_row(group: pd.DataFrame, column: str, agg: str) -> pd.Series:
    """Return row from group based on aggregation request."""
    if group.empty:
        return pd.Series(dtype=object)

    if agg == "max":
        idx = group[column].idxmax()
    elif agg == "min":
        idx = group[column].idxmin()
    elif agg == "absmin":
        idx = (group[column].abs()).idxmin()
    else:
        raise ValueError(f"Unsupported aggregation: {agg}")

    return group.loc[idx]


def main() -> None:
    print(f"Reading sentiment data from {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

    required_cols = {"videoWebUrl", "text", "sentiment", "createTimeISO", "video_average_sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in input CSV: {missing}")

    # Ensure correct dtypes
    df["sentiment"] = pd.to_numeric(df["sentiment"], errors="coerce")
    df = df.dropna(subset=["videoWebUrl", "sentiment", "text"])

    rows = []
    for video_url, group in df.groupby("videoWebUrl"):
        group = group.copy()
        group["text"] = group["text"].astype(str).str.strip()

        top_pos = pick_row(group, "sentiment", "max")
        top_neg = pick_row(group, "sentiment", "min")
        neutral = pick_row(group, "sentiment", "absmin")

        rows.append(
            {
                "videoWebUrl": video_url,
                "comment_count": len(group),
                "avg_comment_sentiment": group["video_average_sentiment"].iloc[0],
                "top_positive_sentiment": top_pos["sentiment"],
                "top_positive_text": top_pos["text"],
                "top_positive_time": top_pos["createTimeISO"],
                "top_negative_sentiment": top_neg["sentiment"],
                "top_negative_text": top_neg["text"],
                "top_negative_time": top_neg["createTimeISO"],
                "neutral_sentiment": neutral["sentiment"],
                "neutral_text": neutral["text"],
                "neutral_time": neutral["createTimeISO"],
            }
        )

    highlights = pd.DataFrame(rows).sort_values("avg_comment_sentiment", ascending=False)
    highlights.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Saved sentiment highlights to {OUTPUT_CSV} ({len(highlights)} rows).")


if __name__ == "__main__":
    main()

