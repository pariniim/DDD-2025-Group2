#!/usr/bin/env python3
"""Sentiment analysis on TikTok comments with visualizations."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
import numpy as np

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
COMMENTS_PATH = ROOT / "tiktok25ComentsPerVideo.csv"
VIDEOS_PATH = ROOT / "dataset_tiktok-hashtag-scraper_2025-11-11_13-24-55-129.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_CSV = OUTPUT_DIR / "sentiment_analysis.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_face(sentiment: float) -> str:
    """Get emoji face based on sentiment score."""
    if sentiment <= -0.5:
        return "😞"  # Bad
    elif sentiment <= 0:
        return "😐"  # Neutral-negative
    elif sentiment <= 0.5:
        return "🙂"  # Neutral-positive
    else:
        return "😊"  # Good


def get_sentiment_color(sentiment: float) -> str:
    """Get background color based on sentiment score."""
    if sentiment <= -0.5:
        return "#FF0000"  # Red
    elif sentiment <= 0:
        return "#FFA500"  # Orange
    elif sentiment <= 0.5:
        return "#00FFFF"  # Cyan
    else:
        return "#00FF00"  # Green


def normalize(series: pd.Series, min_val: float = 0.1, max_val: float = 1.0) -> pd.Series:
    """Min-max normalize to [min_val, max_val]."""
    if series.max() == series.min():
        return pd.Series(np.full(len(series), (min_val + max_val) / 2), index=series.index)
    scaled = (series - series.min()) / (series.max() - series.min())
    return min_val + scaled * (max_val - min_val)


def main() -> None:
    """Perform sentiment analysis and create visualizations."""
    print("Reading data files...")
    comments = pd.read_csv(COMMENTS_PATH, encoding="utf-8-sig")
    videos = pd.read_csv(VIDEOS_PATH, encoding="utf-8-sig")
    
    # Clean and process comments
    print("Analyzing sentiment...")
    comments["text"] = comments["text"].fillna("").astype(str)
    comments["sentiment"] = comments["text"].apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    
    # Calculate average sentiment per video
    video_sentiment = (
        comments.groupby("videoWebUrl")
        .agg(
            average_sentiment=("sentiment", "mean"),
            comment_count=("cid", "count"),
            min_sentiment=("sentiment", "min"),
            max_sentiment=("sentiment", "max"),
            std_sentiment=("sentiment", "std")
        )
        .reset_index()
    )
    
    # Add individual comment sentiments to output
    comments_output = comments[["text", "videoWebUrl", "sentiment", "diggCount", "createTimeISO"]].copy()
    comments_output = comments_output.merge(
        video_sentiment[["videoWebUrl", "average_sentiment"]],
        on="videoWebUrl",
        how="left"
    )
    comments_output.rename(columns={"average_sentiment": "video_average_sentiment"}, inplace=True)
    
    # Save sentiment analysis CSV
    print(f"Saving sentiment analysis to {OUTPUT_CSV}")
    comments_output.to_csv(OUTPUT_CSV, index=False)
    
    # Process video data
    for col in ["diggCount", "shareCount", "playCount", "commentCount"]:
        if col in videos.columns:
            videos[col] = pd.to_numeric(videos[col], errors="coerce")
    
    # Merge video data with sentiment
    # Match videos by URL (videoWebUrl in comments, webVideoUrl in videos)
    data = videos.merge(
        video_sentiment,
        left_on="webVideoUrl",
        right_on="videoWebUrl",
        how="inner"  # Only videos that have comments
    ).copy()
    
    if data.empty:
        raise SystemExit("No matching videos found between comments and video data.")
    
    print(f"Found {len(data)} videos with comments")
    
    # Parse dates
    data["createTimeISO"] = pd.to_datetime(data["createTimeISO"], errors="coerce")
    data = data.dropna(subset=["createTimeISO", "average_sentiment"])
    
    # Add sentiment faces and colors
    data["sentiment_face"] = data["average_sentiment"].apply(get_sentiment_face)
    data["sentiment_color"] = data["average_sentiment"].apply(get_sentiment_color)
    
    # Normalize sizes for visualization
    base_font = 20
    sentiment_abs = data["average_sentiment"].abs()
    size_normalized = normalize(sentiment_abs, 0.3, 1.0)
    
    # Create visualizations for each metric
    metrics = [
        ("diggCount", "Likes", "Number of Likes"),
        ("commentCount", "Comments", "Number of Comments"),
        ("playCount", "Play Count", "Number of Plays"),
    ]
    
    for metric_col, metric_name, ylabel in metrics:
        if metric_col not in data.columns:
            print(f"Warning: {metric_col} not found in video data, skipping...")
            continue
        
        # Filter out NaN values
        plot_data = data.dropna(subset=[metric_col, "average_sentiment"]).copy()
        
        if plot_data.empty:
            print(f"Warning: No data for {metric_name}, skipping...")
            continue
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_title(f"{metric_name} vs. Average Comment Sentiment", fontsize=16, pad=20)
        ax.set_xlabel("Average Comment Sentiment", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Plot each point with emoji and colored background
        for idx, row in plot_data.iterrows():
            sentiment = row["average_sentiment"]
            metric_value = row[metric_col]
            face = row["sentiment_face"]
            color = row["sentiment_color"]
            size = size_normalized.loc[idx] if idx in size_normalized.index else 0.5
            
            # Draw colored circle background
            circle = plt.Circle(
                (sentiment, metric_value),
                radius=0.05 * size,
                color=color,
                alpha=0.3,
                zorder=1
            )
            ax.add_patch(circle)
            
            # Add emoji face
            ax.text(
                sentiment,
                metric_value,
                face,
                fontsize=base_font * size,
                ha="center",
                va="center",
                zorder=2
            )
        
        # Set axis limits
        ax.set_xlim(
            plot_data["average_sentiment"].min() - 0.1,
            plot_data["average_sentiment"].max() + 0.1
        )
        ax.set_ylim(0, plot_data[metric_col].max() * 1.1)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color='#FF0000', label='Bad (-1 to -0.5)'),
            mpatches.Patch(color='#FFA500', label='Neutral-Negative (-0.5 to 0)'),
            mpatches.Patch(color='#00FFFF', label='Neutral-Positive (0 to 0.5)'),
            mpatches.Patch(color='#00FF00', label='Good (0.5 to 1)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        output_file = OUTPUT_DIR / f"sentiment_vs_{metric_col.lower()}.png"
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved {metric_name} visualization to {output_file}")
    
    # Create date visualization
    if "createTimeISO" in data.columns:
        plot_data = data.dropna(subset=["createTimeISO", "average_sentiment"]).copy()
        plot_data = plot_data.sort_values("createTimeISO")
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_title("Average Comment Sentiment Over Time", fontsize=16, pad=20)
            ax.set_xlabel("Date of Publishing", fontsize=12)
            ax.set_ylabel("Average Comment Sentiment", fontsize=12)
            
            # Calculate date range for radius calculation
            date_range = (plot_data["createTimeISO"].max() - plot_data["createTimeISO"].min()).total_seconds()
            radius_base = date_range / 100  # Base radius as fraction of date range
            
            # Plot each point with emoji and colored background
            for idx, row in plot_data.iterrows():
                date = row["createTimeISO"]
                sentiment = row["average_sentiment"]
                face = row["sentiment_face"]
                color = row["sentiment_color"]
                size = size_normalized.loc[idx] if idx in size_normalized.index else 0.5
                
                # Use scatter plot with colored markers for background
                ax.scatter(
                    date,
                    sentiment,
                    s=(base_font * size * 15) ** 2,
                    c=color,
                    alpha=0.3,
                    zorder=1,
                    edgecolors='none'
                )
                
                # Add emoji face
                ax.text(
                    date,
                    sentiment,
                    face,
                    fontsize=base_font * size,
                    ha="center",
                    va="center",
                    zorder=2
                )
            
            # Format x-axis dates
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Set axis limits
            ax.set_ylim(
                plot_data["average_sentiment"].min() - 0.1,
                plot_data["average_sentiment"].max() + 0.1
            )
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='#FF0000', label='Bad (-1 to -0.5)'),
                mpatches.Patch(color='#FFA500', label='Neutral-Negative (-0.5 to 0)'),
                mpatches.Patch(color='#00FFFF', label='Neutral-Positive (0 to 0.5)'),
                mpatches.Patch(color='#00FF00', label='Good (0.5 to 1)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            
            output_file = OUTPUT_DIR / "sentiment_vs_date.png"
            plt.savefig(output_file, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved date visualization to {output_file}")
    
    print(f"\nSentiment analysis complete!")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

