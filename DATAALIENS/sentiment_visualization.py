#!/usr/bin/env python3
"""Sentiment analysis on TikTok comments with improved visualizations."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, FancyBboxPatch
import matplotlib.dates as mdates
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from PIL import Image as PILImage
import io
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
COMMENTS_PATH = ROOT / "tiktok25ComentsPerVideo.csv"
VIDEOS_PATH = ROOT / "dataset_tiktok-hashtag-scraper_2025-11-11_13-24-55-129.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_CSV = OUTPUT_DIR / "video_sentiment_analysis.csv"
OUTPUT_DIR.mkdir(exist_ok=True)

analyzer = SentimentIntensityAnalyzer()


def get_sentiment_face_type(sentiment: float) -> str:
    """Get face type based on sentiment score."""
    if sentiment <= -0.5:
        return "angry"  # Angry face
    elif sentiment <= 0:
        return "unhappy"  # Unhappy face
    elif sentiment <= 0.5:
        return "neutral"  # Neutral face
    else:
        return "happy"  # Happy face


def create_simple_face_image(face_type: str, size: int) -> np.ndarray:
    """Create a simple black line art face (like a font icon)."""
    # Create figure and axis for drawing
    fig = plt.figure(figsize=(1, 1), dpi=size, facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Simple black outline circle for face
    face_circle = Circle((0.5, 0.5), 0.4, fill=False, ec='black', lw=2.5, zorder=1)
    ax.add_patch(face_circle)
    
    # Eyes (always two dots)
    eye_size = 0.06
    eye_y = 0.55
    left_eye = Circle((0.35, eye_y), eye_size, color='black', zorder=2)
    right_eye = Circle((0.65, eye_y), eye_size, color='black', zorder=2)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)
    
    # Mouth based on face type
    if face_type == "happy":
        # Smile arc
        smile = Arc((0.5, 0.4), 0.3, 0.2, angle=0, theta1=180, theta2=0, 
                   color='black', lw=3, zorder=2)
        ax.add_patch(smile)
    elif face_type == "neutral":
        # Straight line
        ax.plot([0.35, 0.65], [0.35, 0.35], 'k-', lw=3, zorder=2)
    elif face_type == "unhappy":
        # Frown arc
        frown = Arc((0.5, 0.3), 0.3, 0.2, angle=0, theta1=0, theta2=180, 
                   color='black', lw=3, zorder=2)
        ax.add_patch(frown)
    elif face_type == "angry":
        # Angry eyebrows (angled lines)
        ax.plot([0.25, 0.4], [0.7, 0.65], 'k-', lw=3, zorder=2)
        ax.plot([0.6, 0.75], [0.65, 0.7], 'k-', lw=3, zorder=2)
        # Frown arc
        frown = Arc((0.5, 0.3), 0.3, 0.2, angle=0, theta1=0, theta2=180, 
                   color='black', lw=3, zorder=2)
        ax.add_patch(frown)
    
    # Save to buffer and read back (transparent background)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=size, bbox_inches='tight', 
                pad_inches=0, facecolor='none', edgecolor='none', 
                transparent=True)
    buf.seek(0)
    img = PILImage.open(buf)
    img_array = np.array(img)
    plt.close(fig)
    buf.close()
    
    return img_array


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def interpolate_color(color1: tuple, color2: tuple, factor: float) -> tuple:
    """Interpolate between two RGB colors."""
    return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))


def get_sentiment_color(sentiment: float) -> str:
    """Get gradient color based on sentiment score using pivot colors.
    
    Pivot colors:
    - Red (#FF0000) at -1.0
    - Orange (#FFA500) at -0.5
    - Blue (#0000FF) at 0.0
    - Green (#00FF00) at 0.5
    - Green (#00FF00) at 1.0
    """
    # Clamp sentiment to [-1, 1]
    sentiment = max(-1.0, min(1.0, sentiment))
    
    # Define pivot colors
    red = hex_to_rgb("#FF0000")
    orange = hex_to_rgb("#FFA500")
    blue = hex_to_rgb("#0000FF")
    green = hex_to_rgb("#00FF00")
    
    # Interpolate based on sentiment value
    if sentiment <= -0.5:
        # Red to Orange: -1.0 to -0.5
        # Map sentiment from [-1, -0.5] to [0, 1]
        factor = (sentiment + 1.0) / 0.5
        rgb = interpolate_color(red, orange, factor)
    elif sentiment <= 0.0:
        # Orange to Blue: -0.5 to 0.0
        # Map sentiment from [-0.5, 0] to [0, 1]
        factor = (sentiment + 0.5) / 0.5
        rgb = interpolate_color(orange, blue, factor)
    elif sentiment <= 0.5:
        # Blue to Green: 0.0 to 0.5
        # Map sentiment from [0, 0.5] to [0, 1]
        factor = sentiment / 0.5
        rgb = interpolate_color(blue, green, factor)
    else:
        # Green (stays green for > 0.5)
        rgb = green
    
    return rgb_to_hex(rgb)


def normalize(series: pd.Series, min_val: float = 0.1, max_val: float = 1.0) -> pd.Series:
    """Min-max normalize to [min_val, max_val]."""
    if series.max() == series.min():
        return pd.Series(np.full(len(series), (min_val + max_val) / 2), index=series.index)
    scaled = (series - series.min()) / (series.max() - series.min())
    return min_val + scaled * (max_val - min_val)


def calculate_circle_radius_data_coords(ax, fontsize_points):
    """Calculate circle radius in data coordinates.
    
    Background should be exactly 2x the emoji size.
    Uses scatter plot size calculation (size is in points^2).
    """
    # Get figure and axis dimensions
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_inches = bbox.width
    height_inches = bbox.height
    
    # Get axis limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # Calculate data-to-inches conversion
    x_scale = (xlim[1] - xlim[0]) / width_inches
    y_scale = (ylim[1] - ylim[0]) / height_inches
    
    # Use the smaller scale to ensure circular appearance
    scale = min(x_scale, y_scale)
    
    # Emoji size in points, background should be 2x
    background_size_points = fontsize_points * 2
    
    # Convert points to inches (1 point = 1/72 inches)
    background_size_inches = background_size_points / 72.0
    
    # Convert to data coordinates
    radius_data = background_size_inches * scale
    
    return radius_data


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
    data = videos.merge(
        video_sentiment,
        left_on="webVideoUrl",
        right_on="videoWebUrl",
        how="inner"
    ).copy()
    
    if data.empty:
        raise SystemExit("No matching videos found between comments and video data.")
    
    print(f"Found {len(data)} videos with comments")
    
    # Parse dates
    data["createTimeISO"] = pd.to_datetime(data["createTimeISO"], errors="coerce")
    data = data.dropna(subset=["createTimeISO", "average_sentiment"])
    
    # Add sentiment face types and colors
    data["sentiment_face_type"] = data["average_sentiment"].apply(get_sentiment_face_type)
    data["sentiment_color"] = data["average_sentiment"].apply(get_sentiment_color)
    
    # Normalize sizes for visualization (based on absolute sentiment)
    sentiment_abs = data["average_sentiment"].abs()
    size_normalized = normalize(sentiment_abs, 0.4, 1.0)
    
    # Base font size for emojis
    base_font = 16
    
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
        
        # Set axis limits first
        x_min = plot_data["average_sentiment"].min() - 0.1
        x_max = plot_data["average_sentiment"].max() + 0.1
        y_min = 0
        y_max = plot_data[metric_col].max() * 1.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        # Prepare data for plotting
        sentiments = []
        metric_values = []
        colors = []
        marker_sizes = []
        
        for idx, row in plot_data.iterrows():
            sentiment = row["average_sentiment"]
            metric_value = row[metric_col]
            color = row["sentiment_color"]
            size = size_normalized.loc[idx] if idx in size_normalized.index else 0.5
            
            # Circle size based on sentiment magnitude
            marker_size = (base_font * size * 2) ** 2
            
            sentiments.append(sentiment)
            metric_values.append(metric_value)
            colors.append(color)
            marker_sizes.append(marker_size)
        
        # Plot colored circles only (no faces)
        for i in range(len(sentiments)):
            ax.scatter(
                sentiments[i],
                metric_values[i],
                s=marker_sizes[i],
                c=colors[i],
                alpha=0.6,
                zorder=1,
                edgecolors='black',
                linewidths=0.5
            )
        
        # Add legend with gradient pivot colors
        legend_elements = [
            mpatches.Patch(color='#FF0000', label='Red (sentiment -1.0)'),
            mpatches.Patch(color='#FFA500', label='Orange (sentiment -0.5)'),
            mpatches.Patch(color='#0000FF', label='Blue (sentiment 0.0)'),
            mpatches.Patch(color='#00FF00', label='Green (sentiment 0.5+)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
        
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        # Save PNG
        output_png = OUTPUT_DIR / f"sentiment_vs_{metric_col.lower()}.png"
        plt.savefig(output_png, dpi=200, bbox_inches='tight')
        print(f"Saved {metric_name} PNG to {output_png}")
        
        # Save SVG
        output_svg = OUTPUT_DIR / f"sentiment_vs_{metric_col.lower()}.svg"
        plt.savefig(output_svg, format='svg', bbox_inches='tight')
        print(f"Saved {metric_name} SVG to {output_svg}")
        
        plt.close()
    
    # Create date visualization
    if "createTimeISO" in data.columns:
        plot_data = data.dropna(subset=["createTimeISO", "average_sentiment"]).copy()
        plot_data = plot_data.sort_values("createTimeISO")
        
        if not plot_data.empty:
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.set_title("Average Comment Sentiment Over Time", fontsize=16, pad=20)
            ax.set_xlabel("Date of Publishing", fontsize=12)
            ax.set_ylabel("Average Comment Sentiment", fontsize=12)
            
            # Set axis limits first
            date_min = plot_data["createTimeISO"].min()
            date_max = plot_data["createTimeISO"].max()
            y_min = plot_data["average_sentiment"].min() - 0.1
            y_max = plot_data["average_sentiment"].max() + 0.1
            
            ax.set_xlim(date_min, date_max)
            ax.set_ylim(y_min, y_max)
            
            # Plot each point with colored circle only
            for idx, row in plot_data.iterrows():
                date = row["createTimeISO"]
                sentiment = row["average_sentiment"]
                color = row["sentiment_color"]
                size = size_normalized.loc[idx] if idx in size_normalized.index else 0.5
                
                # Circle size based on sentiment magnitude
                marker_size = (base_font * size * 2) ** 2
                
                # Draw colored circle
                ax.scatter(
                    date,
                    sentiment,
                    s=marker_size,
                    c=color,
                    alpha=0.6,
                    zorder=1,
                    edgecolors='black',
                    linewidths=0.5
                )
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            
            # Add legend
            legend_elements = [
                mpatches.Patch(color='#FF0000', label='Angry (-1 to -0.5)'),
                mpatches.Patch(color='#FFA500', label='Unhappy (-0.5 to 0)'),
                mpatches.Patch(color='#00FFFF', label='Neutral (0 to 0.5)'),
                mpatches.Patch(color='#00FF00', label='Happy (0.5 to 1)'),
            ]
            ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9)
            
            ax.grid(True, linestyle="--", alpha=0.3)
            plt.tight_layout()
            
            # Save PNG
            output_png = OUTPUT_DIR / "sentiment_vs_date.png"
            plt.savefig(output_png, dpi=200, bbox_inches='tight')
            print(f"Saved date visualization PNG to {output_png}")
            
            # Save SVG
            output_svg = OUTPUT_DIR / "sentiment_vs_date.svg"
            plt.savefig(output_svg, format='svg', bbox_inches='tight')
            print(f"Saved date visualization SVG to {output_svg}")
            
            plt.close()
    
    print(f"\nSentiment analysis complete!")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

