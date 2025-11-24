#!/usr/bin/env python3
"""Analyze user engagement metrics by month."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
VIDEOS_PATH = ROOT / "dataset_tiktok-hashtag-scraper_2025-11-11_13-24-55-129.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "video_engagement_by_month.png"
OUTPUT_SVG = OUTPUT_DIR / "video_engagement_by_month.svg"
OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_for_opacity(value, min_val, max_val, min_opacity=0.3, max_opacity=1.0):
    """Normalize value to opacity range."""
    if max_val == min_val:
        return (min_opacity + max_opacity) / 2
    normalized = (value - min_val) / (max_val - min_val)
    return min_opacity + normalized * (max_opacity - min_opacity)


def main() -> None:
    """Analyze engagement metrics and create visualization."""
    print("Reading video data...")
    df = pd.read_csv(VIDEOS_PATH, encoding="utf-8-sig")
    
    # Parse dates and convert numeric columns
    df["createTimeISO"] = pd.to_datetime(df["createTimeISO"], errors="coerce")
    for col in ["playCount", "diggCount", "shareCount", "commentCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["createTimeISO", "playCount"])
    
    print(f"Total videos with engagement data: {len(df)}")
    
    # Create year-month column
    df["year_month"] = df["createTimeISO"].dt.to_period("M")
    df["year_month_str"] = df["year_month"].astype(str)
    
    # Aggregate monthly metrics
    monthly_stats = df.groupby("year_month").agg({
        "playCount": "sum",
        "diggCount": "sum",
        "shareCount": "sum",
        "commentCount": "sum",
        "createTimeISO": "count"  # Count of videos
    }).reset_index()
    
    monthly_stats.columns = ["year_month", "total_views", "total_likes", 
                            "total_shares", "total_comments", "video_count"]
    
    # Calculate engagement metrics
    monthly_stats["likes_per_view"] = monthly_stats["total_likes"] / monthly_stats["total_views"]
    monthly_stats["engagement_rate"] = (monthly_stats["total_likes"] + monthly_stats["total_comments"]) / monthly_stats["total_views"]
    monthly_stats["year_month_str"] = monthly_stats["year_month"].astype(str)
    monthly_stats["date"] = monthly_stats["year_month"].dt.to_timestamp()
    
    # Sort by date
    monthly_stats = monthly_stats.sort_values("date")
    
    # Normalize bar widths (based on number of videos published that month)
    # Wider bars = more videos published
    min_videos = monthly_stats["video_count"].min()
    max_videos = monthly_stats["video_count"].max()
    # Width will be proportional to number of videos, scaled to reasonable bar widths
    base_width = 15  # Base width in days
    if max_videos > min_videos:
        monthly_stats["bar_width"] = base_width + (
            (monthly_stats["video_count"] - min_videos) / (max_videos - min_videos) * 15
        )
    else:
        monthly_stats["bar_width"] = base_width
    
    # Print summary statistics
    print("\n" + "="*60)
    print("ENGAGEMENT ANALYSIS BY MONTH")
    print("="*60)
    print(f"\nTotal views across all months: {monthly_stats['total_views'].sum():,.0f}")
    print(f"Total likes across all months: {monthly_stats['total_likes'].sum():,.0f}")
    print(f"Average likes per view: {monthly_stats['likes_per_view'].mean():.4f}")
    print(f"Average engagement rate: {monthly_stats['engagement_rate'].mean():.4f}")
    
    # Top months by views
    top_views = monthly_stats.nlargest(3, "total_views")
    print("\n🏆 TOP 3 MONTHS BY TOTAL VIEWS:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_views.iterrows(), 1):
        print(f"  {i}. {row['year_month_str']}: {row['total_views']:,.0f} views, "
              f"{row['likes_per_view']:.4f} likes/view")
    
    # Top months by engagement rate
    top_engagement = monthly_stats.nlargest(3, "engagement_rate")
    print("\n⭐ TOP 3 MONTHS BY ENGAGEMENT RATE:")
    print("-" * 60)
    for i, (_, row) in enumerate(top_engagement.iterrows(), 1):
        print(f"  {i}. {row['year_month_str']}: {row['engagement_rate']:.4f} "
              f"({row['total_views']:,.0f} views)")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Create color gradient based on engagement rate
    # Use a colormap from blue (low) to red (high) engagement
    engagement_min = monthly_stats["engagement_rate"].min()
    engagement_max = monthly_stats["engagement_rate"].max()
    
    # Normalize engagement rates for colormap
    normalized_engagement = (monthly_stats["engagement_rate"] - engagement_min) / (
        engagement_max - engagement_min
    ) if engagement_max > engagement_min else np.zeros(len(monthly_stats))
    
    # Create colormap (blue to green to yellow to red)
    colors = plt.cm.RdYlGn(normalized_engagement)
    
    # Create bars with varying width (no opacity variation)
    bars = []
    for i, (_, row) in enumerate(monthly_stats.iterrows()):
        bar = ax.bar(row["date"], row["total_views"],
                    width=row["bar_width"],
                    color=colors[i],
                    alpha=1.0,  # Full opacity
                    edgecolor='black',
                    linewidth=1.5,
                    label=row["year_month_str"] if i < 5 else "")
        bars.append(bar)
    
    # Add value labels on top of bars
    for _, row in monthly_stats.iterrows():
        # Format large numbers
        views = row["total_views"]
        if views >= 1_000_000:
            views_str = f"{views/1_000_000:.1f}M"
        elif views >= 1_000:
            views_str = f"{views/1_000:.1f}K"
        else:
            views_str = f"{views:.0f}"
        
        ax.text(row["date"], row["total_views"] + max(monthly_stats["total_views"]) * 0.02,
               views_str,
               ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_title("Monthly Video Engagement: Total Views\n"
                "(Bar Height = Total Views, Width = Number of Videos Published)",
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Total Views", fontsize=14)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add colorbar for engagement rate
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                              norm=plt.Normalize(vmin=engagement_min, vmax=engagement_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Engagement Rate (Likes + Comments / Views)', 
                  rotation=270, labelpad=20, fontsize=11)
    
    # Add text annotation explaining the visualization
    textstr = f'Bar Width: Number of videos published (wider = more videos)\nColor: Engagement rate (green = high, red = low)'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved engagement chart PNG to {OUTPUT_PNG}")
    
    plt.savefig(OUTPUT_SVG, format='svg', bbox_inches='tight')
    print(f"✅ Saved engagement chart SVG to {OUTPUT_SVG}")
    
    plt.close()
    
    print("\n" + "="*60)
    print("VISUALIZATION DETAILS")
    print("="*60)
    print("• Bar Height: Total views per month")
    print("• Bar Width: Number of videos published (wider = more videos)")
    print("• Bar Color: Engagement rate gradient (green = high, red = low)")


if __name__ == "__main__":
    main()

