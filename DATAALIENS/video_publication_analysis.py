#!/usr/bin/env python3
"""Analyze video publication dates and identify spikes."""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
VIDEOS_PATH = ROOT / "dataset_tiktok-hashtag-scraper_2025-11-11_13-24-55-129.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "video_publications_by_month.png"
OUTPUT_SVG = OUTPUT_DIR / "video_publications_by_month.svg"
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    """Analyze publication dates and create bar chart."""
    print("Reading video data...")
    df = pd.read_csv(VIDEOS_PATH, encoding="utf-8-sig")
    
    # Parse publication dates
    df["createTimeISO"] = pd.to_datetime(df["createTimeISO"], errors="coerce")
    df = df.dropna(subset=["createTimeISO"])
    
    print(f"Total videos: {len(df)}")
    print(f"Date range: {df['createTimeISO'].min()} to {df['createTimeISO'].max()}")
    
    # Create year-month column for grouping
    df["year_month"] = df["createTimeISO"].dt.to_period("M")
    df["year_month_str"] = df["year_month"].astype(str)
    
    # Count videos per month
    monthly_counts = df.groupby("year_month").size().reset_index(name="count")
    monthly_counts["year_month_str"] = monthly_counts["year_month"].astype(str)
    
    # Convert period to datetime for plotting
    monthly_counts["date"] = monthly_counts["year_month"].dt.to_timestamp()
    
    # Sort by date
    monthly_counts = monthly_counts.sort_values("date")
    
    # Find spikes (months with significantly higher counts)
    mean_count = monthly_counts["count"].mean()
    std_count = monthly_counts["count"].std()
    threshold = mean_count + (1.5 * std_count)  # 1.5 standard deviations above mean
    
    spikes = monthly_counts[monthly_counts["count"] >= threshold].copy()
    
    # Print analysis results
    print("\n" + "="*60)
    print("PUBLICATION SPIKE ANALYSIS")
    print("="*60)
    print(f"\nAverage videos per month: {mean_count:.1f}")
    print(f"Standard deviation: {std_count:.1f}")
    print(f"Spike threshold (mean + 1.5*std): {threshold:.1f}")
    print(f"\nTotal months with data: {len(monthly_counts)}")
    print(f"Months identified as spikes: {len(spikes)}")
    
    if len(spikes) > 0:
        print("\n📈 PUBLICATION SPIKES:")
        print("-" * 60)
        for _, row in spikes.iterrows():
            print(f"  {row['year_month_str']}: {row['count']} videos")
        
        # Top 3 spikes
        top_spikes = spikes.nlargest(3, "count")
        print("\n🏆 TOP 3 PUBLICATION SPIKES:")
        print("-" * 60)
        for i, (_, row) in enumerate(top_spikes.iterrows(), 1):
            print(f"  {i}. {row['year_month_str']}: {row['count']} videos")
    else:
        print("\nNo significant spikes detected above threshold.")
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create bars
    bars = ax.bar(monthly_counts["date"], monthly_counts["count"], 
                  width=20,  # Width in days
                  color='steelblue', 
                  alpha=0.7,
                  edgecolor='navy',
                  linewidth=1)
    
    # Highlight spikes
    if len(spikes) > 0:
        spike_dates = spikes["date"].values
        spike_counts = spikes["count"].values
        ax.bar(spike_dates, spike_counts,
               width=20,
               color='red',
               alpha=0.8,
               edgecolor='darkred',
               linewidth=2,
               label='Spikes')
    
    # Add threshold line
    ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2, 
               label=f'Spike Threshold ({threshold:.1f})', alpha=0.7)
    
    # Formatting
    ax.set_title("Video Publications by Month", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Month", fontsize=14)
    ax.set_ylabel("Number of Videos Published", fontsize=14)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11)
    
    # Add value labels on bars
    for i, (date, count) in enumerate(zip(monthly_counts["date"], monthly_counts["count"])):
        if count >= threshold or count == monthly_counts["count"].max():
            ax.text(date, count + max(monthly_counts["count"]) * 0.01, 
                   str(int(count)),
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plots
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    print(f"\n✅ Saved bar chart PNG to {OUTPUT_PNG}")
    
    plt.savefig(OUTPUT_SVG, format='svg', bbox_inches='tight')
    print(f"✅ Saved bar chart SVG to {OUTPUT_SVG}")
    
    plt.close()
    
    # Additional statistics
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    print(f"Month with most videos: {monthly_counts.loc[monthly_counts['count'].idxmax(), 'year_month_str']} ({monthly_counts['count'].max()} videos)")
    print(f"Month with least videos: {monthly_counts.loc[monthly_counts['count'].idxmin(), 'year_month_str']} ({monthly_counts['count'].min()} videos)")
    print(f"Median videos per month: {monthly_counts['count'].median():.1f}")
    
    # Yearly summary
    df["year"] = df["createTimeISO"].dt.year
    yearly_counts = df.groupby("year").size()
    print("\n📊 Videos by Year:")
    for year, count in yearly_counts.items():
        print(f"  {year}: {count} videos")


if __name__ == "__main__":
    main()

