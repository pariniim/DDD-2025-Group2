#!/usr/bin/env python3
"""Generate word cloud from TikTok comments."""

from pathlib import Path
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re

ROOT = Path("/Users/andrei/Documents/DATAALIENS")
COMMENTS_PATH = ROOT / "tiktok25ComentsPerVideo.csv"
OUTPUT_DIR = ROOT / "analysis_outputs"
OUTPUT_PNG = OUTPUT_DIR / "comment_text_wordcloud.png"
OUTPUT_DIR.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    """Clean and preprocess text for word cloud."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


def main() -> None:
    """Generate word cloud from comments."""
    print(f"Reading comments from {COMMENTS_PATH}")
    df = pd.read_csv(COMMENTS_PATH, encoding="utf-8-sig")
    
    # Combine all comment texts
    all_text = " ".join(df["text"].fillna("").astype(str).apply(clean_text))
    
    if not all_text.strip():
        raise SystemExit("No text found in comments.")
    
    # Generate word cloud
    print("Generating word cloud...")
    wordcloud = WordCloud(
        width=1200,
        height=800,
        background_color='white',
        max_words=200,
        colormap='viridis',
        relative_scaling=0.5,
        random_state=42
    ).generate(all_text)
    
    # Create plot
    plt.figure(figsize=(15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud from TikTok Comments', fontsize=20, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Word cloud saved to {OUTPUT_PNG}")


if __name__ == "__main__":
    main()


