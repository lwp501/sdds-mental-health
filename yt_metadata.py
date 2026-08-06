"""
yt_metadata.py

Extracts metadata from YouTube video URLs in .txt format using yt-dlp (no API key required)
Outputs structured records into a .jsonl file.

Author:
Lewis Paton, University of York
Code assist with Gemini 3.1 Pro    
"""


import argparse
import json
import os
import re
import sys
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi


def extract_video_id(url):
    """Extracts YouTube video IDs from various URL formats"""
    if not isinstance(url, str):
        return None
    
    # handle unicode-escaped equals signs, trim quotes/whitespace
    url = url.replace(r"\u003d", "=").replace("%3D", "=").replace("%3d", "=").strip().strip('"\'')
    if not url:
        return None

    # Regex pattern for standard YouTube video URLs (watch, embed, shorts, live, youtu.be)
    # [a-zA-Z0-9_-]{11} = data to extract, 11 characters
    pattern = r"(?:v=|\/embed\/|\/shorts\/|\/live\/|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None


def load_urls(file_path):
    """Loads URLs from a txt file, ignore empty lines"""
    if not os.path.exists(file_path):
        print(f"Error: Input file '{file_path}' not found.")
        sys.exit(1)
    
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_jsonl(data, output_path):
    """Saves metadata records into a .jsonl file"""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(data)} records to '{output_path}'.")


def deduplicate_videos(input_urls):
    """
    Filter out duplicate videos to reduce scraping needed """

    seen_ids = set() # track seen_ids

    unique_urls = []

    for url in input_urls:
        video_id = extract_video_id(url)
        
        if video_id in seen_ids:
            print(f"[SKIP] Duplicate video detected: {video_id}")
            continue
            
        seen_ids.add(video_id)
        unique_urls.append(url)
        
    return unique_urls

class YouTubeScraperFetcher:
    """Fetches YouTube video metadata using yt-dlp"""

    def __init__(self, fetch_transcript=False, *args, **kwargs):
        # Accepts extra arguments gracefully for interface compatibility
        self.fetch_transcript = fetch_transcript

    def fetch_single(self, raw_url_or_id):
        """Processes a single URL or video ID and returns a metadata dictionary."""
        video_id = extract_video_id(raw_url_or_id)
        if not video_id:
            return None

        clean_url = f"https://www.youtube.com/watch?v={video_id}"

        ydl_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': False,
            'no_warnings': True,
            'sleep_interval': 1,
            'max_sleep_interval': 5 # randomised sleep between 1 and 5 seconds 
        }


        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=False)

                vid_info = {
                    "video_id": video_id,
                    "url": clean_url,
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "published_at": info.get("upload_date"),
                    "channel_id": info.get("channel_id"),
                    "channel_title": info.get("uploader"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count")
                }
                if self.fetch_transcript:
                    try:
                        ytt_api = YouTubeTranscriptApi()
                        fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
                        # Concatenate text entries into a single string
                        vid_info["transcript"] = " ".join([item.text for item in fetched_transcript])
                    except Exception as sub_err:
                        print(f"Notice: No transcript available for video '{video_id}' ({sub_err})")
                        vid_info["transcript"] = None

                return vid_info
            
        except Exception as e:
            print(f"Error scraping video ID '{video_id}': {e}")
            return None

    def process_urls(self, urls):
        """Processes a list of URLs and returns valid metadata dictionaries."""
        results = []
        total = len(urls)
        for idx, raw_url in enumerate(urls, start=1):
            vid = extract_video_id(raw_url)
            print(f"[{idx}/{total}] Scraping video ID: {vid}...")
            data = self.fetch_single(raw_url)
            if data:
                results.append(data)
        return results


def main():
    parser = argparse.ArgumentParser(description="Fetch YouTube video metadata using yt-dlp.")
    parser.add_argument("-i", "--input", required=True, help="Path to input text file containing YouTube URLs.")
    parser.add_argument("-o", "--output", default="yt_metadata_output.jsonl", help="Output path for JSONL file.")
    parser.add_argument("-t", "--transcript", action="store_true", help="Fetch video transcripts with metadata")    

    args = parser.parse_args()

    # Load URLs/IDs
    urls = load_urls(args.input)
    print(f"Loaded {len(urls)} lines from '{args.input}'.")

    #deduplicate
    urls_to_scrape = deduplicate_videos(urls)
    print(f"Total of {len(urls_to_scrape)} ready to scrape")

    # Process and save results
    
    fetcher = YouTubeScraperFetcher(fetch_transcript=args.transcript)
    metadata = fetcher.process_urls(urls_to_scrape)
    save_jsonl(metadata, args.output)

if __name__ == "__main__":
    main()