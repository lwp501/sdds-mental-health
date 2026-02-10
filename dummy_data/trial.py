import pandas as pd
import os

from bs4 import BeautifulSoup

def convert_yt_history_to_csv(input_html_path, output_csv_path="youtube_history_cleaned.csv"):
    """
    Parses a YouTube watch-history.html file and exports a cleaned CSV.
    
    :param input_html_path: Path to the .html file from Google Takeout
    :param output_csv_path: Path where the resulting CSV will be saved
    """
    if not os.path.exists(input_html_path):
        print(f"Error: The file '{input_html_path}' was not found.")
        return

    print(f"Reading {input_html_path}...")
    
    with open(input_html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Target the specific container for watch entries
    cells = soup.find_all('div', class_='content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1')
    
    extracted_data = []

    for cell in cells:
        links = cell.find_all('a')
        
        # A standard entry has at least the video link
        if not links:
            continue
            
        # 1. Video Title & URL
        # We strip "Watched " if it's prepended to the link text
        video_title = links[0].text.replace("Watched ", "", 1)
        video_url = links[0].get('href')

        # 2. Channel Name & URL
        if len(links) > 1:
            channel_name = links[1].text
            channel_url = links[1].get('href')
        else:
            channel_name = "Unknown (Deleted or Private)"
            channel_url = None

        # 3. Timestamp
        # In this HTML, the timestamp is usually the last text node in the cell
        br_tags = cell.find_all('br')
        timestamp = br_tags[-1].next_sibling.strip() if br_tags else None

        extracted_data.append({
            "Video Title": video_title,
            "Channel": channel_name,
            "Timestamp": timestamp,
            "Video URL": video_url,
            "Channel URL": channel_url
        })

    # Convert to DataFrame
    df = pd.DataFrame(extracted_data)
    
    # Clean up the timestamp column to actual Python datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    
    # Sort by newest first
    df = df.sort_values(by="Timestamp", ascending=False)

    df.to_csv(output_csv_path, index=False)
    print(f"Done! Created '{output_csv_path}' with {len(df)} entries.")

# --- EXAMPLE USAGE ---
# You can now call this function with any path you like
my_file = "C:/Users/lwp501/Google Drive/research/STREAM/dummy_data/Takeout/YouTube and YouTube Music/history/watch-history.html"
convert_yt_history_to_csv(my_file, "my_youtube_data.csv")

