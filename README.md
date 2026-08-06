# sdds-mental-health

A repo of code to process donated digital trace data, as collected by the Smart Data Donation Service (SDDS) MOSAIC study 

**Status: Work in Progress**  

This project is currently under active development. Tests, sample inputs, and output pipelines are structured around **synthetic or sample data** for demonstration and development purposes.

## Files

* `.python-version`: Specifies the Python version for environment setup
* `requirements.txt`: python package dependencies to install
* `yt_metatadata.py`: Extracts metadata from YouTube video URLs in `.txt` format using `yt-dlp` (no API key required). Outputs structured records into a `.jsonl` file
* `prepare_metadata.py`: Take scraped youtube metadata and selects only those fields to be used for a specific task
* `zero_shot_classification.py`: Run basic zero shot classification model

## TO COME




