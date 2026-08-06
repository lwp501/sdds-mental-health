import json 
import argparse
from pathlib import Path


def prepare_metadata(input_jsonl, output_jsonl, fields):
    """Take a scraped json file, 
    keep only those fields specified to be used in downstream classification
    Save a jsonl"""

    input_path =  Path(input_jsonl)
    output_path = Path(output_jsonl)

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:     #open 2 json files 
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line) # convert json line into a dictionary
            video_id = record.get("video_id")

            filtered = {
                "video_id": video_id,
                **{field: record.get(field) for field in fields} # extract requested field from record'
            }
            f_out.write(json.dumps(filtered) + "\n") 

    print("Done")

def main():
    parser = argparse.ArgumentParser(
        description="Filter scraped YouTube metadata into datasets for classification."
    )
    parser.add_argument("-i", "--input", default="yt_metadata_output.jsonl", help="Path to raw JSONL metadata.")
    parser.add_argument("-o", "--output", default="data_for_classification.jsonl", help="Path to filtered output JSONL.")
    parser.add_argument("-f", "--fields", nargs="+", default=["title"],
                        help="Space-separated list of fields to extract (e.g. -f title description)."
    )

    args = parser.parse_args()

    prepare_metadata(args.input, args.output, args.fields)

if __name__ == "__main__":
    main()