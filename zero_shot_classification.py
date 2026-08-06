import json
import argparse
from pathlib import Path
from transformers import pipeline


def classify_dataset_zero_shot(input_file, output_file, model, revision):
    """
    Use a huggingface zero shot classification model. 
    Defaults to main facebook/bart-large-mnli
    Outputs a json file
    """

    input_path = Path(input_file)
    output_path = Path(output_file)

    #load default huggingface model
    classifier = pipeline("zero-shot-classification",
                          model = model,
                          revision = revision)

    # Print the loaded model
    model_name = classifier.model.config._name_or_path
    print(f"Using Hugging Face model: {model_name}")

    candidate_labels = ["soccer", "non-soccer"] ###simple example based on example_urls.txt
    hypothesis_template = "This video is about {}."
    processed_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            video_id = record.get("video_id") 

            # Combine values into single string, exclude video_id
            text_parts = [
                str(val) for key, val in record.items() 
                if key not in ("video_id", "video-id") and val is not None and str(val).strip()
            ]
            text_to_classify = " | ".join(text_parts)

            if not text_to_classify:
                output_record = {
                    "video_id": video_id,
                    "predicted_label": "unknown",
                    "confidence_score": 0.0,
                    "classification_text": "none"
                }

            else:
                result = classifier(
                    text_to_classify,
                    candidate_labels,
                    hypothesis_template=hypothesis_template
                )

                output_record = {
                    "video_id": video_id,
                    "predicted_label": result["labels"][0],
                    "confidence_score": result["scores"][0],
                    "classification_text" : text_to_classify
                }
                

            f_out.write(json.dumps(output_record) + "\n")
            processed_count += 1
            if processed_count % 5 == 0:
                print(f"Classified {processed_count} records...")

    print(f"\nDone! Classified {processed_count} records and saved to '{output_file}'.")

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify JSONL text data using a Hugging Face zero-shot classification."
    )
    parser.add_argument("-i", "--input", default = "data_for_classification.jsonl", help="Path to input .jsonl file")
    parser.add_argument("-o", "--output", default = "classified_output_ZS.jsonl", help="Path to output .jsonl file")
    parser.add_argument("-m", "--model", default = "facebook/bart-large-mnli", help="Optional Hugging Face model identifier")
    parser.add_argument("-r", "--revision", default = "main", help="Optional - model revision to use (default:main)")

    args = parser.parse_args()

    classify_dataset_zero_shot(args.input, args.output, args.model, args.revision)






