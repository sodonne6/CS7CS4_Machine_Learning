import json
import string
from pathlib import Path

# ---- 1. Define the target character set ----

BASIC_PUNCTUATION = ".,;:'\"!?-()"
TARGET_CHARS = set(string.ascii_lowercase + string.ascii_uppercase + BASIC_PUNCTUATION)

# ---- 2. List your datasets here ----
# Update these paths if your filenames/locations differ.
DATASETS = {
    "child_train": "C:\\Users\\irish\\Computer_Electronic_Engineering_Year5\\Machine_Learning\\Week_9\\week9Assignment\\input_childSpeech_trainingSet.txt",
    "child_test": "C:\\Users\\irish\\Computer_Electronic_Engineering_Year5\\Machine_Learning\\Week_9\\week9Assignment\\input_childSpeech_testSet.txt",
    "shakespeare": "C:\\Users\\irish\\Computer_Electronic_Engineering_Year5\\Machine_Learning\\Week_9\\week9Assignment\\input_shakespeare.txt",
}

# ---- 3. Output JSON path ----
OUTPUT_JSON = "char_coverage_report.json"

def analyse_dataset(path: str, label: str):
    """Return JSON-ready info about which chars are present/missing in this dataset."""
    text = Path(path).read_text(encoding="utf-8")

    present_chars = {ch for ch in text if ch in TARGET_CHARS}
    missing_chars = TARGET_CHARS - present_chars

    # Optional: characters that are NOT in the target set (excluding whitespace)
    extra_chars = {ch for ch in text if ch not in TARGET_CHARS and not ch.isspace()}

    return {
        "dataset": label,
        "present": sorted(present_chars),
        "missing": sorted(missing_chars),
        "extra": sorted(extra_chars),   # remove this line if you don't want it
    }

def main():
    results = {}

    for label, path in DATASETS.items():
        results[label] = analyse_dataset(path, label)

    # Save as pretty JSON to file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Optional: also print a small confirmation
    print(f"Character coverage report saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
