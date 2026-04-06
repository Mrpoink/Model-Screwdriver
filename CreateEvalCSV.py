import json
import csv
import pathlib

def aggregate_evals(root_dir, output_file="eval_summary.csv"):
    results = []
    all_keys = set()
    
    path = pathlib.Path(root_dir)
    # Recursively find all json files in your Model-Screwdriver directory
    files = list(path.rglob("*.json"))
    
    # Sort by folder directory then filename as requested
    files.sort(key=lambda p: (p.parent, p.name))
    
    for file_path in files:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                
                # Unpack configuration and metrics
                row = {
                    "directory": str(file_path.parent),
                    "filename": file_path.name,
                    "timestamp": data.get("timestamp"),
                    **data.get("configuration", {}),
                    **data.get("metrics", {})
                }
                results.append(row)
                # Track every unique key found across all files
                all_keys.update(row.keys())
            except (json.JSONDecodeError, KeyError):
                continue

    if not results:
        print("No valid evaluation files found.")
        return

    # Sort keys so 'directory', 'filename', and 'timestamp' come first for readability
    fixed_headers = ["directory", "filename", "timestamp"]
    remaining_headers = sorted(list(all_keys - set(fixed_headers)))
    headers = fixed_headers + remaining_headers

    with open(output_file, 'w', newline='') as f:
        # extrasaction='ignore' prevents crashes if a row somehow has a mystery key
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Successfully processed {len(results)} files into {output_file}")

if __name__ == "__main__":
    # Pointed to your logs directory based on your traceback
    aggregate_evals(root_dir="./logs")