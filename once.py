import csv
import json
import numpy as np

CSV_PATH = "isl_dataset.csv"          # correct name
JSON_PATH = "isl_simple_model.json"   # used by realtime_predict_simple.py

def build_means_from_csv(csv_path):
    data = {}  # label -> list of vectors

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)

        # --- skip header row ---
        header = next(reader, None)
        # header should contain things like: label,x_0,y_0,z_0,...
        # ------------------------

        for row in reader:
            if not row:
                continue

            label = row[0].strip()

            try:
                values = np.array([float(x) for x in row[1:]], dtype=np.float32)
            except ValueError as e:
                print("Skipping row for label", label, "due to parse error:", e)
                continue

            if label not in data:
                data[label] = []
            data[label].append(values)

    means = {}
    for label, vectors in data.items():
        stacked = np.stack(vectors, axis=0)   # (N, D)
        mean_vec = stacked.mean(axis=0)       # (D,)
        means[label] = mean_vec.tolist()
        print(f"Label {label}: {stacked.shape[0]} samples, dim={stacked.shape[1]}")

    return means

if __name__ == "__main__":
    means = build_means_from_csv(CSV_PATH)
    with open(JSON_PATH, "w") as f:
        json.dump(means, f)
    print("Saved means to", JSON_PATH)