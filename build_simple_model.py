import csv
import json
import numpy as np

DATA_CSV = "isl_dataset.csv"
MODEL_JSON = "isl_simple_model.json"

def main():
    data = {}
    with open(DATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"]
            vec = []
            for i in range(21):
                x = float(row[f"x_{i}"])
                y = float(row[f"y_{i}"])
                z = float(row[f"z_{i}"])
                vec.extend([x, y, z])
            vec = np.array(vec)
            data.setdefault(label, []).append(vec)

    means = {}
    for label, vectors in data.items():
        stack = np.vstack(vectors)
        means[label] = stack.mean(axis=0).tolist()
        print(f"Label {label}: {len(vectors)} samples, mean vector length {len(means[label])}")

    with open(MODEL_JSON, "w") as f:
        json.dump(means, f)

    print(f"Saved simple model to {MODEL_JSON}")

if __name__ == "__main__":
    main()