import pickle
import pandas as pd
from pathlib import Path

example_id = 41

project_root = Path(__file__).resolve().parent
pickle_path = project_root / "Data" / f"Example_{example_id}" / "dataset" / "Dataset.pickle"
csv_path = project_root / "Data" / f"Example_{example_id}" / "dataset" / "rl_dataset_readable.csv"

with open(pickle_path, "rb") as f:
    obj = pickle.load(f)

if hasattr(obj, "get_Data"):
    data = obj.get_Data()
elif isinstance(obj, dict) and "List_Representant_Classes" in obj:
    data = obj["List_Representant_Classes"]
else:
    raise TypeError("Unsupported dataset format")

rows = []

for class_id, class_entry in enumerate(data):
    class_probs = class_entry[0]
    pilot_samples = class_entry[1]

    for sample_id, (p, pilot_vec) in enumerate(zip(class_probs, pilot_samples)):
        row = {
            "class_id": class_id,
            "sample_id": sample_id,
            "class_probability": float(p),
        }

        for j, val in enumerate(pilot_vec):
            row[f"pilot_{j}"] = float(val)

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)

print(f"Saved CSV to: {csv_path}")
print(df.head())