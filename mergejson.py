import json
import os

from constants import TRAIN_JSON_PATH

merged_data = []

with open(TRAIN_JSON_PATH, "r") as dataset_file:
    dataset = json.load(dataset_file)
for i in range(1, 8659+1):
    file_path=f"extract/{i}.json"
    if os.path.exists(file_path):
        db_id = dataset[i-1]["db_id"]
        with open(file_path, "r") as f:
            data = json.load(f)
            for case in data:
                case["db_id"] = db_id
            merged_data.extend(data)
print("len: ", len(merged_data))
# Write the merged data to a new JSON file
with open("oss.json", "w") as f:
    json.dump(merged_data, f)
