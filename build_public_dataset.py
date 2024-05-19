import json
import os


def build(db_path_format: str, dataset_path: str, type: str, output_path):
    with open(dataset_path, "r") as input_file:
        dataset_set = json.load(input_file)
    output = []
    for case in dataset_set:
        db_path = os.path.abspath(db_path_format.format(db_name=case["db_id"]))
        if type == "spider":
            sql = case["query"]
            evidence = ""
        else:
            sql = case["SQL"]
            evidence = case["evidence"]

        output.append(
            {
                "question": case["question"],
                "sql": sql,
                "evidence": evidence,
                "database": {"type": "sqlite", "url": db_path},
            }
        )
    with open(output_path, "w") as output_file:
        json.dump(output, output_file)


dataset_metadata = [
    {
        "db_path_format": "../spider/database/{db_name}/{db_name}.sqlite",
        "dataset_path": "../spider/train_spider_all.json",
        "type": "spider",
        "output_path": "public_dataset/raw/spider_train.json",
    },
    {
        "db_path_format": "../spider/database/{db_name}/{db_name}.sqlite",
        "dataset_path": "../spider/dev.json",
        "type": "spider",
        "output_path": "public_dataset/raw/spider_dev.json",
    },
    {
        "db_path_format": "../spider/test_database/{db_name}/{db_name}.sqlite",
        "dataset_path": "../spider/test_data/dev.json",
        "type": "spider",
        "output_path": "public_dataset/raw/spider_test.json",
    },
    {
        "db_path_format": "../bird_bench/train/train_databases/{db_name}/{db_name}.sqlite",
        "dataset_path": "../bird_bench/train/train.json",
        "type": "bird",
        "output_path": "public_dataset/raw/bird_train.json",
    },
    {
        "db_path_format": "../bird_bench/dev/dev_databases/{db_name}/{db_name}.sqlite",
        "dataset_path": "../bird_bench/dev/dev.json",
        "type": "bird",
        "output_path": "public_dataset/raw/bird_dev.json",
    },
]

for metadata in dataset_metadata:
    build(**metadata)
