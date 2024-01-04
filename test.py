import json

import util
from constants import DEV_JSON_PATH, DATABASE_PATH_PATTERN

with open(DEV_JSON_PATH, "r") as dev_json_file:
    dataset = json.load(dev_json_file)
    for test_case in dataset[:1]:
        db_id = test_case["db_id"]
        db_path = DATABASE_PATH_PATTERN.format(db_id=db_id)
        d = util.get_sqlite_schema_dict(db_path)
        print(d)
