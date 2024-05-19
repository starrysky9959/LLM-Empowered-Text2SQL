# %%
# python3
import os
from openai import OpenAI
from constants import DATABASE_PATH_PATTERN, TRAIN_JSON_PATH
import json
import util
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

# %%
import time

TABLE_PATTERN = """
[Table Name]:{table_name}
```csv
{sample_data}
```
""".lstrip()

INSTRUCTION_PATTERN = """
Given SQLite tables and their sample data in the following code blocks:
{table_info}
Here is a sample question-SQL pair.
```json
{{
    "question": "{seed_quesion}",
    "sql": "{seed_sql}"
}}
```

Please gain inspiration from the above tables and sample question-SQL pair to create {result_size} new high-quality question-SQL pairs.
Gradually increase the complexity of the generated results to enrich the diversity.
Present your all output in JSON format like the given sample.
""".strip()


def chat_deepseek(instruction: str):
    conversation = [
        {
            "role": "system",
            "content": "You are exceptionally skilled at crafting high-quality question-SQL pair.",
        },
    ]
    conversation.append({"role": "user", "content": instruction})

    # print("conversation: ", conversation)
    response = client.chat.completions.create(
        model="deepseek-coder", messages=conversation
    )

    return response.choices[0].message.content


pattern = re.compile(r"```json\n(.+)\n```", re.S)

generated_datasets = []
with open(TRAIN_JSON_PATH, "r") as dataset_file:
    dataset = json.load(dataset_file)
    i = 7911
    for case in dataset[i:]:
        table_info = ""
        db_path = DATABASE_PATH_PATTERN.format(db_id=case["db_id"])
        # print(db_path)
        table_names = util.get_sqlite_schema_dict(db_path)[0]
        for table_name in table_names:
            # print(table_name)
            table_info += TABLE_PATTERN.format(
                table_name=table_name,
                sample_data=util.get_sqlite_sample_csv(db_path, table_name),
            )

        instruction = INSTRUCTION_PATTERN.format(
            table_info=table_info,
            seed_quesion=case["question"],
            seed_sql=case["query"],
            result_size="10",
        )
        # print("instruction", instruction)
        output = ""
        while True:
            try:
                output = chat_deepseek(instruction)
                print("intruction: ", instruction)
                print("output: ", output)
                break
            except Exception as e:
                print(e)
                print("i=", i)
                time.sleep(1)

        # try:
        #     for json_str in pattern.findall(output):
        #         question_sql_pairs_json = json.loads(json_str)
        #         if isinstance(question_sql_pairs_json, list):
        #             print("append list")
        #             for pair in question_sql_pairs_json:
        #                 pair["db_id"] = case["db_id"]
        #             generated_datasets.extend(question_sql_pairs_json)
        #         elif isinstance(question_sql_pairs_json, dict):
        #             print("append dict")
        #             question_sql_pairs_json["db_id"] = case["db_id"]
        #             generated_datasets.append(question_sql_pairs_json)
        #         else:
        #             print("type: ", type(question_sql_pairs_json))

        # except Exception as e:
        #     print(e)

        time.sleep(0.5)
        i += 1
        print("i:", i)
        with open(f"oss_insight/new_dataset_{i}.json", "w") as output_file:
            output_file.write(output)
