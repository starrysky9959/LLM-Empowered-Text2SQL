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


INSTRUCTION_PATTERN = """
Given SQLite database schema in the following code block:
```sql
{schema}
```

Question: {question}
The correct SQL query to answer this question is showed in the following code block:
```sql
{query}
```

Now please explain the above SQL query based on the given information.
""".strip()


def chat_deepseek(instruction: str):
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful database assistant chatbot designed to help users with their SQL queries.",
        },
    ]
    conversation.append({"role": "user", "content": instruction})

    # print("conversation: ", conversation)
    response = client.chat.completions.create(
        model="deepseek-coder", messages=conversation
    )

    return response.choices[0].message.content


generated_datasets = []
with open(TRAIN_JSON_PATH, "r") as dataset_file:
    dataset = json.load(dataset_file)
    i = 1000
    for case in dataset[1000:]:
        db_path = DATABASE_PATH_PATTERN.format(db_id=case["db_id"])
        instruction = INSTRUCTION_PATTERN.format(
            schema=util.get_sqlite_schema_str(db_path),
            question=case["question"],
            query=case["query"],
        )
        output = ""
        while True:
            try:
                output = chat_deepseek(instruction)
                # print("intruction: ", instruction)
                # print("output: ", output)
                break
            except Exception as e:
                print("i=", i)
                print(e)
        generated_datasets.append({"instruction": instruction, "output": output})
        time.sleep(0.5)
        i += 1
        print("i:", i)
        if i % 100 == 0:
            with open(f"dataset/new_dataset_{i}.json", "w") as output_file:
                json.dump(generated_datasets, output_file)
                generated_datasets = []


with open("dataset/new_dataset_end.json", "w") as output_file:
    json.dump(generated_datasets, output_file)
