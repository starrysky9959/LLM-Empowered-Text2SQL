# %%
import prompt
import chromadb

# client = chromadb.PersistentClient(path="vector_store.chromadb")
# print(client.heartbeat())
# collection_question = client.get_or_create_collection(name="spider_train_question")
# collection_query = client.get_or_create_collection(name="spider_train_query")


import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from constants import DATABASE_PATH_PATTERN, DEV_JSON_PATH
import util

print(torch.version.cuda)
model_path = "../LLaMA-Factory/0321_merge_step_1"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
print(model.device)
stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"


# %%
import re


def text2sql(instruction: str, few_shots: list):
    conversation = [
        {
            "role": "system",
            "content": prompt.SYSTEM_PROMPT,
        },
    ]
    # for shot in few_shots:
    #     db_path = DATABASE_PATH_PATTERN.format(db_id=shot["db_id"])

    #     conversation.extend(
    #         [
    #             {
    #                 "role": "user",
    #                 "content": INSTRUCTION_PATTERN.format(
    #                     schema=util.get_sqlite_schema_str(db_path),
    #                     question=shot["question"],
    #                 ),
    #             },
    #             {
    #                 "role": "assistant",
    #                 "content": RESPONSE_PATTERN.format(query=shot["query"]),
    #             },
    #         ]
    #     )
    conversation.append(
        {
            "role": "user",
            "content": instruction,
        },
    )
    #     prompt_with_shots = """
    # You are a database expert and please help me to write SQL query to answer a question.
    # Here are {n} examples of question and corresponding SQL query provided for your reference.
    # {examples}

    # And given the following SQLite database schema:
    # {schema}

    # Based on the above examples and schema, please return SQL query only to answer this question: {question}
    # """.strip().format(
    #         n=len(examples),
    #         examples="\n\n".join(examples),
    #         schema=schema,
    #         question=question,
    #     )

    # print(prompt_with_shots)

    inputs = tokenizer.apply_chat_template(
        conversation, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    print("conversation: ", conversation)
    outputs = model.generate(
        inputs,
        max_new_tokens=1024,
        do_sample=False,
        # top_p=0.95,
        # temperature=temperature,
        pad_token_id=stop_id,
        eos_token_id=stop_id,
    )

    # len(inputs[0])
    output = tokenizer.decode(outputs[0][len(inputs[0]) :], skip_special_tokens=True)
    print("output:", output)
    return output


step1_output = []
step2_output = []
with open("./cot_dataset/test/step_1.json", "r") as input_file:

    dataset = json.load(input_file)
for case in dataset[:]:
    # db_path = DATABASE_PATH_PATTERN.format(db_id=case["db_id"])
    # prompt_1 = prompt.step_1_prompt(db_path, case["question"])
    instruction = case["instruction"]
    # instruction_2, output_2 = prompt.step_2_prompt(
    #     db_path, case["question"], case["query"]
    # )
    answer = text2sql(
        instruction,
        [],
    )

    answer = answer.strip()
    step1_output.append(answer)
    # step2_output.append(answer)

with open("0325_step1_test.json", "w") as output:
    json.dump(step1_output, output)
