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
from constants import DATABASE_PATH_PATTERN, DEV_JSON_PATH, TEST_JSON_PATH
import util

print(torch.version.cuda)
model_path = "../LLaMA-Factory/0322_merge_step_2"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, device_map="auto"
)
print(model.device)
stop_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
assert isinstance(stop_id, int), "Invalid tokenizer, EOT id not found"


# %%
import re


def text2sql(instruction: str):
    conversation = [
        {
            "role": "system",
            "content": prompt.SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": instruction,
        },
    ]

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


# def get_tables(out: str):
#     position = out.find("[Relevant Tables]\n")
#     sql = out[position + 18 :].strip()
#     return list(sql.split(","))


step1_output = []
step2_output = []

with open("spider_test_comment.json", "r") as input_file:
    comment_map = json.load(input_file)

with open(TEST_JSON_PATH, "r") as input_file:
    dev_set = json.load(input_file)
# with open("./0319_step_2.json", "r") as input_file:
#     step_1_dataset = json.load(input_file)

with open("0328_step_1_output.json", "r") as input_file:
    step_1_set = json.load(input_file)
    # dataset = json.load(input_file)
# with open("cot_dataset/dev/step_2.json", "r") as input_file:
#     step_1_set = json.load(input_file)
for i in range(len(dev_set)):
    db_id = dev_set[i]["db_id"]
    db_path = DATABASE_PATH_PATTERN.format(db_id=db_id)
    table_map = prompt.get_table_map(db_path, step_1_set[i])
    instruction, _ = prompt.step_2_prompt(
        db_path,
        dev_set[i]["question"],
        table_map,
        "",
        comment_map[db_id],
    )
    
    # instruction = step_1_set[i]["instruction"]
    # prompt_1 = prompt.step_1_prompt(db_path, case["question"])
    # instruction = case["instruction"]
    # instruction_2, output_2 = prompt.step_2_prompt(
    #     db_path, case["question"], case["query"]
    # )
    answer = text2sql(
        instruction,
    )

    answer = answer.strip()
    step2_output.append(answer)
    # step2_output.append(answer)

with open("0328_test_end2end.json", "w") as output:
    json.dump(step2_output, output)
