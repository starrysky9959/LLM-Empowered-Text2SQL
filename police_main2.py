# %%
import prompt


import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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


def get_tables(out: str):
    position = out.find("[Relevant Tables]\n")
    sql = out[position + 18 :].strip()
    return list(sql.split(","))


step1_output = []
step2_output = []
db_path = "/home/bingxing2/home/scx8900/projects/police_data/police.db"
with open("police_step1.json", "r") as input_file:
    step1_out = json.load(input_file)
with open("police_input.json", "r") as input_file:
    dataset = json.load(input_file)
for i in range(len(dataset)):
    case = dataset[i]
    table_infos = prompt.get_table_map(db_path, get_tables(step1_out[i]))
    print(table_infos)
    instruction,output = prompt.step_2_prompt(db_path, case["question"], table_infos, "", {},case["evidence"])

    answer = text2sql(
        instruction,
    )

    answer = answer.strip()
    step2_output.append(answer)

with open("police_step2.json", "w") as output:
    json.dump(step2_output, output,ensure_ascii=False)
