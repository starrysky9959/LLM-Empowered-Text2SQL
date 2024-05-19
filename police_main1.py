import prompt

import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

    conversation.append(
        {
            "role": "user",
            "content": instruction,
        },
    )

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
db_path = "/home/bingxing2/home/scx8900/projects/police_data/police.db"
with open("police_input.json", "r") as input_file:
    dataset = json.load(input_file)
    
    
for case in dataset[:]:
    question = case["question"]
    instruction = prompt.step_1_prompt(db_path,question,"",{})
    
    answer = text2sql(
        instruction,
        [],
    )
    answer = answer.strip()
    step1_output.append(answer)
    # step2_output.append(answer)

with open("police_step1.json", "w") as output:
    json.dump(step1_output, output,ensure_ascii=False)
