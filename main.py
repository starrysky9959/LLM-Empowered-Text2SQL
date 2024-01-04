# %%
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import util
import sys
from accelerate import (
    init_empty_weights,
    infer_auto_device_map,
    load_checkpoint_in_model,
    dispatch_model,
)

print(sys.version)
print(torch.cuda.device_count())
print()  # = "0,1"


cuda_list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
memory = "35GiB"

max_memory = {int(cuda): memory for cuda in cuda_list}
print(max_memory)
# with init_empty_weights():
#     # 加载到meta设备中，不需要耗时，不需要消耗内存和显存
#     model = AutoModelForCausalLM.from_pretrained(
#         "../deepseek-coder-33b-instruct",
#         trust_remote_code=True,
#         torch_dtype=torch.bfloat16,
#     )

# device_map = infer_auto_device_map(
#     model, max_memory=max_memory
# )  # 自动划分每个层的设备
# load_checkpoint_in_model(model, "../deepseek-coder-33b-instruct", device_map=device_map)  # 加载权重
# model = dispatch_model(model, device_map=device_map)  # 并分配到具体的设备上


tokenizer = AutoTokenizer.from_pretrained(
    "../deepseek-coder-6.7b-instruct", trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "../deepseek-coder-6.7b-instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)#.cuda()

Spider_Dataset_Path = "../spider/"
Dev_Json_Path = Spider_Dataset_Path + "dev.json"
Database_Path = Spider_Dataset_Path + "database/"

# %%
import re


pattern = re.compile(r"```sql (.+) ```", re.S)

# \nComplete .
system_prompt = "### Instruction:\nGiven the following SQLite database schema:\n"


def text2sql(
    schema: str,
    question: str,
):
    # conversation = [
    #     {"role": "system", "content": system_prompt+schema},
    #     # {"role": "user", "content": "### Complete sqlite SQL query only and with no explanation \n ### Sqlite SQL tables, with their properties: \n# \n# stadium(Stadium_ID,Location,Name,Capacity,Highest,Lowest,Average)\n;# singer(Singer_ID,Name,Country,Song_Name,Song_release_year,Age,Is_male)\n;# concert(concert_ID,concert_Name,Theme,Stadium_ID,Year)\n;# singer_in_concert(concert_ID,Singer_ID)\n.# \n ### How many singers do we have? \n SELECT"},
    #     # {"role": "system", "content": schema},
    #     {"role": "user", "content": question+"\nSELECT"},
    # ]
    conversation = (
        system_prompt
        + schema
        + "\nWrite the SQL query only to answer this question without explanation: "
        + question
        + "\n### Response:\n"
    )
    print("conversation: " + conversation)
    # print(conv)
    input_text = tokenizer(conversation, return_tensors="pt").to(model.device)
    # input_text = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)  # type: ignore

    outputs = model.generate(
        **input_text,
        # temperature=1.0,
        # do_sample=True,
        max_new_tokens=512,
        # repetition_penalty=1.5,
        pad_token_id=0,
        bos_token_id=32013,
        eos_token_id=32021,
    )
    # [len(input_text[0]) :]
    answer = tokenizer.decode(
        outputs[0][len(input_text[0]) :], skip_special_tokens=True
    )
    return answer


fo = open("result-6.7b-2.txt", "w")


with open(Dev_Json_Path, "r") as dataset_file:
    dataset = json.load(dataset_file)
    for test_case in dataset[:]:
        db_id = test_case["db_id"]
        question = test_case["question"]

        answer = text2sql(util.get_sqlite_schema_str(Database_Path, db_id), question)
        print("answer",answer)
        # answer = answer[: answer.find(";") + 1]
        # print("answer",answer)
        answer = " ".join(answer.split())
        print("answer",answer)
        fo.write(answer + "\n")
        # print(answer)

        # break
        # continue
        # sql = pattern.findall(answer)
        # if len(sql) > 0:
        #     formatted_answer = sql[0]

        # else:
        #     formatted_answer = answer
        # print(formatted_answer)
        # print(is_valid(db_id, formatted_answer))

        # break
        # if i > 10:
        #     break
# 关闭打开的文件
fo.close()
