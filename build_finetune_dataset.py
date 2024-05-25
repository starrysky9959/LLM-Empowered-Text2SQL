import json
from dotenv import load_dotenv
import os
import ruamel.yaml
from transformers import AutoTokenizer
from tool import (
    get_engine,
    step_2,
    step_1,
    STEP_1_SYSTEM_PROMPT,
    STEP_2_SYSTEM_PROMPT,
    get_relevant_tables,
    format_sql,
)
import tool


load_dotenv()  # 从.env文件加载环境变量
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
tokenizer = AutoTokenizer.from_pretrained(
    os.path.abspath(os.getenv("MODEL_PATH")), trust_remote_code=True
)
MAX_TOKENS = int(os.getenv("MAX_TOKENS"))


def user_dataset_to_finetune_dataset(
    user_dataset_path: str,
    finetune_step_1_path: str,
    finetune_step_2_path: str,
    all_path: str,
    comment_dataset_path=None,
):
    with open(user_dataset_path, "r") as input_file:
        user_dataset = json.load(input_file)
    if comment_dataset_path is not None:
        with open(comment_dataset_path, "r") as input_file:
            comment_dataset = json.load(input_file)

    step_1_dataset = []
    step_2_dataset = []
    for user_case in user_dataset:
        engine = get_engine(user_case["database"])
        print("dbname", user_case["database"])
        sql = format_sql(user_case["sql"])
        instruction, output = step_2(
            engine, None, user_case["question"], user_case["evidence"], sql
        )
        relevant_tables = get_relevant_tables(sql)
        if tool.under_max_tokens(
            tokenizer, MAX_TOKENS, STEP_2_SYSTEM_PROMPT, instruction, output
        ):
            step_1_dataset.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "system": STEP_2_SYSTEM_PROMPT,
                }
            )

        instruction, output = step_2(
            engine, relevant_tables, user_case["question"], user_case["evidence"], sql
        )
        if tool.under_max_tokens(
            tokenizer, MAX_TOKENS, STEP_2_SYSTEM_PROMPT, instruction, output
        ):
            step_2_dataset.append(
                {
                    "instruction": instruction,
                    "output": output,
                    "system": STEP_2_SYSTEM_PROMPT,
                }
            )

    # with open(finetune_step_1_path, "w") as output_file:
    #     json.dump(step_1_dataset, output_file)
    # with open(finetune_step_2_path, "w") as output_file:
    #     json.dump(step_2_dataset, output_file)
    with open(all_path, "w") as output_file:
        json.dump(step_1_dataset + step_2_dataset, output_file)


def add_custom_dataset_to_llama_factory(
    step_1_path: str,
    step_2_path: str,
    all_path: str,
):
    config_file_path = "./LLaMA-Factory/data/dataset_info.json"

    # 读取现有的JSON文件
    with open(config_file_path, "r") as file:
        data = json.load(file)
    data["text2sql"] = {
        "file_name": all_path,
        "columns": {
            "prompt": "instruction",
            "response": "output",
            "system": "system",
        },
    }
    data["text2sql_step_1"] = {
        "file_name": step_1_path,
        "columns": {
            "prompt": "instruction",
            "response": "output",
            "system": "system",
        },
    }

    data["text2sql_step_2"] = {
        "file_name": step_2_path,
        "columns": {
            "prompt": "instruction",
            "response": "output",
            "system": "system",
        },
    }

    # 将更新后的数据写回JSON文件
    with open(config_file_path, "w") as file:
        json.dump(data, file)


def config_llama_factory_lora_yaml(yaml_file_path: str):
    with open(yaml_file_path, "r") as file:
        data = yaml.load(file)

    data["model_name_or_path"] = os.path.abspath(os.getenv("MODEL_PATH"))
    data["output_dir"] = os.path.abspath(os.getenv("OUTPUT_MODEL_DIR"))

    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file)


def config_llama_factory_merge_yaml(yaml_file_path: str):
    with open(yaml_file_path, "r") as file:
        data = yaml.load(file)

    data["model_name_or_path"] = os.path.abspath(os.getenv("MODEL_PATH"))
    data["adapter_name_or_path"] = os.path.abspath(os.getenv("OUTPUT_MODEL_DIR"))
    data["template"] = os.path.abspath(os.getenv("MODEL_TEMPLATE"))
    data["export_dir"] = os.path.abspath(os.getenv("EXPORT_MODEL_DIR"))

    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file)


if __name__ == "__main__":
    step_1_path = os.path.abspath(os.getenv("FINETUNE_DATASET_STEP_1_PATH"))
    step_2_path = os.path.abspath(os.getenv("FINETUNE_DATASET_STEP_2_PATH"))
    all_path = os.path.abspath(os.getenv("FINETUNE_DATASET_ALL_PATH"))
    user_dataset_to_finetune_dataset(
        os.getenv("USER_DATASET_PATH"), step_1_path, step_2_path, all_path
    )
    add_custom_dataset_to_llama_factory(step_1_path, step_2_path, all_path)
    config_llama_factory_lora_yaml("./text2sql_lora_sft.yaml")
    # config_llama_factory_lora_yaml("./text2sql_step_1_lora_sft.yaml")
    # config_llama_factory_lora_yaml("./text2sql_step_2_lora_sft.yaml")
    config_llama_factory_merge_yaml("./merge.yaml")
