import json
from dotenv import load_dotenv
import os

load_dotenv()  # 从.env文件加载环境变量


from sql_metadata import Parser
import sqlalchemy
import sqlparse

SAMPLE_SIZE = 3
STEP_1_SYSTEM_PROMPT = """You are an experienced and professional database administrator. Given [Database Schema] and [Foreign Keys], your task is to identify the [Relevant Tables] to answer the [Question].
""".lstrip()

STEP_1_INSTRUCTION_PATTERN = """
[Database Schema] Every table consists of several columns. Each line describes the name, type of the column and optional value examples. In some cases, column name can be ambiguous, and extra comment is provided to assist in understanding.
{schema}
[Question]
{question}
[Evidence] Some external knowledge about the question.
{evidence}
[Relevant Tables]
""".lstrip()

STEP_1_OUTPUT_PATTERN = """
{tables}
""".strip()

STEP_2_SYSTEM_PROMPT = """You are an experienced and professional database administrator. Given [Database Schema] and [Foreign Keys], your task is to write a [SQL] to answer the [Question].
[Constraints] Your [SQL] should satisfy the following constraints:
- In `SELECT <column>`, must only use the column given in the [Database Schema].
- In `FROM <table>` or `JOIN <table>`, must only use the table given in the [Database Schema].
- In `JOIN`, must only use the tables and columns in the [Foreign keys].
- Without any specific instructions, Use `ASC` for `ORDER BY` by default, 
- Consider use `DISTINCT` when you need to eliminate duplicates.
- The content in quotes is case sensitive.
- Prioritize column whose value are more relevant to the [Question].
""".lstrip()

STEP_2_INSTRUCTION_PATTERN = """
[Database Schema] Every table consists of several columns. Each line describes the name, type of the column and optional value examples. In some cases, column name can be ambiguous, and extra comment is provided to assist in understanding.
{schema}
[Question]
{question}
[Evidence] Some external knowledge about the question.
{evidence}
[SQL]
""".lstrip()

STEP_2_OUTPUT_PATTERN = """
{sql}
""".strip()


def format_sql(sql: str):
    if sql[-1] != ";":
        sql += ";"
    parsed_query = sqlparse.parse(sql)[0]
    formatted_query = sqlparse.format(
        str(parsed_query),
        reindent=False,
        keyword_case="upper",
        identifier_case="lower",
        strip_whitespace=True,
    )
    return formatted_query


def get_table_schema(engine, revelant_tables=None):
    # 使用inspect获取数据库的元数据
    inspector = sqlalchemy.inspect(engine)
    if revelant_tables is None:
        table_names = inspector.get_table_names()
    else:
        table_names = [
            table
            for table in inspector.get_table_names()
            if table.lower() in revelant_tables
        ]

    schema_str = ""
    # print("tables :", table_names)
    for table_name in table_names:
        schema_str += f"Table: {table_name}\n"

        # 获取表的所有列信息
        columns = inspector.get_columns(table_name)
        for column in columns:
            schema_str += f"  Column: {column['name']}, Type: {column['type']}\n"

        # 获取表的主键信息
        primary_keys = inspector.get_pk_constraint(table_name)
        schema_str += f"  Primary Key: {primary_keys['constrained_columns']}\n"

        # 获取表的外键信息
        foreign_keys = inspector.get_foreign_keys(table_name)
        for foreign_key in foreign_keys:
            schema_str += f"  Foreign Key: {foreign_key['constrained_columns']}, References: {foreign_key['referred_table']}.{foreign_key['referred_columns']}\n"
        schema_str += f"Sample rows from {table_name}:\n"
        with engine.connect() as conn:
            result = conn.execute(
                sqlalchemy.text(f"select * from {table_name} limit {SAMPLE_SIZE}")
            )

            for row in result:
                schema_str += str(row) + "\n"
    return schema_str


def step_1(engine, question: str, sql: str, evidence: str):
    schema_str = get_table_schema(engine, None)
    instruction = STEP_1_INSTRUCTION_PATTERN.format(
        schema=schema_str, question=question, evidence=evidence
    )

    sql = format_sql(sql)
    relevant_tables = [table.lower() for table in Parser(sql).tables]
    output = STEP_1_OUTPUT_PATTERN.format(tables=",".join(relevant_tables))
    return instruction, output, relevant_tables


def step_2(engine, relevant_tables: list, question: str, sql: str, evidence: str):
    schema_str = get_table_schema(engine, relevant_tables)
    instruction = STEP_2_INSTRUCTION_PATTERN.format(
        schema=schema_str, question=question, evidence=evidence
    )

    sql = format_sql(sql)
    output = STEP_2_OUTPUT_PATTERN.format(sql=sql)
    return instruction, output


def get_engine(database: dict):
    db_type = database["type"]
    url = database["url"]
    if db_type == "sqlite":
        engine = sqlalchemy.create_engine(f"sqlite:///{url}")
    else:
        assert False
    return engine


def user_dataset_to_finetune_dataset(
    user_dataset_path: str,
    finetune_step_1_path: str,
    finetune_step_2_path: str,
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
        # print("dbname", user_case["database"])
        instruction, output, relevant_tables = step_1(
            engine,
            user_case["question"],
            user_case["sql"],
            user_case["evidence"],
        )

        step_1_dataset.append(
            {
                "instruction": instruction,
                "output": output,
                "system": STEP_1_SYSTEM_PROMPT,
            }
        )

        instruction, output = step_2(
            engine,
            relevant_tables,
            user_case["question"],
            user_case["sql"],
            user_case["evidence"],
        )
        step_2_dataset.append(
            {
                "instruction": instruction,
                "output": output,
                "system": STEP_1_SYSTEM_PROMPT,
            }
        )

    with open(finetune_step_1_path, "w") as output_file:
        json.dump(step_1_dataset, output_file)
    with open(finetune_step_2_path, "w") as output_file:
        json.dump(step_2_dataset, output_file)


def add_custom_dataset_to_llama_factory(step_1_path: str, step_2_path: str):
    config_file_path = "./LLaMA-Factory/data/dataset_info.json"

    # 读取现有的JSON文件
    with open(config_file_path, "r") as file:
        data = json.load(file)
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


def config_llama_factory_yaml(yaml_file_path: str):
    import ruamel.yaml

    yaml = ruamel.yaml.YAML()

    yaml.preserve_quotes = True

    with open(yaml_file_path, "r") as file:
        data = yaml.load(file)

    data["model_name_or_path"] = os.path.abspath(os.getenv("MODEL_PATH"))
    data["output_dir"] = os.path.abspath(os.getenv("OUTPUT_MODEL_STEP_1_DIR"))

    with open(yaml_file_path, "w") as file:
        yaml.dump(data, file)


if __name__ == "__main__":
    step_1_path = os.path.abspath(os.getenv("FINETUNE_DATASET_STEP_1_PATH"))
    step_2_path = os.path.abspath(os.getenv("FINETUNE_DATASET_STEP_2_PATH"))
    user_dataset_to_finetune_dataset(
        os.getenv("USER_DATASET_PATH"),
        step_1_path,
        step_2_path,
    )
    add_custom_dataset_to_llama_factory(
        step_1_path,
        step_2_path,
    )
    config_llama_factory_yaml("./text2sql_step_1_lora_sft.yaml")
    config_llama_factory_yaml("./text2sql_step_2_lora_sft.yaml")
