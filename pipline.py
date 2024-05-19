import json

from sql_metadata import Parser
import sqlalchemy
import sqlparse

SAMPLE_SIZE = 3
STEP_1_SYSTEM_PROMPT = """
You are an experienced and professional database administrator.
Given [Database Schema] and [Foreign Keys], your task is to identify the [Relevant Tables] to answer the [Question].
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

STEP_2_SYSTEM_PROMPT = """
Given [Database Schema] and [Foreign Keys], your task is to write a [SQL Query] to answer the [Question].
[Constraints] Your [SQL Query] should satisfy the following constraints:
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
[SQL Query]
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


def get_table_schema(engine, table_names=None):
    # 使用inspect获取数据库的元数据
    inspector = sqlalchemy.inspect(engine)
    if table_names is None:
        table_names = inspector.get_table_names()
    schema_str = ""

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
        schema_str + "Sample rows:"
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
    output_path: str,
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

    with open(f"{output_path}/step_1.json", "w") as output_file:
        json.dump(step_1_dataset, output_file)
    with open(f"{output_path}/step_2.json", "w") as output_file:
        json.dump(step_2_dataset, output_file)


if __name__ == "__main__":
    import argparse

    # 创建解析器
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="user defined dataset path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="otuput path for finetune dataset",
    )

    # 解析参数
    args = parser.parse_args()

    user_dataset_to_finetune_dataset(args.dataset_path, args.output_path)
