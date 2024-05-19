import json

from transformers import AutoTokenizer
import sqlparse
import util
from sql_metadata import Parser

# model_path = "/home/bingxing2/home/scx8900/projects/deepseek-coder-6.7b-instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# MAX_TOKENS = 4096
STEP_1_SYSTEM_PROMPT = (
    "You are an experienced and professional database administrator.\n"
)
STEP_1_INSTRUCTION_PATTERN = """
Given [Database Schema] and [Foreign Keys], your task is to identify the [Relevant Tables] to answer the [Question].

[Database Schema] Every table consists of several columns. Each line describes the name, type of the column and optional value examples. In some cases, column name can be ambiguous, and extra comment is provided to assist in understanding.
{schema}
[Foreign keys]
{foreign_key}
[Question]
{question}
[Evidence] Some external knowledge about the question.
{evidence}
[Relevant Tables]
""".lstrip()
STEP_1_OUTPUT_PATTERN = """
{tables}
""".strip()


def get_tables_from_step_1(out: str):
    # position = out.find("[Relevant Tables]\n")
    # sql = out[position + 18 :].strip()
    return list(out.split(","))


def step_1_prompt(db_path: str, question: str, evidence: str, comment_map: dict):
    tables = util.get_sqlite_schema_table_with_type_map(db_path, [])

    schema_str = build_schema_str(db_path, tables, question, comment_map, 3)
    foreign_key = build_foreign_key_str(db_path, list(tables.keys()))
    instruction = STEP_1_INSTRUCTION_PATTERN.format(
        schema=schema_str, foreign_key=foreign_key, question=question, evidence=evidence
    )

    return instruction


def step_1_gold_output(db_path: str, query: str):
    formatted_query = format_query(query)
    relevant_tables = get_relevant_tables(db_path, formatted_query)
    table_str = ",".join(relevant_tables)
    return STEP_1_OUTPUT_PATTERN.format(tables=table_str)


STEP_2_INSTRUCTION_PATTERN = """
Given [Database Schema] and [Foreign Keys], your task is to write a [SQL Query] to answer the [Question].

[Database Schema] Every table consists of several columns. Each line describes the name, type of the column and optional value examples. In some cases, column name can be ambiguous, and extra comment is provided to assist in understanding.
{schema}
[Foreign keys]
{foreign_key}
[Question]
{question}
[Evidence] Some external knowledge about the question.
{evidence}
[Constraints] Your [SQL Query] should satisfy the following constraints:
- In `SELECT <column>`, must only use the column given in the [Database Schema].
- In `FROM <table>` or `JOIN <table>`, must only use the table given in the [Database Schema].
- In `JOIN`, must only use the tables and columns in the [Foreign keys].
- Without any specific instructions, Use `ASC` for `ORDER BY` by default, 
- Consider use `DISTINCT` when you need to eliminate duplicates.
- The content in quotes is case sensitive.
- Prioritize column whose value are more relevant to the [Question].
[SQL Query]
""".lstrip()

STEP_2_OUTPUT_PATTERN = """
{answer}
""".strip()


def build_schema_str(db_path: str, table: dict, question: str, comment_map, topk=3):
    schema_str = ""
    for table, info in table.items():
        pk, column_info = info
        pk_str = ",".join(pk)
        schema_str += f"- Table: {table}. Primary Key: ({pk_str})\n"
        for column_name, column_type in column_info:
            # print(column_name)
            # print(column_type)

            column_str = f"{column_name}, {column_type}"
            if len(comment_map) > 0:
                comment = comment_map[f"{table}.{column_name}"]
                if need_comment(column_name, comment):
                    column_str += f", Comment: '{comment}'"
            if check_string_type(column_type):
                sample_values = util.get_sqlite_sample_value(
                    db_path,
                    table,
                    column_name,
                    question,
                    topk,
                )
                column_str += f", Value Examples: {sample_values}"

            schema_str += column_str + "\n"
    return schema_str.strip()


def build_foreign_key_str(db_path: str, tables: list):
    foreign_key = ""
    for table in tables:
        for from_column, fk_table, to_column in util.get_sqlite_foreign_key(
            db_path, table
        ):
            if fk_table in tables:
                foreign_key += (
                    f"{table}({from_column}) references {fk_table}({to_column})\n"
                )
    return foreign_key.strip()


def get_relevant_tables(db_path: str, query: str, list_only=True):
    relevant_tables = Parser(query).tables
    relevant_tables = [table.lower() for table in relevant_tables]
    if list_only:
        return relevant_tables

    table_map = util.get_sqlite_schema_table_with_type_map(db_path, relevant_tables)
    return table_map


def format_query(query: str):
    if query[-1] != ";":
        query += ";"
    parsed_query = sqlparse.parse(query)[0]
    formatted_query = sqlparse.format(
        str(parsed_query),
        reindent=False,
        keyword_case="upper",
        identifier_case="lower",
        strip_whitespace=True,
    )
    return formatted_query


def get_table_map(db_path: str, tables: list):
    relevant_tables = [table.lower() for table in tables]
    table_map = util.get_sqlite_schema_table_with_type_map(db_path, relevant_tables)
    return table_map


def step_2_prompt(
    db_path: str,
    question: str,
    relevant_tables: dict,
    query: str,
    comment_map: dict,
    evidence: str,
):
    if question[-1] != "\n":
        question += "\n"
    output = ""
    if len(relevant_tables) == 0:
        formatted_query = format_query(query)
        relevant_tables = get_relevant_tables(db_path, formatted_query, False)
        output = STEP_2_OUTPUT_PATTERN.format(answer=formatted_query)
    foreign_key = build_foreign_key_str(db_path, relevant_tables)
    schema_str = build_schema_str(db_path, relevant_tables, question, comment_map, 3)
    instruction = STEP_2_INSTRUCTION_PATTERN.format(
        schema=schema_str, foreign_key=foreign_key, question=question, evidence=evidence
    )

    return instruction, output


def check_string_type(s: str):
    return (
        (s.upper().find("CHAR") != -1)
        or (s.upper().find("TEXT") != -1)
        or (s.upper().find("BOOL") != -1)
        or (s.upper().find("NUMBER") != -1)
        or (s.upper().find("DATE") != -1)
    )


def need_comment(name: str, comment: str):
    name = name.replace("_", "").replace(" ", "").lower()
    comment = comment.replace("_", "").replace(" ", "").lower()
    return name != comment


def under_max_tokens(instruction: str, output: str):

    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": instruction,
        },
        {"role": "assistant", "content": output},
    ]
    tokens = len(tokenizer.apply_chat_template(conversation, tokenize=True))
    return tokens < MAX_TOKENS
