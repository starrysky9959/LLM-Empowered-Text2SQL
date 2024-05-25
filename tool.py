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

STEP_2_SYSTEM_PROMPT = """You are an experienced and professional database administrator. Given the [Schema] about the database, your task is to write a [SQL] to answer the [Question]. In the [Schema], each table consists of several columns and each line describes the name and type of the column. Some external knowledge about the [Schema] and [Question] is provided in the [Evidence]. 
Attention please, [SQL] should satisfy the following constraints:
- In `SELECT <column>`, must only use the column given in the [Schema].
- In `FROM <table>` or `JOIN <table>`, must only use the table given in the [Schema].
- In `JOIN`, must only use the columns with foreign key references in the [Schema].
- Without any specific instruction, use `ASC` for `ORDER BY` by default.
- Consider using `DISTINCT` when you need to eliminate duplicates.
- The content in quotes is case sensitive.
- Prioritize columns whose value are more relevant to the [Question].
""".lstrip()

STEP_2_INSTRUCTION_PATTERN = """
[Schema]
{schema}
[Question]
{question}
[Evidence]
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


def get_relevant_tables(sql: str):
    relevant_tables = [table.lower() for table in Parser(sql).tables]
    return relevant_tables


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
                sqlalchemy.text(f"select * from `{table_name}` limit {SAMPLE_SIZE}")
            )

            for row in result:
                schema_str += str(row) + "\n"
    return schema_str


def step_1(engine, question: str, evidence: str, sql=None):
    schema_str = get_table_schema(engine, None)
    instruction = STEP_1_INSTRUCTION_PATTERN.format(
        schema=schema_str, question=question, evidence=evidence
    )

    if sql is None:
        return instruction

    sql = format_sql(sql)
    relevant_tables = [table.lower() for table in Parser(sql).tables]
    output = STEP_1_OUTPUT_PATTERN.format(tables=",".join(relevant_tables))
    return instruction, output, relevant_tables


def step_2(engine, relevant_tables: list, question: str, evidence: str, sql=None):
    schema_str = get_table_schema(engine, relevant_tables)
    instruction = STEP_2_INSTRUCTION_PATTERN.format(
        schema=schema_str, question=question, evidence=evidence
    )
    if sql is None:
        return instruction

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


def under_max_tokens(
    tokenizer, max_tokens, system_prompt: str, instruction: str, output: str
):

    conversation = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": instruction,
        },
        {"role": "assistant", "content": output},
    ]
    tokens = len(tokenizer.apply_chat_template(conversation, tokenize=True))

    if tokens <= max_tokens:
        return True
    else:
        print(f"too many tokens:{tokens}")
