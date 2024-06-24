from typing import List, Set, Tuple
from sql_metadata import Parser
import sqlalchemy
import sqlparse
from datetime import date
import logging
import networkx


class ColumnSchema:
    def __init__(
        self, column_name: str, column_description: str = None, value: str = None
    ):
        self.column_name = column_name
        self.column_description = column_description
        self.sample_values = set()
        if value is not None:
            self.sample_values.add(value)

    def add_sample_value(self, value: Set[str]):
        self.sample_values.update(value)

    def __hash__(self):
        return hash(self.column_name)

    def __eq__(self, other):
        if isinstance(other, ColumnSchema):
            return self.column_name == other.column_name
        return False

    def __repr__(self):
        return f"Column(column_name={self.column_name},\n column_description={self.column_description},\n sample_values={self.sample_values})"


class TableSchema:
    def __init__(self, table_name=None):
        self.table_name = table_name
        # <column_name, ColumnSchema>
        self.columns = {}

    def add_column(self, column: ColumnSchema):
        if isinstance(column, ColumnSchema):
            if column.column_name in self.columns:
                self.columns[column.column_name].add_sample_value(column.sample_values)
            else:
                self.columns[column.column_name] = column
        else:
            raise ValueError("column must be an instance of ColumnSchema")

    def __repr__(self):
        return f"Table(table_name={self.table_name}, columns={"\n".join(self.columns)})"


logging.basicConfig(
    level=logging.DEBUG,  # 设置日志级别为 DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    filename="example.log",  # 设置日志文件名
    filemode="w",
)  # 设置文件模式为覆盖写入

SAMPLE_SIZE = 2
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

COMMON_SYSTEM_PROMPT = """You are an experienced and professional database administrator. Given the [Schema] about the database, your task is to write a [SQL] to answer the [Question]. In the [Schema], each table consists of several columns and each line describes the name and type of the column. Some external knowledge about the [Schema] and [Question] is provided in the [Evidence]. 
Attention please, [SQL] should satisfy the following constraints:
- In `SELECT <column>`, must only use the column given in the [Schema].
- In `FROM <table>` or `JOIN <table>`, must only use the table given in the [Schema].
- In `JOIN`, must only use the columns with foreign key references in the [Schema].
- Without any specific instruction, use `ASC` for `ORDER BY` by default.
- Consider using `DISTINCT` when you need to eliminate duplicates.
- The content in quotes is case sensitive.
- Prioritize columns whose value are more relevant to the [Question].
""".lstrip()

COMMON_INSTRUCTION_PATTERN = """
[Schema]
{schema}
[Question]
{question}
[Evidence]
{evidence}
[Error SQL]
{error_sql}
[Error Message]
{error_message}
[SQL]
""".lstrip()
COMMON_OUTPUT_PATTERN = """
{sql}
""".strip()

SCHEMA_PATTERN = """

"""


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


def get_relevant_columns(sql: str):
    relevant_columns = [column.lower() for column in Parser(sql).columns]
    return relevant_columns


def get_string_columns_in_database(engine, table_name: str, lower_format: bool = False):
    assert engine is not None
    inspector = sqlalchemy.inspect(engine)
    columns = inspector.get_columns(table_name)
    if lower_format:
        string_columns = [
            column["name"].lower()
            for column in columns
            if isinstance(column["type"], sqlalchemy.String)
        ]
    else:
        string_columns = [
            column["name"]
            for column in columns
            if isinstance(column["type"], sqlalchemy.String)
        ]
    return string_columns


def get_columns_in_database(engine, table_name: str, lower_format: bool = False):
    """
    table_name in database
    """
    assert engine is not None
    inspector = sqlalchemy.inspect(engine)
    columns = inspector.get_columns(table_name)
    if lower_format:
        column_names = [column["name"].lower() for column in columns]
    else:
        column_names = [column["name"] for column in columns]
    return column_names


def get_tables_in_database(engine, lower_format=False):

    assert engine is not None
    inspector = sqlalchemy.inspect(engine)
    if lower_format:
        table_names = [table.lower() for table in inspector.get_table_names()]
    else:
        table_names = inspector.get_table_names()
    return table_names


def get_table_schema(
    engine,
    relevant_tables=None,
    relevant_columns=None,
    column_descriptions: dict = None,
    add_sample_row=True,
):
    # 使用inspect获取数据库的元数据
    inspector = sqlalchemy.inspect(engine)
    if relevant_tables is None:
        table_names = inspector.get_table_names()
    else:
        table_names = [
            table
            for table in inspector.get_table_names()
            if table.lower() in relevant_tables
        ]

    schema_str = ""
    for table_name in table_names:
        table_name_lower = table_name.lower()
        schema_str += f"Table: {table_name}\n"

        # 获取表的所有列信息
        columns = inspector.get_columns(table_name)

        if relevant_columns is not None:
            print("relevant_columns:", relevant_columns)
            columns = [
                col
                for col in columns
                if col["name"].lower() in relevant_columns
                or f"{table_name}.{col['name']}".lower() in relevant_columns
            ]

        for column in columns:
            if "comment" in column:
                comment = column["comment"]
            else:
                comment = None
            column_name_lower = column["name"].lower()
            schema_str += f"  Column: `{column['name']}`, Type: {column['type']}"

            if comment is None:
                if column_descriptions is not None:
                    key = f"{table_name_lower}.{column_name_lower}"
                    if key in column_descriptions:
                        schema_str += f", Comment: {column_descriptions[key]}"
            else:
                schema_str += f", Comment: {comment}"
            schema_str += "\n"
        # 获取表的主键信息
        primary_keys = inspector.get_pk_constraint(table_name)
        schema_str += f"  Primary Key: {primary_keys['constrained_columns']}\n"

        # 获取表的外键信息
        foreign_keys = inspector.get_foreign_keys(table_name)
        for foreign_key in foreign_keys:
            schema_str += f"  Foreign Key: {foreign_key['constrained_columns']}, References: {foreign_key['referred_table']}.{foreign_key['referred_columns']}\n"
        if add_sample_row:
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


def build_system_prompt():
    return COMMON_SYSTEM_PROMPT


def build_instruction(
    step_index: int,
    engine,
    question: str,
    evidence: str,
    relevant_tables: list = None,
    relevant_columns: list = None,
    sql: str = None,
    error_sql: str = None,
    error_message: str = None,
    add_current_date: bool = False,
):
    if step_index == 1:
        # complete schema
        schema_str = get_table_schema(engine, None, None)
    elif step_index == 2:
        # filtered table schema
        assert relevant_tables is not None
        schema_str = get_table_schema(engine, relevant_tables, None)
    elif step_index == 3:
        # filtered column schema
        assert relevant_tables is not None
        # assert relevant_columns is not None
        schema_str = get_table_schema(engine, relevant_tables, relevant_columns)
    elif step_index == 4:
        # revision
        assert relevant_tables is not None
        # assert relevant_columns is not None
        assert error_sql is not None
        assert error_message is not None
        schema_str = get_table_schema(engine, relevant_tables, relevant_columns)
    if add_current_date:
        evidence = f"Today is {date.today()}. " + evidence
    instruction = COMMON_INSTRUCTION_PATTERN.format(
        schema=schema_str,
        question=question,
        evidence=evidence,
        error_sql="",
        error_message="",
    )

    if sql is None:
        return instruction

    sql = format_sql(sql)
    output = COMMON_OUTPUT_PATTERN.format(sql=sql)

    return instruction, output


def is_valid_question(relevant_tables: list, all_table_names: List[str]) -> bool:
    return set(relevant_tables).issubset(set(all_table_names))


def text2sql(
    client,
    model_name: str,
    stop_token_ids: List[int],
    database_info: dict,
    question: str,
    evidence: str,
    temperature: float = 0,
) -> Tuple[bool, str]:
    try:
        # print(f"database_info:{database_info}")
        logging.debug(f"database_info:{database_info}")
        engine = get_engine(database_info)
        table_names = get_tables_in_database(engine)
    except Exception as e:
        print(e)
        return False, "无法连接到数据库"
    # step 1
    instruction = build_instruction(1, engine, question, evidence, None, None)
    openai_format_messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": instruction},
    ]
    response = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=openai_format_messages,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=False,  # Stream response
        extra_body={
            # "repetition_penalty": 1,
            "add_generation_prompt": True,
            "stop_token_ids": stop_token_ids,
        },
    )
    sql_v1 = response.choices[0].message.content

    # print(f"prompt_v1:{openai_format_messages}")
    # print("sqlv1:", sql_v1)

    relevant_tables = get_relevant_tables(sql_v1)
    if not is_valid_question(relevant_tables, table_names):
        return (
            False,
            "您的提问似乎与该数据库中的表无关或者存在难以判别的歧义，暂时无法进行回答。请修改数据库设置或问题，补充额外信息。",
        )

    instruction = build_instruction(
        2, engine, question, evidence, relevant_tables, None
    )
    openai_format_messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": instruction},
    ]
    response = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=openai_format_messages,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=False,  # Stream response
        extra_body={
            # "repetition_penalty": 1,
            "add_generation_prompt": True,
            "stop_token_ids": stop_token_ids,
        },
    )

    sql_v2 = response.choices[0].message.content
    # print(f"prompt_v2:{openai_format_messages}")
    # print("sqlv2:", sql_v2)

    relevant_columns = get_relevant_columns(sql_v2)
    if len(relevant_columns) == 0:
        relevant_columns = None

    # print(relevant_columns)
    instruction = build_instruction(
        3, engine, question, evidence, relevant_tables, relevant_columns
    )
    openai_format_messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": instruction},
    ]
    response = client.chat.completions.create(
        model=model_name,  # Model name to use
        messages=openai_format_messages,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=False,  # Stream response
        extra_body={
            # "repetition_penalty": 1,
            "add_generation_prompt": True,
            "stop_token_ids": stop_token_ids,
        },
    )
    sql_v3 = response.choices[0].message.content
    # print(f"prompt_v3:{openai_format_messages}")
    # print("sqlv3:", sql_v3)
    return True, sql_v3


def get_engine(database: dict):
    db_type = database["type"]

    if db_type == "sqlite":
        url = database["url"]
        engine = sqlalchemy.create_engine(f"sqlite:///{url}")
    elif db_type == "mysql":
        username = database["username"]
        password = database["password"]
        host = database["host"]
        port = database["port"]
        dbname = database["dbname"]

        engine = sqlalchemy.create_engine(
            f"mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}"
        )
    else:
        raise Exception("Unsupported database type")
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


def build_graph(engine):
    inspector = sqlalchemy.inspect(engine)
    table_names = inspector.get_table_names()
    G = networkx.DiGraph()
    for table_name in table_names:
        G.add_node(table_name)
        foreign_keys = inspector.get_foreign_keys(table_name)
        for foreign_key in foreign_keys:
            ref_table = foreign_key["referred_table"]
            G.add_edge(table_name, ref_table)
    return G


def one_hop_query(graph, table_name):
    if table_name in graph:
        neighbors = list(graph.neighbors(table_name))
        neighbors.insert(0, table_name)  # 将本身插入到结果的最前面
        return neighbors
    else:
        return [table_name] if table_name in graph else []


# 2跳查询：查询某个表的所有邻居的邻居，并保留本身和邻居
def two_hop_query(graph, table_name):
    if table_name in graph:
        neighbors = list(graph.neighbors(table_name))
        two_hop_neighbors = set(neighbors)
        for neighbor in neighbors:
            two_hop_neighbors.update(graph.neighbors(neighbor))
        # 移除直接邻居和自身
        two_hop_neighbors.discard(table_name)
        # 将本身和直接邻居插入到结果的最前面
        result = [table_name] + neighbors + list(two_hop_neighbors)
        return result
    else:
        return [table_name] if table_name in graph else []
