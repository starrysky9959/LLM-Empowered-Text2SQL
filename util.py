import sqlite3
import collections
import os
import re
import sqlite3
import spacy

import os
import psutil


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml

    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r"gpu_id {} 对应的显卡不存在!".format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(
        psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2
    )
    print("mem_total", mem_total)
    print("mem_free", mem_free)
    print("mem_process_used", mem_process_used)
    return mem_total, mem_free, mem_process_used


def log_memory():
    gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id=0)
    print(
        r"gpu memory：total {} MB, used {} MB， free {} MB".format(
            gpu_mem_total, gpu_mem_used, gpu_mem_free
        )
    )

    cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
    print(
        r"memory: total {} MB, used {} MB， free {} MB".format(
            cpu_mem_total, cpu_mem_free, cpu_mem_process_used
        )
    )


nlp = spacy.load("en_core_web_sm")

# from transformers import AutoTokenizer

# from utils.enums import LLM
from sql_metadata import Parser

from constants import DOMAIN_TAG, FUZZY_MATCH_THRESHOLD


class SqliteTable(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def get_tables(path_db):
    if not os.path.exists(path_db):
        raise RuntimeError(f"{path_db} not exists")

    # init sqlite connection
    connection = sqlite3.connect(path_db)
    cur = connection.cursor()

    # extract table information
    table_info = parse_db(path_db, cur)
    # TODO: ! add here
    table_names = get_table_names(cur=cur)

    res = list()
    for table_name in table_names:
        # schema
        schema = [_[1] for _ in cur.execute(f'PRAGMA table_info("{table_name}")')]

        # data
        data = None
        # data = cur.execute(f"SELECT * FROM {table_name} LIMIT 5").fetchall()

        # append table
        res.append(
            SqliteTable(
                name=table_name,
                schema=schema,
                data=data,
                table_info=table_info.get(table_name, dict()),
            )
        )

    cur.close()
    return res


def parse_db(path_db, cur=None):
    """Parse the sql file and extract primary and foreign keys

    :param path_file:
    :return:
    """
    table_info = dict()
    table_names = get_table_names(path_db, cur)

    for table_name in table_names:
        pks = get_primary_key(table_name, path_db, cur)
        fks = get_foreign_key(table_name, path_db, cur)

        table_info[table_name] = {"primary_key": pks, "foreign_key": fks}
    return table_info


def execute_query(queries, path_db=None, cur=None):
    """Execute queries and return results. Reuse cur if it's not None."""
    assert not (
        path_db is None and cur is None
    ), "path_db and cur cannot be NoneType at the same time"

    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    if isinstance(queries, str):
        results = cur.execute(queries).fetchall()
    elif isinstance(queries, list):
        results = list()
        for query in queries:
            res = cur.execute(query).fetchall()
            results.append(res)
    else:
        raise TypeError(f"queries cannot be {type(queries)}")

    # close the connection if needed
    if close_in_func:
        con.close()

    return results


def format_foreign_key(table_name: str, res: list):
    # FROM: self key | TO: target key
    res_clean = list()
    for row in res:
        table, source, to = row[2:5]
        row_clean = f"({table_name}.{source}, {table}.{to})"
        res_clean.append(row_clean)
    return res_clean


def get_foreign_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f'PRAGMA foreign_key_list("{table_name}")', path_db, cur)
    res = format_foreign_key(table_name, res_raw)
    return res


def get_primary_key(table_name, path_db=None, cur=None):
    res_raw = execute_query(f'PRAGMA table_info("{table_name}")', path_db, cur)
    pks = list()
    for row in res_raw:
        if row[5] == 1:
            pks.append(row[1])
    return pks


def get_table_names(path_db=None, cur=None):
    """Get names of all tables within the database, and reuse cur if it's not None"""
    table_names = execute_query(
        queries="SELECT name FROM sqlite_master WHERE type='table'",
        path_db=path_db,
        cur=cur,
    )
    table_names = [_[0] for _ in table_names]
    return table_names


def filter_json(raw_response: str) -> str:
    try:
        id_s = raw_response.index("{")
        id_e = raw_response.rindex("}")
        if id_s > id_e:
            raise ValueError("Wrong json format")
        else:
            return raw_response[id_s : id_e + 1]
    except ValueError:
        raise ValueError("Wrong json format")


def cost_estimate(n_tokens: int, model):
    return LLM.costs_per_thousand[model] * n_tokens / 1000


def get_sql_for_database(path_db=None, cur=None):
    close_in_func = False
    if cur is None:
        con = sqlite3.connect(path_db)
        cur = con.cursor()
        close_in_func = True

    table_names = get_table_names(path_db, cur)

    queries = [
        f"SELECT sql FROM sqlite_master WHERE tbl_name='{name}'" for name in table_names
    ]

    sqls = execute_query(queries, path_db, cur)

    if close_in_func:
        cur.close()

    return [_[0][0] for _ in sqls]


# def get_tokenizer(tokenizer_type: str):
#     return 0
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_type, use_fast=False)
#     return tokenizer


# def count_tokens(string: str, tokenizer_type: str = None, tokenizer=None):
#     return 0
#     # if tokenizer is None:
#     #     tokenizer = get_tokenizer(tokenizer_type)
#     #
#     # n_tokens = len(tokenizer.encode(string))
#     # return n_tokens


def sql_normalization(sql):
    sql = sql.strip()

    def white_space_fix(s):
        parsed_s = Parser(s)
        s = " ".join([token.value for token in parsed_s.tokens])

        return s

    # convert everything except text between single quotation marks to lower case
    def lower(s):
        in_quotation = False
        out_s = ""
        for char in s:
            if in_quotation:
                out_s += char
            else:
                out_s += char.lower()

            if char == "'":
                if in_quotation:
                    in_quotation = False
                else:
                    in_quotation = True

        return out_s

    # remove ";"
    def remove_semicolon(s):
        if s.endswith(";"):
            s = s[:-1]
        return s

    # double quotation -> single quotation
    def double2single(s):
        return s.replace('"', "'")

    def add_asc(s):
        pattern = re.compile(
            r"order by (?:\w+ \( \S+ \)|\w+\.\w+|\w+)(?: (?:\+|\-|\<|\<\=|\>|\>\=) (?:\w+ \( \S+ \)|\w+\.\w+|\w+))*"
        )
        if "order by" in s and "asc" not in s and "desc" not in s:
            for p_str in pattern.findall(s):
                s = s.replace(p_str, p_str + " asc")

        return s

    def sql_split(s):
        while "  " in s:
            s = s.replace("  ", " ")
        s = s.strip()
        i = 0
        toks = []
        while i < len(s):
            tok = ""
            if s[i] == "'":
                tok = tok + s[i]
                i += 1
                while i < len(s) and s[i] != "'":
                    tok = tok + s[i]
                    i += 1
                if i < len(s):
                    tok = tok + s[i]
                    i += 1
            else:
                while i < len(s) and s[i] != " ":
                    tok = tok + s[i]
                    i += 1
                while i < len(s) and s[i] == " ":
                    i += 1
            toks.append(tok)
        return toks

    def remove_table_alias(s):
        tables_aliases = Parser(s).tables_aliases
        new_tables_aliases = {}
        for i in range(1, 11):
            if "t{}".format(i) in tables_aliases.keys():
                new_tables_aliases["t{}".format(i)] = tables_aliases["t{}".format(i)]
        table_names = []
        for tok in sql_split(s):
            if "." in tok:
                table_names.append(tok.split(".")[0])
        for table_name in table_names:
            if table_name in tables_aliases.keys():
                new_tables_aliases[table_name] = tables_aliases[table_name]
        tables_aliases = new_tables_aliases

        new_s = []
        pre_tok = ""
        for tok in sql_split(s):
            if tok in tables_aliases.keys():
                if pre_tok == "as":
                    new_s = new_s[:-1]
                elif pre_tok != tables_aliases[tok]:
                    new_s.append(tables_aliases[tok])
            elif "." in tok:
                split_toks = tok.split(".")
                for i in range(len(split_toks)):
                    if (
                        len(split_toks[i]) > 2
                        and split_toks[i][0] == "'"
                        and split_toks[i][-1] == "'"
                    ):
                        split_toks[i] = split_toks[i].replace("'", "")
                        split_toks[i] = split_toks[i].lower()
                    if split_toks[i] in tables_aliases.keys():
                        split_toks[i] = tables_aliases[split_toks[i]]
                new_s.append(".".join(split_toks))
            else:
                new_s.append(tok)
            pre_tok = tok

        # remove as
        s = new_s
        new_s = []
        for i in range(len(s)):
            if s[i] == "as":
                continue
            if i > 0 and s[i - 1] == "as":
                continue
            new_s.append(s[i])
        new_s = " ".join(new_s)

        # for k, v in tables_aliases.items():
        #     s = s.replace("as " + k + " ", "")
        #     s = s.replace(k, v)

        return new_s

    processing_func = lambda x: remove_table_alias(
        add_asc(lower(white_space_fix(double2single(remove_semicolon(x)))))
    )

    return processing_func(sql.strip())


def sql2skeleton(sql: str, db_path: str) -> str:
    sql = sql_normalization(sql)
    (
        table_names_original,
        column_names_original,
        table_dot_column_names_original,
    ) = get_sqlite_schema_dict(db_path)

    parsed_sql = Parser(sql)
    new_sql_tokens = []
    for token in parsed_sql.tokens:
        # mask table names
        if token.value in table_names_original:
            new_sql_tokens.append("_")
        # mask column names
        elif (
            token.value in column_names_original
            or token.value in table_dot_column_names_original
        ):
            new_sql_tokens.append("_")
        # mask string values
        elif token.value.startswith("'") and token.value.endswith("'"):
            new_sql_tokens.append("_")
        # mask positive int number
        elif token.value.isdigit():
            new_sql_tokens.append("_")
        # mask negative int number
        elif isNegativeInt(token.value):
            new_sql_tokens.append("_")
        # mask float number
        elif isFloat(token.value):
            new_sql_tokens.append("_")
        else:
            new_sql_tokens.append(token.value.strip())

    sql_skeleton = " ".join(new_sql_tokens)

    # remove JOIN ON keywords
    sql_skeleton = sql_skeleton.replace("on _ = _ and _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace("on _ = _ or _ = _", "on _ = _")
    sql_skeleton = sql_skeleton.replace(" on _ = _", "")
    pattern3 = re.compile("_ (?:join _ ?)+")
    sql_skeleton = re.sub(pattern3, "_ ", sql_skeleton)

    # "_ , _ , ..., _" -> "_"
    while "_ , _" in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("_ , _", "_")

    # remove clauses in WHERE keywords
    ops = ["=", "!=", ">", ">=", "<", "<="]
    for op in ops:
        if "_ {} _".format(op) in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("_ {} _".format(op), "_")
    while "where _ and _" in sql_skeleton or "where _ or _" in sql_skeleton:
        if "where _ and _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ and _", "where _")
        if "where _ or _" in sql_skeleton:
            sql_skeleton = sql_skeleton.replace("where _ or _", "where _")

    # remove additional spaces in the skeleton
    while "  " in sql_skeleton:
        sql_skeleton = sql_skeleton.replace("  ", " ")

    # double check for order by
    split_skeleton = sql_skeleton.split(" ")
    for i in range(2, len(split_skeleton)):
        if (
            split_skeleton[i - 2] == "order"
            and split_skeleton[i - 1] == "by"
            and split_skeleton[i] != "_"
        ):
            split_skeleton[i] = "_"
    sql_skeleton = " ".join(split_skeleton)

    return sql_skeleton


def isNegativeInt(string):
    if string.startswith("-") and string[1:].isdigit():
        return True
    else:
        return False


def isFloat(string):
    if string.startswith("-"):
        string = string[1:]

    s = string.split(".")
    if len(s) > 2:
        return False
    else:
        for s_i in s:
            if not s_i.isdigit():
                return False
        return True


def jaccard_similarity(skeleton1, skeleton2):
    tokens1 = skeleton1.strip().split(" ")
    tokens2 = skeleton2.strip().split(" ")
    total = len(tokens1) + len(tokens2)

    def list_to_dict(tokens):
        token_dict = collections.defaultdict(int)
        for t in tokens:
            token_dict[t] += 1
        return token_dict

    token_dict1 = list_to_dict(tokens1)
    token_dict2 = list_to_dict(tokens2)

    intersection = 0
    for t in token_dict1:
        if t in token_dict2:
            intersection += min(token_dict1[t], token_dict2[t])
    union = (len(tokens1) + len(tokens2)) - intersection
    return float(intersection) / union


def get_sqlite_schema_str(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    results = cursor.fetchall()

    schema = ""
    for result in results:
        table_schema = (
            " ".join(result[0].split()).replace('"', "").replace("'", "") + ";\n"
        )
        schema += table_schema
    conn.close()

    return schema.strip()


def get_sqlite_schema_table_map(db_path: str):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    table_map = {}
    for table in tables:
        table_name = table[0].lower()

        column_names = set()
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        for column in columns:
            column_name = column[1].lower()
            column_names.add(column_name)

        table_map[table_name] = column_names

    cursor.close()
    conn.close()
    return table_map


def get_sqlite_sample_value(
    db_path: str,
    table_name: str,
    column_name: str,
    # is_string: bool,
    question: str,
    limit: int,
):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()
    # print("table"table_name)
    # if not is_string:
    #     cursor.execute(
    #         f"SELECT DISTINCT `{column_name}` FROM `{table_name}` LIMIT {limit};"
    #     )
    # else:
    cursor.execute(f"SELECT DISTINCT `{column_name}` FROM `{table_name}`;")
    result = cursor.fetchall()
    samples = [str(t[0]) for t in result]
    cursor.close()
    conn.close()

    # if not is_string:
    #     return samples

    from thefuzz import process

    matches = process.extractBests(question, samples, limit=limit, score_cutoff=0)
    return [t[0] for t in matches]


def get_sqlite_schema_table_with_type_map(db_path: str, tables: list):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    if len(tables) == 0:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        tables = [table[0].lower() for table in tables]

    table_map = {}
    for table_name in tables:
        column_info = []
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        primary_keys = []
        for column in columns:
            column_name = column[1].lower()
            if column[5] == 1:
                primary_keys.append(column_name)
            # name,type
            column_info.append((column_name, column[2].upper()))

        table_map[table_name] = (primary_keys, column_info)

    cursor.close()
    conn.close()
    return table_map


def get_sqlite_schema_dict(db_path: str) -> tuple:
    """
    Lowercase
    """
    table_names_original = []
    table_dot_column_names_original = []
    column_names_original = ["*"]

    conn = sqlite3.connect(db_path)

    # 创建游标对象
    cursor = conn.cursor()

    # 执行查询获取表的模式信息
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    # 打印表名和列名
    for table in tables:
        table_name = table[0].lower()
        table_names_original.append(table_name)
        table_dot_column_names_original.append(table_name + ".*")
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        for column in columns:
            column_name = column[1].lower()
            column_names_original.append(column_name)
            table_dot_column_names_original.append(
                "{0}.{1}".format(table_name, column_name)
            )

    cursor.close()
    conn.close()

    return (
        table_names_original,
        column_names_original,
        table_dot_column_names_original,
    )


def get_sqlite_foreign_key(db_path: str, table_name: str):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    fk_info = []
    cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
    columns = cursor.fetchall()
    for column in columns:
        fk_table = column[2].lower()
        from_column = column[3].lower()
        to_column = column[4].lower()
        fk_info.append((from_column, fk_table, to_column))

    cursor.close()
    conn.close()
    return fk_info


def get_sqlite_all_foreign_key(db_path: str):
    conn = sqlite3.connect(db_path)

    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    # 打印表名和列名
    fk_info = []
    for table in tables:
        table_name = table[0].lower()
        cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
        columns = cursor.fetchall()
        for column in columns:
            fk_table = column[2].lower()
            from_column = column[3].lower()
            to_column = column[4].lower()
            fk_info.append(f"{table_name}.{from_column} = {fk_table}.{to_column}")

    cursor.close()
    conn.close()
    return fk_info


def get_sqlite_alpaca_sft_prompt_schema(db_path: str):
    schema = {}

    conn = sqlite3.connect(db_path)

    # 创建游标对象
    cursor = conn.cursor()

    # 执行查询获取表的模式信息
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()

    # 打印表名和列名
    for table in tables:
        table_name = table[0].lower()
        column_names = []
        cursor.execute(f"PRAGMA table_info(`{table_name}`);")
        columns = cursor.fetchall()
        for column in columns:
            column_name = "{name}({type})".format(
                name=column[1].lower(), type=column[2].lower()
            )
            column_names.append(column_name)
        schema[table_name] = column_names

    cursor.close()
    conn.close()
    return schema


def get_sqlite_sample_csv(db_path: str, table_name: str) -> str:
    output = ""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(f"PRAGMA table_info(`{table_name}`);")
    columns = cur.fetchall()
    for column in columns:
        output += column[1] + ","
    output += "\n"

    cur.execute(f"SELECT * FROM {table_name} LIMIT 1")
    row = cur.fetchone()
    if row is not None:
        for field in row:
            output += str(field) + ","

    conn.close()
    return output


def get_sqlite_terminology_list(db_path: str) -> list:
    table_names_original, column_names_original, _ = get_sqlite_schema_dict(db_path)
    terminology_list = []
    for name in table_names_original:
        terminology_list.append(name)
    for name in column_names_original:
        terminology_list.append(name)
    return terminology_list


def is_terminology(word: str, terminology_list: list) -> bool:
    from thefuzz import process

    _, val = process.extractOne(word, terminology_list)
    return val > FUZZY_MATCH_THRESHOLD


def question_to_masked_tokens(question: str, db_path: str) -> list[str]:
    terminology_list = get_sqlite_terminology_list(db_path)
    masked_tokens = []
    for token in nlp(question):
        if token.is_punct or token.is_space:
            masked_tokens.append(token.tag_)
        elif is_terminology(token.lemma_, terminology_list):
            masked_tokens.append(DOMAIN_TAG)
        else:
            masked_tokens.append(token.tag_)
    return masked_tokens


def mask_question(question: str, db_path: str) -> str:
    return " ".join(question_to_masked_tokens(question.lower(), db_path))


def get_relevant_schema(info: str):
    schema = {}
    for line in info.splitlines():
        line = line.strip()
        parts = line.split(":")
        table = parts[0].strip()
        columns = parts[1].strip().split(",")
        columns = [col.strip() for col in columns]
        schema[table] = columns
    return schema
