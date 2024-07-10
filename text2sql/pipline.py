# keyword extract
# rag->tableschema
# graph_build, one_hop
# tableschema
# select table,column
# generate sql
# revision
import json
import logging
import sys
from typing import Dict, Set
import ast
from openai import OpenAI
from sql_metadata import Parser
from util import PrecisionRecallAverageCalculator, SchemaSizeAverageCalculator

sys.path.append("/home/data2/luzhan/projects/LLM-Empowered-Text2SQL/text2sql")

from fk_graph import ForeignKeyGraph
from schema import TableSchema, ColumnSchema, DatabaseSchema
from rag import RAGHandler
import sqlalchemy
from prompt_template import (
    SQL_SYSTEM_PROMPT,
    SQL_INSTRUCTION_PATTERN,
    KEYWORDS_EXTRACT_PATTERN,
)
import logging


logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="example.log",
    filemode="w",
)

logger = logging.getLogger("example_logger")


class Text2SQLSolution:
    rag_schema: DatabaseSchema
    graph_schema: DatabaseSchema
    all_schema: DatabaseSchema
    openai_client: OpenAI

    def __init__(
        self,
        database: dict,
        # milvus_connection: dict,
        openai_client: OpenAI,
    ):
        self.database = database
        self.engine = self.get_engine()
        self.graph = ForeignKeyGraph(self.engine)
        self.rag_handler = RAGHandler()
        self.rag_schema = None
        self.graph_schema = None
        self.all_schema = self.get_table_schema(None)
        self.openai_client = openai_client
        self.model_name = "deepseek"
        self.stop_token_ids = [32021]

    def get_engine(self):
        db_type = self.database["type"]

        if db_type == "sqlite":
            url = self.database["url"]
            engine = sqlalchemy.create_engine(f"sqlite:///{url}")
        elif db_type == "mysql":
            username = self.database["username"]
            password = self.database["password"]
            host = self.database["host"]
            port = self.database["port"]
            dbname = self.database["dbname"]

            engine = sqlalchemy.create_engine(
                f"mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}"
            )
        else:
            raise Exception("Unsupported database type")
            assert False
        return engine

    def get_table_schema(self, table_names: Set[str]) -> DatabaseSchema:
        inspector = sqlalchemy.inspect(self.engine)

        if table_names is None:
            table_names = inspector.get_table_names()
        db_schema = DatabaseSchema(table_names=table_names)
        # print("table_names: ",table_names)
        for table_name in table_names:
            table_schema = db_schema[table_name]

            columns = inspector.get_columns(table_name)

            for column in columns:
                table_schema.add_column(
                    ColumnSchema(column_name=column["name"], column_type=column["type"])
                )
            # print(table_name)
            primary_keys = inspector.get_pk_constraint(table_name)
            table_schema.pk_names = primary_keys["constrained_columns"]

            foreign_keys = inspector.get_foreign_keys(table_name)
            for foreign_key in foreign_keys:
                column, referred_table, referred_column = (
                    foreign_key["constrained_columns"][0],
                    foreign_key["referred_table"],
                    foreign_key["referred_columns"][0],
                )
                # print(column)
                # print(referred_table)
                # print(referred_column)
                table_schema.fk_out.append((column, referred_table, referred_column))

                # ignore extra referred table for graph schema
                # one hop is enough
                if referred_table in db_schema.table_names():
                    db_schema[referred_table].fk_in.append(
                        (referred_column, table_name, column)
                    )
        # print(db_schema)
        return db_schema

    def build_database_schema_by_sql(self, sql: str)->DatabaseSchema:
        tables = dict()
        table_column_names = self.get_relevant_tables(sql, self.all_schema)
        print(f"table_column_names:{table_column_names}")
        for table_name, column_names in table_column_names.items():
            table_schema = TableSchema(table_name)
            for column_name in column_names:
                table_schema.add_column(ColumnSchema(column_name))
            tables[table_name] = table_schema
        return DatabaseSchema(tables=tables)

    def get_relevant_tables(self, sql: str, refer: DatabaseSchema):
        parser = Parser(sql)
        relevant_tables = [table for table in parser.tables]
        relevant_columns = [column for column in parser.columns]
        # print(f"relevant_tables:{relevant_tables}")
        # print(f"relevant_columns:{relevant_columns}")

        # align
        relevant_tables = [
            refer.align_table(table_name) for table_name in relevant_tables
        ]
        relevant_tables = [
            table_name for table_name in relevant_tables if table_name is not None
        ]

        result = {}
        for table_name in relevant_tables:
            result[table_name] = list()
        # print(f"result: {result}")
        for column_name in relevant_columns:
            # print(f"column_name:{column_name}")
            pos = column_name.find(".")
            if pos == -1:
                for table_name in result.keys():
                    align_column_name = refer.align_column(table_name, column_name)
                    if align_column_name is not None:
                        # print(f"align_column_name:{align_column_name}")
                        result[table_name].append(align_column_name)
            else:
                table_name = column_name[:pos]
                column_name_without_table_name = column_name[pos + 1 :]
                align_table_name = refer.align_table(table_name)
                if align_table_name is not None:
                    align_column_name = refer.align_column(
                        align_table_name, column_name_without_table_name
                    )
                    if align_column_name is not None:
                        # print(f"align_column_name:{align_column_name}")
                        result[align_table_name].append(align_column_name)

        # print(f"result: {result}")
        return result

    def run(self, db_name: str, question: str = None, evidence: str = None):
        keywords = self.keywords_extract(question, evidence)
        embeddings = self.rag_handler.embedding_queries(keywords)

        self.rag_schema = DatabaseSchema(
            self.rag_handler.get_schema(embeddings, db_name), None
        )
        # print(self.rag_schema)
        # print("---")
        # print(self.all_schema)
        self.rag_schema.merge(self.all_schema)
        self.build_graph_schema()
        
        sqlv1 = self.generate_sql(self.rag_schema)
        filter_schema = self.build_database_schema_by_sql(sql=sql)
        filter_schema.merge(self.rag_schema)
        filter_schema.merge(self.all_schema)
        
        

    def build_graph_schema(self):
        table_names = self.rag_schema.table_names()
        # print(self.rag_schema)
        # print("table_name_before:",table_names)
        table_names_after_one_hop = self.graph.one_hop(table_names)
        self.graph_schema = self.get_table_schema(table_names_after_one_hop)

    def generate_sql(self, schema: DatabaseSchema, question: str, evidence: str) -> str:
        instruction = SQL_INSTRUCTION_PATTERN.format(
            schema=str(schema),
            question=question,
            evidence=evidence,
            error_sql="",
            error_message="",
        )

        messages = [
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        
        response = self.openai_client.chat.completions.create(
            model=self.model_name,  # Model name to use
            messages=messages,  # Chat history
            temperature=0,  # Temperature for text generation
            stream=False,  # Stream response
        )
        sql = response.choices[0].message.content
        print(sql)
        return sql

    def keywords_extract(self, question: str, evidence: str):
        instruction = KEYWORDS_EXTRACT_PATTERN.format(
            question=question, evidence=evidence
        )
        messages = [
            # {"role": "system", "content": COMMON_SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        response = self.openai_client.chat.completions.create(
            model=self.model_name,  # Model name to use
            messages=messages,  # Chat history
            temperature=0,  # Temperature for text generation
            stream=False,  # Stream response
            # extra_body={
            #     # "repetition_penalty": 1,
            #     "add_generation_prompt": True,
            #     "stop_token_ids": self.stop_token_ids,
            # },
        )
        result = response.choices[0].message.content
        print(result)
        try:
            words = result.strip().split(",")
            words = [word.strip() for word in words]
        except ValueError as e:
            print(f"转换出错: {e}")
        return words




openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8001/v1"

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


# print(sol.rag_schema.statistics())
# print(sol.graph_schema.statistics())
# print(sol.all_schema.statistics())

# sql = "SELECT movie_title FROM movies WHERE movie_release_year = 1945 ORDER BY movie_popularity DESC LIMIT 1"


with open("/home/data2/luzhan/projects/bird_bench/dev/dev.json", "r") as input_file:
    dataset = json.load(input_file)
last_database = dict()

rag_eval = PrecisionRecallAverageCalculator()
graph_eval = PrecisionRecallAverageCalculator()
all_eval = PrecisionRecallAverageCalculator()

rag_schema_size = SchemaSizeAverageCalculator()
graph_schema_size = SchemaSizeAverageCalculator()
all_schema_size = SchemaSizeAverageCalculator()
refer_schema_size = SchemaSizeAverageCalculator()

i = 0
for case in dataset[0:]:
    print(i)
    print(case)
    db_name = case["db_id"]
    question = case["question"]
    evidence = case["evidence"]
    sql = case["SQL"]
    new_database = {
        "type": "sqlite",
        "url": f"/home/data2/luzhan/projects/bird_bench/dev/dev_databases/{db_name}/{db_name}.sqlite",
    }

    if last_database != new_database:
        sol = Text2SQLSolution(
            database=new_database,
            openai_client=client,
        )
    last_database = new_database
    try:
        sol.run(db_name=db_name, question=question, evidence=evidence)
        refer_schema = sol.build_database_schema_by_sql(sql=sql)
    except Exception as e:
        print(e)
        logger.error(f"error case id: {i}")
        i += 1
        continue

    # print(sol.rag_schema)
    # print("---")
    # print(sol.all_schema)
    # print("---")
    # print(refer_schema)
    # print("---")
    rag_eval.add_value(sol.rag_schema.eval_precision_and_recall(refer_schema))
    graph_eval.add_value(sol.graph_schema.eval_precision_and_recall(refer_schema))
    all_eval.add_value(sol.all_schema.eval_precision_and_recall(refer_schema))

    rag_schema_size.add_value(sol.rag_schema.schema_size())
    graph_schema_size.add_value(sol.graph_schema.schema_size())
    all_schema_size.add_value(sol.all_schema.schema_size())
    refer_schema_size.add_value(refer_schema.schema_size())
    # print(sql)
    # print(refer_schema)
    # print("----")
    # print(sol.rag_schema)
    i += 1

print(rag_eval.calculate_average())
print(graph_eval.calculate_average())
print(all_eval.calculate_average())

print(rag_schema_size.calculate_average())
print(graph_schema_size.calculate_average())
print(all_schema_size.calculate_average())
print(refer_schema_size.calculate_average())
