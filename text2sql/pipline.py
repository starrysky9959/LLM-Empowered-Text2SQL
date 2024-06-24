# keyword extract
# rag->tableschema
# graph_build, one_hop
# tableschema
# select table,column
# generate sql
# revision
import sys
from typing import Dict, Set

sys.path.append("/home/data2/luzhan/projects/LLM-Empowered-Text2SQL/text2sql")

from fk_graph import ForeignKeyGraph
from schema import TableSchema, ColumnSchema
from rag import RAGHandler
import sqlalchemy

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


class Text2SQLSolution:
    schema_by_rag: Dict[str, TableSchema]
    schema_by_graph: Dict[str, TableSchema]
    schema_by_all: Dict[str, TableSchema]

    def __init__(
        self,
        database: dict,
        # milvus_connection: dict,
    ):
        self.database = database
        self.engine = self.get_engine()
        self.graph = ForeignKeyGraph(self.engine)
        self.rag_handler = RAGHandler()
        self.schema_by_rag = dict()
        self.schema_by_graph = dict()
        self.schema_by_all = self.get_table_schema(None)

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

    def get_table_schema(self, table_names: Set[str]) -> Dict[str, TableSchema]:
        schema = {}
        inspector = sqlalchemy.inspect(self.engine)

        if table_names is None:
            table_names = inspector.get_table_names()

        for table_name in table_names:
            table_schema = TableSchema(table_name)

            columns = inspector.get_columns(table_name)

            for column in columns:
                table_schema.add_column(ColumnSchema(column["name"], column["type"]))

            primary_keys = inspector.get_pk_constraint(table_name)

            table_schema.pk_names = primary_keys["constrained_columns"]

            fk_list = []
            foreign_keys = inspector.get_foreign_keys(table_name)
            for foreign_key in foreign_keys:

                fk_list.append(
                    (
                        foreign_key["constrained_columns"],
                        foreign_key["referred_table"],
                        foreign_key["referred_columns"],
                    )
                )
                table_schema.fk_names = fk_list
            schema[table_name] = table_schema
        return schema

    def run(self, db_name: str):
        self.schema_by_rag = self.rag_handler.get_schema(
            self.rag_handler.embedding_queries(), db_name
        )
        self.merge_schema()
        self.build_graph_schema()

    def build_graph_schema(self):
        table_names = self.schema_by_rag.keys()
        table_names_after_one_hop = self.graph.one_hop(table_names)
        self.schema_by_graph = self.get_table_schema(table_names_after_one_hop)

    def merge_schema(self):
        table_names = self.schema_by_rag.keys()
        for table_name in table_names:
            print(f"merge table_name: {table_name}")
            self.schema_by_rag[table_name].merge(self.schema_by_all[table_name])

    def __str__(self):
        return self.schema_by_rag


sol = Text2SQLSolution(
    database={
        "type": "sqlite",
        "url": "/home/data2/luzhan/projects/bird_bench/train/train_databases/address/address.sqlite",
    }
)
# for key, value in sol.schema_by_all.items():
#     print(f"Key: {key}, Value: {value}")
# print(sol.schema_by_all)
sol.run(db_name="address")
print(sol.schema_by_rag)
print(sol.schema_by_graph)

