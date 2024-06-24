from typing import List, Set, Tuple
from schema import ColumnSchema, TableSchema
from collections import defaultdict
from pymilvus import MilvusClient, utility, connections, db, model

embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # Specify the model name
    device="cuda:0",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    trust_remote_code=True,
    local_files_only=True,
)

COLLECTION_NAME_FOR_TABLE_SCHEMA = "{db_name}_table_schema"
COLLECTION_NAME_FOR_COLUMN_SCHEMA = "{db_name}_column_schema"
COLLECTION_NAME_FOR_COLUMN_VALUE = "{db_name}_column_value"


class RAGHandler:
    def __init__(self):
        conn = connections.connect(host="127.0.0.1", port=19530)
        db_name_milvus = "bird_bench"
        if db_name_milvus not in db.list_database():
            database = db.create_database(db_name_milvus)
        db.using_database(db_name_milvus)
        self.client = MilvusClient(uri="http://localhost:19530", db_name=db_name_milvus)

    def get_schema(self, query_embeddings: List[List[float]], db_name: str):
        # print(query_embeddings)
        schema = defaultdict(TableSchema)
        self.search_column_schema(
            COLLECTION_NAME_FOR_COLUMN_SCHEMA.format(db_name=db_name),
            query_embeddings,
            schema,
        )

        self.search_column_value(
            COLLECTION_NAME_FOR_COLUMN_VALUE.format(db_name=db_name),
            query_embeddings,
            schema,
        )
        return dict(schema)

    def embedding_queries(self):
        return embedding_fn.encode_documents(
            [
                "full name",
                "driver",
                "delivered",
                "most shipments",
                "least populated city",
                "Min(population)",
                "first_name",
                "last_name",
                "driver_id",
                "Max(Count(ship_id))",
            ]
        )

    def search_column_schema(
        self, collection_name: str, query_emdbeddings: List[List[float]], schema
    ):
        res = self.client.load_collection(collection_name=collection_name)
        res = self.client.get_load_state(collection_name=collection_name)

        res = self.client.search(
            collection_name=collection_name,
            data=query_emdbeddings,
            limit=3,
            output_fields=["table_name", "column_name", "column_description"],
            search_params={"metric_type": "COSINE", "params": {}},
            anns_field="column_name_vector",
        )
        # parse_table_column_from_result(res)
        print(res)
        for result_set in res:
            for single_result in result_set:
                table_name = single_result["entity"]["table_name"]

                schema[table_name].add_column(
                    ColumnSchema(
                        single_result["entity"]["column_name"],
                        single_result["entity"]["column_description"],
                        None,
                    )
                )

    def search_column_value(
        self,
        collection_name: str,
        query_emdbeddings: List[List[float]],
        schema,
    ):

        res = self.client.load_collection(collection_name=collection_name)
        res = self.client.get_load_state(collection_name=collection_name)

        res = self.client.search(
            collection_name=collection_name,
            data=query_emdbeddings,
            limit=3,
            output_fields=["table_name", "column_name", "column_value"],
            search_params={"metric_type": "COSINE", "params": {}},
            anns_field="column_value_vector",
        )
        print(res)
        for result_set in res:
            for single_result in result_set:
                table_name = single_result["entity"]["table_name"]

                schema[table_name].add_column(
                    ColumnSchema(
                        single_result["entity"]["column_name"],
                        None,
                        single_result["entity"]["column_value"],
                    )
                )
