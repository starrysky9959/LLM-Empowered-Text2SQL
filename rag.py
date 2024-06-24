# %%
import json
import os


def get_database_info(base_path: str):
    output = []
    for database_name in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, database_name)):
            database_path = os.path.join(
                base_path, database_name, f"{database_name}.sqlite"
            )
            output.append((database_name, database_path))
    return output


database_info = get_database_info(
    "/home/data2/luzhan/projects/bird_bench/dev/dev_databases"
) + get_database_info("/home/data2/luzhan/projects/bird_bench/train/train_databases")
database_info

# %%
from pymilvus import DataType, MilvusClient, utility, connections, db, model
import tool


# from pymilvus.model.hybrid import BGEM3EmbeddingFunction
# import pandas as pd

# bge_m3_ef = BGEM3EmbeddingFunction(
#     model_name="BAAI/bge-m3",  # Specify the model name
#     device="cpu",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
#     use_fp16=False,  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
# )
embedding_fn = model.dense.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",  # Specify the model name
    device="cuda:0",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    trust_remote_code=True,
    local_files_only=True,
)

COLLECTION_NAME_FOR_TABLE_SCHEMA = "{db_name}_table_schema"
COLLECTION_NAME_FOR_COLUMN_SCHEMA = "{db_name}_column_schema"
COLLECTION_NAME_FOR_COLUMN_VALUE = "{db_name}_column_value"


# %%
def build_table_schema(client: MilvusClient, engine, db_name: str):
    collection_name = COLLECTION_NAME_FOR_TABLE_SCHEMA.format(db_name=db_name)
    client.drop_collection(collection_name=collection_name)
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="table_name", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="table_name_vector", datatype=DataType.FLOAT_VECTOR, dim=384
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="table_name_vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="table_name_vector_index",
        params={"nlist": 128},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    res = client.list_indexes(collection_name=collection_name)

    table_names = tool.get_tables_in_database(engine)
    data = []
    for table_name in table_names:
        table_name_vector = embedding_fn.encode_documents(table_name)[0]
        # print(table_name_vector)
        data.append(
            {
                "table_name": table_name,
                "table_name_vector": table_name_vector,
            }
        )
    res = client.insert(collection_name=collection_name, data=data)
    print(res)


# %%
import pandas as pd


def build_column_schema(client: MilvusClient, df: pd.DataFrame, db_name: str):
    collection_name = COLLECTION_NAME_FOR_COLUMN_SCHEMA.format(db_name=db_name)
    client.drop_collection(collection_name=collection_name)

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="table_name", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="column_name", datatype=DataType.VARCHAR, max_length=128
    )
    schema.add_field(
        field_name="column_name_vector", datatype=DataType.FLOAT_VECTOR, dim=384
    )
    schema.add_field(
        field_name="column_description", datatype=DataType.VARCHAR, max_length=4096
    )
    schema.add_field(
        field_name="column_description_vector", datatype=DataType.FLOAT_VECTOR, dim=384
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="column_name_vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="column_name_vector_index",
        params={"nlist": 128},
    )
    index_params.add_index(
        field_name="column_description_vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="column_description_vector_index",
        params={"nlist": 128},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)
    res = client.list_indexes(collection_name=collection_name)

    data = []
    for index, row in df.iterrows():
        table_name = row["table_name"]
        column_name = row["original_column_name"]
        column_description = row["description"]
        column_name_vector, column_description_vector = embedding_fn.encode_documents(
            [column_name, column_description]
        )

        data.append(
            {
                "table_name": table_name,
                "column_name": column_name,
                "column_name_vector": column_name_vector,
                "column_description": column_description,
                "column_description_vector": column_description_vector,
            }
        )
    res = client.insert(collection_name=collection_name, data=data)
    print(res)


# %%
from sqlalchemy import select, text


def build_column_value(client: MilvusClient, engine, db_name: str):
    print(f"db_name:{db_name}")
    collection_name = COLLECTION_NAME_FOR_COLUMN_VALUE.format(db_name=db_name)
    client.drop_collection(collection_name=collection_name)

    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="table_name", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(
        field_name="column_name", datatype=DataType.VARCHAR, max_length=128
    )
    schema.add_field(
        field_name="column_value", datatype=DataType.VARCHAR, max_length=65535
    )
    schema.add_field(
        field_name="column_value_vector", datatype=DataType.FLOAT_VECTOR, dim=384
    )
    client.create_collection(
        collection_name=collection_name,
        schema=schema,
    )

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index(
        field_name="column_value_vector",
        metric_type="COSINE",
        index_type="IVF_FLAT",
        index_name="column_value_vector_index",
        params={"nlist": 128},
    )
    client.create_index(collection_name=collection_name, index_params=index_params)

    for table_name in tool.get_tables_in_database(engine):
        string_columns_names = tool.get_string_columns_in_database(engine, table_name)
        print(string_columns_names)
        for column_name in string_columns_names:
            stmt = (
                select(text(f"`{column_name}`"))
                .distinct()
                .select_from(text(f"`{table_name}`"))
            )

            with engine.connect() as connection:
                result = connection.execute(stmt)
                unique_values = result.fetchall()
            if len(unique_values) == 0:
                continue
            unique_values = [str(value[0]) for value in unique_values]
            data = []
            i = 0
            for column_value, column_value_vector in list(
                zip(unique_values, embedding_fn.encode_documents(unique_values))
            ):
                i += 1
                data.append(
                    {
                        "table_name": table_name,
                        "column_name": column_name,
                        "column_value": column_value,
                        "column_value_vector": column_value_vector,
                    }
                )
                if i > 100:
                    i = 0
                    res = client.insert(collection_name=collection_name, data=data)
                    data = []


# %%
import pandas as pd

df_train = pd.read_csv("./public_dataset/rag/bird_train.csv")
df_dev = pd.read_csv("./public_dataset/rag/bird_dev.csv")
df = pd.concat([df_train, df_dev], ignore_index=True)
df = df.apply(lambda x: x.fillna(""))

# prepare milvus client and database
conn = connections.connect(host="127.0.0.1", port=19530)
db_name_milvus = "bird_bench"
if db_name_milvus not in db.list_database():
    database = db.create_database(db_name_milvus)
db.using_database(db_name_milvus)
client = MilvusClient(uri="http://localhost:19530", db_name=db_name_milvus)


ignore = True
for db_name, db_path in database_info:
    if db_name == "codebase_comments":
        ignore = False
    if ignore:
        continue
    db_info = {
        "type": "sqlite",
        "url": db_path,
    }
    db_engine = tool.get_engine(db_info)

    # build_table_schema(client, db_engine, db_name)
    # build_column_schema(client, df[df["database_name"] == db_name], db_name)
    # build_column_value(client, db_engine, db_name)
    res = client.load_collection(
        collection_name="shipping_column_schema",
    )
    res = client.get_load_state(collection_name="shipping_column_schema")

    print(res)
    res = client.search(
        collection_name="shipping_column_schema",
        data=embedding_fn.encode_documents(
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
        ),
        limit=3,
        output_fields=["table_name", "column_name"],
        search_params={"metric_type": "COSINE", "params": {}},
        anns_field="column_name_vector",
    )
    res = json.dumps(res, indent=4)
    print(res)
    break
