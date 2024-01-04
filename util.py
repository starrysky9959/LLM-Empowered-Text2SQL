
import sqlite3


def get_sqlite_schema_str(db_dataset_path:str, db_id: str):
    db_path = db_dataset_path+"/{db_id}/{db_id}.sqlite".format(db_id=db_id)
    conn = sqlite3.connect(db_path)

    # 创建游标对象
    cursor = conn.cursor()

    # 执行查询获取表的模式信息
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")

    # 获取查询结果
    results = cursor.fetchall()

    # # 打印表的模式信息
    # # schema = "```sql\n"
    schema=""
    for result in results:
        schema += result[0].lower() + ";\n"
    # schema+="```\n"
    # 关闭连接
    conn.close()

    return schema

def get_sqlite_schema_dict(db_path: str) -> dict:
    table_names_original, table_dot_column_names_original, column_names_original = (
        [],
        [],
        [],
    )
    conn = sqlite3.connect(db_path)

    # 创建游标对象
    cursor = conn.cursor()

    # 执行查询获取表的模式信息
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    # 打印表名和列名
    for table in tables:
        table_name = table[0]
        table_names_original.append(table_name)
        # print("表名:", table_name)
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        # print("列名:")
        for column in columns:
            column_name = column[1]
            column_names_original.append(column_name)
            table_dot_column_names_original.append(
                "{0}.{1}".format(table_name, column_name)
            )
            # print(column_name)
    # 关闭游标和数据库连接
    cursor.close()
    conn.close()

    return {
        "table_names_original": table_names_original,
        "table_dot_column_names_original": table_dot_column_names_original,
        "column_names_original": column_names_original,
    }