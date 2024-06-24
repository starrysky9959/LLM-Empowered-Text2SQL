import argparse
import json
import multiprocessing

from openai import OpenAI
import sqlparse
import tool

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Chatbot Interface with Customizable Parameters"
)
parser.add_argument(
    "--model-url", type=str, default="http://localhost:8000/v1", help="Model URL"
)
parser.add_argument(
    "-m", "--model", type=str, default="deepseek", help="Model name for the chatbot"
)
parser.add_argument(
    "--temp", type=float, default=0, help="Temperature for text generation"
)
parser.add_argument(
    "--stop-token-ids", type=str, default="32021", help="Comma-separated stop token IDs"
)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8001)

# Parse the arguments
args = parser.parse_args()

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = args.model_url

# Create an OpenAI client to interact with the API server
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


def predict(database_info: dict, question, evidence, temperature):
    try:
        engine = tool.get_engine(database_info)
        table_names = tool.get_tables_in_database(engine)
    except Exception as e:
        print(e)
        return "无法连接到数据库", ""
    # print("database_type", database_type)
    # print("url", url)
    # print("question", question)
    # print("evidence", evidence)
    # print("temperature", temperature)

    instruction = tool.step_2(engine, None, question, evidence)
    # print(instruction)
    history_openai_format = [
        {"role": "system", "content": tool.STEP_2_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    # print(history_openai_format)
    # Create a chat completion request and send it to the API server
    response = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=False,  # Stream response
        extra_body={
            # "repetition_penalty": 1,
            "add_generation_prompt": True,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )

    # Read and return generated text from response stream
    raw_sql = response.choices[0].message.content
    relevant_tables = tool.get_relevant_tables(raw_sql)

    if not set(relevant_tables).issubset(set(table_names)):
        return (
            "您的提问似乎与该数据库中的表无关或者存在难以判别的歧义，暂时无法进行回答。请修改数据库设置或问题，补充额外信息。",
            "",
        )

    # for chunk in stream:
    #     raw_sql += chunk.choices[0].delta.content or ""
    # yield raw_sql
    # print("raw_sql", raw_sql)

    instruction = tool.step_2(engine, relevant_tables, question, evidence)
    # print(instruction)

    history_openai_format = [
        {"role": "system", "content": tool.STEP_2_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    # print(history_openai_format)
    # Create a chat completion request and send it to the API server
    response = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        temperature=temperature,  # Temperature for text generation
        stream=False,  # Stream response
        extra_body={
            "repetition_penalty": 1,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )

    # Read and return generated text from response stream
    answer = response.choices[0].message.content
    # for chunk in stream:
    #     answer += chunk.choices[0].delta.content or ""
    # yield answer, ""
    return answer


def worker(case):
    # print("case", case)
    success, sql_or_error_message = tool.text2sql(
        client,
        args.model,
        (
            [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
            if args.stop_token_ids
            else []
        ),
        case["database"],
        case["question"],
        case["evidence"],
        0,
    )
    print(sql_or_error_message)
    return sql_or_error_message


def single_test():
    db_info = {
        "type": "mysql",
        "username": "root",
        "password": "root",
        "host": "127.0.0.1",
        "port": "3306",
        "dbname": "text2sql",
    }
    sql = predict(
        db_info,
        "查询今天青岛市的中断用户数",
        "中断用户数=不同的网关SN数量。今天是2024年6月3日",
        0,
    )
    print(sql)


def batch_test():
    with open("./public_dataset/raw/spider_test.json", "r") as input_file:
        dataset = json.load(input_file)
    with multiprocessing.Pool(processes=30) as pool:
        # 使用map方法将任务分配给进程池
        results = pool.map(worker, dataset[:])
    with open("./public_dataset/bench/spider_test.txt", "w") as output_file:
        for sql in results:
            output_file.write(sql + "\n")


if __name__ == "__main__":
    # single_test()
    batch_test()
