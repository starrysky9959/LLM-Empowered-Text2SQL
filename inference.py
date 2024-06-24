import argparse

import gradio as gr
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


def predict(database_type, url, question, evidence, temperature):
    success, sql_or_error_message = tool.text2sql(
        client,
        args.model,
        (
            [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
            if args.stop_token_ids
            else []
        ),
        {
            "type": "mysql",
            "username": "",
            "password": "",
            "host": "",
            "port": 3306,
            "dbname": "",
        },
        question,
        evidence,
        temperature,
    )
    if success:
        return sql_or_error_message, sqlparse.format(
            sql_or_error_message, reindent=True, keyword_case="upper"
        )
    else:
        return sql_or_error_message, ""


examples = [
    # [
    #     "sqlite",
    #     "/home/data2/luzhan/projects/spider/database/concert_singer/concert_singer.sqlite",
    #     "How many singers do we have?",
    #     "",
    # ],
    [
        "sqlite",
        "/home/data2/luzhan/projects/police_data/police.db",
        "找出曾在2024-04-11日入住过酒店, 户籍地属于'宁波市'并且有'贩卖毒品案'前科的在逃人员的姓名和身份证",
        "可以通过模糊匹配判断户籍地的归属",
    ],
    [
        "sqlite",
        "/home/data2/luzhan/projects/police_data/police.db",
        "最近30天所有有交通违章记录的驾驶员, 并按照他们的累计记分降序排列.",
        "",
    ],
    [
        "sqlite",
        "/home/data2/luzhan/projects/police_data/police.db",
        "找出2024-04-01到2024-04-30期间有航班, 铁路或者旅馆住宿任一记录的在逃人员",
        "",
    ],
    [
        "sqlite",
        "/home/data2/luzhan/projects/police_data/police.db",
        "分组统计不同品牌的小型汽车的数量, 按照降序排列, 并且要求车辆所有人的年龄在20到30岁之间",
        "今年是2024年",
    ],
    [
        "sqlite",
        "/home/data2/luzhan/projects/police_data/police.db",
        "查询2024年乘坐过D3132次列车的未婚男性乘客",
        "",
    ],
]
# custom_css = """
# .submit_button {
#     background-color: #FFA500;
#     color: orange;
# }
# """
with gr.Blocks() as demo:
    gr.Markdown("# BI empowered by LLM")
    with gr.Column():
        inputs = [
            gr.Dropdown(
                label="数据库类型/Database Type",
                choices=["sqlite"],
                value="sqlite",
            ),
            gr.Textbox(
                label="数据库连接/Connection URL",
                placeholder="请输入连接目标数据库的URL/Please input the connection url for the target database",
                value="/home/data2/luzhan/projects/spider/database/concert_singer/concert_singer.sqlite",
            ),
            gr.Textbox(
                label="问题/Question",
                placeholder="请输入您想咨询的问题/Please input the question you want to ask",
                value="How many singers do we have?",
            ),
            gr.Textbox(
                label="额外的补充信息/Extra infomation",
                placeholder="请输入额外的信息，帮助模型更好地理解问题和数据库/Please input the extra information to help the model better understand your question and database.",
            ),
        ]

        with gr.Row():
            inputs.append(
                gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0,
                    label="温度/Temperature",
                    info="温度越高, 模型输出的随机性越强/Higher temperature make the model more random",
                )
            )
            run_button = gr.Button(
                "运行/Run", variant="primary", elem_classes=["submit_button"]
            )
            clear_button = gr.ClearButton(inputs, value="清空/Clear")
        with gr.Row():
            outputs = [
                gr.Code(label="SQL", language="sql"),
                gr.Code(label="Formatted SQL/格式化的SQL", language="sql"),
            ]
        gr.Examples(
            examples=examples,
            inputs=inputs,
            outputs=outputs,
            fn=predict,
            # cache_examples=True,
        )
    run_button.click(fn=predict, inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port)

# # Create and launch a chat interface with Gradio
# gr.ChatInterface(predict).queue().launch(
#     server_name=args.host, server_port=args.port, share=True
# )
