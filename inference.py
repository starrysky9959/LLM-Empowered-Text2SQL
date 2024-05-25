import argparse

import gradio as gr
from openai import OpenAI
import tool

# Argument parser setup
parser = argparse.ArgumentParser(
    description="Chatbot Interface with Customizable Parameters"
)
parser.add_argument(
    "--model-url", type=str, default="http://localhost:8000/v1", help="Model URL"
)
parser.add_argument(
    "-m", "--model", type=str, default="sql-lora", help="Model name for the chatbot"
)
parser.add_argument(
    "--temp", type=float, default=1, help="Temperature for text generation"
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


def predict(
    database_type,
    url,
    question,
    evidence,
):
    print("database_type", database_type)
    print("url", url)
    print("question", question)
    print("evidence", evidence)
    engine = tool.get_engine({"type": database_type, "url": url})
    instruction = tool.step_2(engine, None, question, evidence)
    print(instruction)
    # tool.get_relevant_tables()

    # Convert chat history to OpenAI format
    history_openai_format = [
        {"role": "system", "content": tool.STEP_2_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    print(history_openai_format)
    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        # temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
        extra_body={
            # "repetition_penalty": 1,
            "stop_token_ids": (
                [int(id.strip()) for id in args.stop_token_ids.split(",") if id.strip()]
                if args.stop_token_ids
                else []
            ),
        },
    )

    # Read and return generated text from response stream
    raw_sql = ""
    for chunk in stream:
        raw_sql += chunk.choices[0].delta.content or ""
        # yield raw_sql
    print("raw_sql", raw_sql)

    instruction = tool.step_2(
        engine, tool.get_relevant_tables(raw_sql), question, evidence
    )
    print(instruction)

    history_openai_format = [
        {"role": "system", "content": tool.STEP_2_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    print(history_openai_format)
    # Create a chat completion request and send it to the API server
    stream = client.chat.completions.create(
        model=args.model,  # Model name to use
        messages=history_openai_format,  # Chat history
        # temperature=args.temp,  # Temperature for text generation
        stream=True,  # Stream response
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
    answer = ""
    for chunk in stream:
        answer += chunk.choices[0].delta.content or ""
        yield answer


with gr.Blocks() as demo:
    gr.Markdown("# Text2SQL")
    with gr.Column():
        inputs = [
            gr.Dropdown(
                label="Database Type/数据库类型",
                choices=["sqlite"],
                value="sqlite",
            ),
            gr.Textbox(
                label="Connection URL/数据库连接",
                placeholder="Please input the url for the target database.",
                value="/home/data2/luzhan/projects/spider/database/concert_singer/concert_singer.sqlite",
            ),
            gr.Textbox(
                label="Question/问题",
                placeholder="Please input the question you want to ask.",
                value="How many singers do we have?",
            ),
            gr.Textbox(
                label="Extra infomation/额外的补充信息",
                placeholder="Please input the extra information to help the model better understand your question and database.",
            ),
        ]
        with gr.Row():
            run_button = gr.Button("Run")
            clear_button = gr.ClearButton(inputs)
        outputs = [gr.Code(label="SQL", language="sql")]

    run_button.click(fn=predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name=args.host, server_port=args.port)

# # Create and launch a chat interface with Gradio
# gr.ChatInterface(predict).queue().launch(
#     server_name=args.host, server_port=args.port, share=True
# )
