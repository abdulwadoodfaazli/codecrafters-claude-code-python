import argparse
import os
import sys
import json

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

read_tool_schema = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read and return the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read",
                }
            },
            "required": ["file_path"],
        },
    },
}


def read_tool(file):
    if not os.path.exists(file):
        raise RuntimeError(f"File {file} does not exist")
    with open(file, "r") as f:
        return f.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[read_tool_schema],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    tool_calls = (
        chat.model_dump()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("tool_calls", [])
    )
    for call in tool_calls:
        name = call.get("function", {}).get("name")
        args = json.loads(call.get("function", {}).get("arguments", "{}"))
        if name == "Read":
            tool_res = read_tool(args.get("file_path"))
            print(tool_res)

    if len(tool_calls) == 0:
        print(chat.choices[0].message.content)


if __name__ == "__main__":
    main()
