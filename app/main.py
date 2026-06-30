import argparse
import os
import json
import subprocess

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

tools_schema = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to execute",
                    }
                },
            },
        },
    },
]


def read_tool(file):
    if not os.path.exists(file):
        raise RuntimeError(f"File {file} does not exist")
    with open(file, "r") as f:
        return f.read()


def write_tool(file, content):
    with open(file, "w") as f:
        f.write(content)
    return content


def bash_tool(command):
    cmd = subprocess.run(command.split(), capture_output=True, text=True)
    return cmd.stdout + cmd.stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools_schema,
        )
        messages.append(chat.choices[0].message.model_dump())

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        tool_calls = (
            chat.model_dump()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("tool_calls", [])
        )
        if tool_calls:
            for call in tool_calls:
                name = call.get("function", {}).get("name")
                args = json.loads(call.get("function", {}).get("arguments", "{}"))
                if name == "Read":
                    tool_res = read_tool(args.get("file_path"))
                elif name == "Write":
                    tool_res = write_tool(args.get("file_path"), args.get("content"))
                elif name == "Bash":
                    tool_res = bash_tool(args.get("command"))
                tool_res_msg = {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": tool_res,
                }
                messages.append(tool_res_msg)
        else:
            print(chat.choices[0].message.content)
            break


if __name__ == "__main__":
    main()
