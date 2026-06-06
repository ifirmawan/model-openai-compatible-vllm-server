"""OpenAI-compatible client for the Modal vLLM server.

Usage:
    python openai_compatible/client.py
    python openai_compatible/client.py --chat
    python openai_compatible/client.py --prompt "What is 2+2?" --no-stream
"""
import argparse

import modal
from openai import OpenAI


class Colors:
    GREEN = "\033[0;32m"
    RED = "\033[0;31m"
    BLUE = "\033[0;34m"
    BOLD = "\033[1m"
    END = "\033[0m"


def get_completion(client: OpenAI, model_id: str, messages: list, args: argparse.Namespace):
    kwargs = {
        "model": model_id,
        "messages": messages,
        "frequency_penalty": args.frequency_penalty,
        "max_tokens": args.max_tokens,
        "n": args.n,
        "presence_penalty": args.presence_penalty,
        "seed": args.seed,
        "stop": args.stop,
        "stream": args.stream,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(Colors.RED, f"Error: {e}", Colors.END, sep="")
        return None


def build_client(args: argparse.Namespace) -> tuple[OpenAI, str]:
    """Return (client, model_id)."""
    client = OpenAI(api_key=args.api_key)
    workspace = args.workspace or modal.config._profile
    environment = args.environment or modal.config.config["environment"]
    prefix = workspace + (f"-{environment}" if environment else "")
    client.base_url = f"https://{prefix}--{args.app_name}-{args.function_name}.modal.run/v1"

    if args.model:
        model_id = args.model
        print(Colors.BOLD + f"🧠 Using model: {model_id}" + Colors.END)
    else:
        print(Colors.BOLD + f"🔎 Fetching available models from {client.base_url}..." + Colors.END)
        model_id = client.models.list().data[0].id
        print(Colors.BOLD + f"🧠 Using: {model_id}" + Colors.END)

    return client, model_id


def chat_loop(client: OpenAI, model_id: str, args: argparse.Namespace):
    messages = [{"role": "system", "content": args.system_prompt}]
    print(Colors.GREEN + Colors.BOLD + "\nChat mode. Type 'bye' to exit." + Colors.END)
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "bye":
            break
        if len(messages) > 10:
            messages = messages[:1] + messages[-9:]
        messages.append({"role": "user", "content": user_input})
        response = get_completion(client, model_id, messages, args)
        if not response:
            continue
        if args.stream:
            print(Colors.BLUE + "\n🤖: ", end="")
            assistant_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    print(content, end="")
                    assistant_message += content
            print(Colors.END)
        else:
            assistant_message = response.choices[0].message.content
            print(Colors.BLUE + f"\n🤖: {assistant_message}" + Colors.END)
        messages.append({"role": "assistant", "content": assistant_message})


def single_prompt(client: OpenAI, model_id: str, args: argparse.Namespace):
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
    print(Colors.GREEN + f"\nYou: {args.prompt}" + Colors.END)
    response = get_completion(client, model_id, messages, args)
    if not response:
        return
    if args.stream:
        print(Colors.BLUE + "\n🤖:", end="")
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        print(Colors.END)
    else:
        for i, choice in enumerate(response.choices):
            print(Colors.BLUE + f"\n🤖 Choice {i + 1}: {choice.message.content}" + Colors.END)


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible client for Modal vLLM server")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--workspace", type=str, default=None)
    parser.add_argument("--environment", type=str, default=None)
    parser.add_argument("--app-name", type=str, default="example-vllm-inference")
    parser.add_argument("--function-name", type=str, default="serve")
    parser.add_argument("--api-key", type=str, default="super-secret-key")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--stop", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default="Compose a limerick about baboons and raccoons.")
    parser.add_argument("--system-prompt", type=str, default="You are a poetic assistant with creative flair.")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--chat", action="store_true", help="Interactive chat mode")
    args = parser.parse_args()

    client, model_id = build_client(args)
    print(Colors.BOLD + "🧠 System prompt: " + args.system_prompt + Colors.END)

    if args.chat:
        chat_loop(client, model_id, args)
    else:
        single_prompt(client, model_id, args)


if __name__ == "__main__":
    main()
