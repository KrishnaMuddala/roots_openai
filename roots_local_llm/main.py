import asyncio
import sys
import os
from dotenv import load_dotenv
from contextlib import AsyncExitStack

from mcp_client import MCPClient
from core.openai_client import Openai
from core.cli_chat import CliChat
from core.cli import CliApp

load_dotenv()

# Ollama / local LLM config
llm_model = os.getenv("LLM_MODEL", "qwen2.5:7b")
local_llm_base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")

assert llm_model, "Error: LLM_MODEL cannot be empty. Update .env"


async def main():
    llm_service = Openai(model=llm_model)

    root_paths = sys.argv[1:]
    if not root_paths:
        print("Usage: uv run main.py <root1> [root2] ...")
        print("Example: uv run main.py /path/to/videos /another/path")
        sys.exit(1)

    clients = {}

    async with AsyncExitStack() as stack:
        doc_client = await stack.enter_async_context(
            MCPClient(
                command="uv", args=["run", "mcp_server.py"], roots=root_paths
            )
        )
        clients["doc_client"] = doc_client

        chat = CliChat(
            doc_client=doc_client,
            clients=clients,
            llm_service=llm_service,
        )

        cli = CliApp(chat)
        await cli.initialize()
        await cli.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    asyncio.run(main())
