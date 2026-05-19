import json
from typing import Literal, List
from mcp.types import CallToolResult, TextContent
from mcp_client import MCPClient

# Plain dicts — no Anthropic dependency
ToolResultBlockParam = dict


class ToolManager:

    @classmethod
    async def get_all_tools(cls, clients: dict[str, MCPClient]) -> list[dict]:
        """Gets all tools from the provided clients."""
        tools = []
        for client in clients.values():
            tool_models = await client.list_tools()
            tools += [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema,
                }
                for t in tool_models
            ]
        return tools

    @classmethod
    async def _build_tool_index(
        cls, clients: dict[str, MCPClient]
    ) -> dict[str, MCPClient]:
        """Returns a mapping of tool_name -> client for fast lookup."""
        index: dict[str, MCPClient] = {}
        for client in clients.values():
            tools = await client.list_tools()
            for t in tools:
                index[t.name] = client
        return index

    @classmethod
    def _build_tool_result_part(
        cls,
        tool_use_id: str,
        text: str,
        status: Literal["success", "error"],
    ) -> ToolResultBlockParam:
        """Builds a tool result dict."""
        return {
            "tool_use_id": tool_use_id,
            "type": "tool_result",
            "content": text,
            "is_error": status == "error",
        }

    @classmethod
    async def execute_tool_requests(
        cls, clients: dict[str, MCPClient], message  # message is _FakeMessage
    ) -> List[ToolResultBlockParam]:
        """Executes all tool_use blocks in a message against the provided clients."""
        tool_requests = [
            block for block in message.content if block.type == "tool_use"
        ]
        if not tool_requests:
            return []

        tool_index = await cls._build_tool_index(clients)
        tool_result_blocks: list[ToolResultBlockParam] = []

        for tool_request in tool_requests:
            tool_use_id = tool_request.id
            tool_name = tool_request.name
            tool_input = tool_request.input  # already a dict from _ToolUseBlock

            client = tool_index.get(tool_name)
            if not client:
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_use_id, "Could not find that tool", "error"
                    )
                )
                continue

            try:
                tool_output: CallToolResult | None = await client.call_tool(
                    tool_name, tool_input
                )
                items = tool_output.content if tool_output else []
                content_list = [
                    item.text for item in items if isinstance(item, TextContent)
                ]
                status: Literal["success", "error"] = (
                    "error" if tool_output and tool_output.isError else "success"
                )
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_use_id, json.dumps(content_list), status
                    )
                )
            except Exception as e:
                error_message = f"Error executing tool '{tool_name}': {e}"
                print(error_message)
                tool_result_blocks.append(
                    cls._build_tool_result_part(
                        tool_use_id,
                        json.dumps({"error": error_message}),
                        "error",
                    )
                )

        return tool_result_blocks
