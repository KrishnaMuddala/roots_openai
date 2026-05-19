from core.openai_client import Openai
from mcp_client import MCPClient
from core.tools import ToolManager

MessageParam = dict


class Chat:
    def __init__(self, llm_service: Openai, clients: dict[str, MCPClient]):
        self.llm_service: Openai = llm_service
        self.clients: dict[str, MCPClient] = clients
        self.messages: list[MessageParam] = []

    async def _process_query(self, query: str):
        self.messages.append({"role": "user", "content": query})

    async def run(
        self,
        query: str,
        stream: bool = False,
        on_event=None,
    ) -> str:
        final_text_response = ""

        await self._process_query(query)

        while True:
            if stream and on_event:
                response = await self.llm_service.chat_stream(
                    messages=self.messages,
                    tools=await ToolManager.get_all_tools(self.clients),
                    on_event=on_event,
                )
            else:
                response = await self.llm_service.chat(
                    messages=self.messages,
                    tools=await ToolManager.get_all_tools(self.clients),
                )

            self.llm_service.add_assistant_message(self.messages, response)

            if response.stop_reason == "tool_use":
                if not stream:
                    print(self.llm_service.text_from_message(response))
                tool_result_parts = await ToolManager.execute_tool_requests(
                    self.clients, response
                )
                self.llm_service.add_user_message(
                    self.messages, tool_result_parts
                )
            else:
                final_text_response = self.llm_service.text_from_message(response)
                break

        return final_text_response
