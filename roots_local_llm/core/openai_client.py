import json
from typing import Literal
from openai import OpenAI


class _FakeMessage:
    """Wraps an OpenAI-style response to look like an Anthropic Message
    so the rest of the codebase does not need to change."""

    def __init__(self, openai_response):
        self._response = openai_response
        choice = openai_response.choices[0]
        self.content = []
        msg = choice.message

        if msg.content:
            self.content.append(_TextBlock(msg.content))

        if msg.tool_calls:
            for tc in msg.tool_calls:
                self.content.append(_ToolUseBlock(tc))

        finish = choice.finish_reason
        if finish == "tool_calls":
            self.stop_reason = "tool_use"
        elif finish == "stop":
            self.stop_reason = "end_turn"
        else:
            self.stop_reason = finish or "end_turn"


class _TextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Mimics anthropic.types.ToolUseBlock."""

    def __init__(self, tool_call):
        self.type = "tool_use"
        self.id = tool_call.id
        self.name = tool_call.function.name
        try:
            self.input = json.loads(tool_call.function.arguments)
        except Exception:
            self.input = {}


class _FakeFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _FakeToolCallResult:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeChoiceMessage:
    def __init__(self, content, tool_calls):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, content, tool_calls, finish_reason):
        self.message = _FakeChoiceMessage(content, tool_calls)
        self.finish_reason = finish_reason


class _FakeOpenAIResponse:
    def __init__(self, content, tool_calls, finish_reason):
        self.choices = [_FakeChoice(content, tool_calls, finish_reason)]


class Openai:
    def __init__(self, model: str):
        import os
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key="ollama")

    def add_user_message(self, messages: list, message):
        messages.append({
            "role": "user",
            "content": self._extract_content(message),
        })

    def add_assistant_message(self, messages: list, message):
        if isinstance(message, _FakeMessage):
            text_parts = []
            tool_calls = []
            for block in message.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    })
            entry = {"role": "assistant", "content": "\n".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)
        else:
            messages.append({"role": "assistant", "content": str(message)})

    def text_from_message(self, message) -> str:
        return "\n".join(
            block.text for block in message.content if block.type == "text"
        )

    async def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> _FakeMessage:
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append(self._normalise_message(msg))

        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": 8000,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = self._convert_tools(tools)
            params["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**params)
        return _FakeMessage(response)

    async def chat_stream(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=None,
        tools=None,
        thinking=False,
        thinking_budget=1024,
        on_event=None,
    ) -> _FakeMessage:
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for msg in messages:
            openai_messages.append(self._normalise_message(msg))

        params = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": temperature,
            "max_tokens": 8000,
        }
        if stop_sequences:
            params["stop"] = stop_sequences
        if tools:
            params["tools"] = self._convert_tools(tools)
            params["tool_choice"] = "auto"

        collected_content = ""
        collected_tool_calls: dict[int, dict] = {}

        stream = self.client.chat.completions.create(**params, stream=True)
        for chunk in stream:
            if on_event:
                await on_event(chunk)

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            if delta.content:
                collected_content += delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in collected_tool_calls:
                        collected_tool_calls[idx] = {
                            "id": "",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        collected_tool_calls[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        collected_tool_calls[idx]["function"]["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        collected_tool_calls[idx]["function"]["arguments"] += tc.function.arguments

        tool_calls_list = [
            _FakeToolCallResult(
                tc["id"],
                tc["function"]["name"],
                tc["function"]["arguments"],
            )
            for tc in collected_tool_calls.values()
        ] or None

        finish_reason = "tool_calls" if tool_calls_list else "stop"
        fake_response = _FakeOpenAIResponse(
            content=collected_content or None,
            tool_calls=tool_calls_list,
            finish_reason=finish_reason,
        )
        return _FakeMessage(fake_response)

    @staticmethod
    def _extract_content(message):
        if isinstance(message, _FakeMessage):
            return "\n".join(
                b.text for b in message.content if b.type == "text"
            )
        if isinstance(message, list):
            return message
        return message

    @staticmethod
    def _normalise_message(msg: dict) -> dict:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, list):
            tool_results = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if tool_results:
                tr = tool_results[0]
                return {
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": str(tr.get("content", "")),
                }
            text = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            return {"role": role, "content": text}

        return {"role": role, "content": str(content)}

    @staticmethod
    def _convert_tools(tools: list) -> list:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in tools
        ]
