import json
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.history import InMemoryHistory
from core.cli_chat import CliChat
from pyboxen import boxen


class CliApp:
    def __init__(self, agent: CliChat):
        self.agent = agent
        self.history = InMemoryHistory()
        self.session = PromptSession(
            history=self.history,
            style=Style.from_dict(
                {
                    "prompt": "#aaaaaa",
                    "completion-menu.completion": "bg:#222222 #ffffff",
                    "completion-menu.completion.current": "bg:#444444 #ffffff",
                }
            ),
            complete_while_typing=True,
            complete_in_thread=True,
        )

    async def initialize(self):
        pass

    async def run(self):
        while True:
            try:
                user_input = await self.session.prompt_async("> ")
                if not user_input.strip():
                    continue

                print()

                # OpenAI streaming chunks arrive as ChatCompletionChunk objects.
                # Each chunk has: chunk.choices[0].delta
                #   delta.content      -> str fragment (text)
                #   delta.tool_calls   -> list of ChoiceDeltaToolCall
                #     .index           -> int (which tool call slot)
                #     .id              -> str or None (only on first chunk for that slot)
                #     .function.name   -> str or None (only on first chunk)
                #     .function.arguments -> str fragment
                # chunk.choices[0].finish_reason -> "stop" | "tool_calls" | None

                tool_calls: dict[int, dict] = {}  # index -> {name, args}

                async def handle_event(chunk):
                    if not chunk.choices:
                        return

                    delta = chunk.choices[0].delta
                    finish_reason = chunk.choices[0].finish_reason

                    # ── Text delta ────────────────────────────────────────────
                    if delta.content:
                        print(delta.content, end="", flush=True)

                    # ── Tool call deltas ──────────────────────────────────────
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            idx = tc.index
                            if idx not in tool_calls:
                                tool_calls[idx] = {"name": "", "args": ""}
                                print()  # newline before first tool chunk

                            if tc.function and tc.function.name:
                                tool_calls[idx]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                tool_calls[idx]["args"] += tc.function.arguments

                    # ── Tool call finished (finish_reason signals completion) ─
                    if finish_reason == "tool_calls":
                        for idx, tc in tool_calls.items():
                            try:
                                parsed_args = json.loads(tc["args"])
                                formatted_args = json.dumps(parsed_args, indent=2)
                                tool_content = f"🔧 {tc['name']}\n\nArguments:\n{formatted_args}"
                            except (json.JSONDecodeError, TypeError, ValueError):
                                tool_content = f"🔧 {tc['name']}\n\nArguments: {tc['args']}"

                            tool_box = boxen(
                                tool_content,
                                title="Tool Call",
                                style="rounded",
                                color="blue",
                                padding=0,
                            )
                            print(tool_box)
                        tool_calls.clear()

                await self.agent.run(user_input, stream=True, on_event=handle_event)

                print()  # newline after response

            except KeyboardInterrupt:
                break
