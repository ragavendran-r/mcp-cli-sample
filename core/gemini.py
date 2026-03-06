import os
import uuid
from typing import cast
from google import genai
from google.genai import types
from google.genai.types import ToolListUnion
from core.llm_service import LLMMessage


class FakeContentBlock:
    """Mimics Anthropic's text ContentBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeToolUseBlock:
    """Mimics Anthropic's tool_use ContentBlock."""

    def __init__(self, tool_use_id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = tool_use_id
        self.name = name
        self.input = input


class FakeMessage:
    """Mimics Anthropic's Message so existing code works without changes."""

    def __init__(
        self,
        content: list | None = None,
        text: str | None = None,
        stop_reason: str = "end_turn",
    ):
        if content is not None:
            self.content = content
        else:
            self.content = [FakeContentBlock(text or "")]
        self.stop_reason = stop_reason
        self.role = "assistant"


class Gemini:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def _extract_content(self, message) -> str:
        if isinstance(message, FakeMessage):
            return "\n".join(
                block.text for block in message.content if block.type == "text"  # type: ignore[union-attr]
            )
        if isinstance(message, str):
            return message
        if isinstance(message, list):
            return "\n".join(
                item.get("content", "") if isinstance(item, dict) else str(item) for item in message
            )
        return str(message)

    def add_user_message(self, messages: list, message) -> None:
        messages.append({"role": "user", "content": self._extract_content(message)})

    def add_assistant_message(self, messages: list, message) -> None:
        # Store raw FakeMessage for tool_use so we can reconstruct function_call parts
        if isinstance(message, FakeMessage) and message.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": None, "_fake_message": message})
        else:
            messages.append({"role": "assistant", "content": self._extract_content(message)})

    def text_from_message(self, message: LLMMessage) -> str:
        return "\n".join(
            [block.text for block in message.content if block.type == "text"]  # type: ignore[union-attr]
        )

    def chat(
        self,
        messages,
        system=None,
        temperature=1.0,
        stop_sequences=[],
        tools=None,
        thinking=False,
        thinking_budget=1024,
    ) -> FakeMessage:
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"

            # Reconstruct function_call parts for tool_use assistant messages
            fake_msg = msg.get("_fake_message")
            if fake_msg and isinstance(fake_msg, FakeMessage):
                parts = []
                for block in fake_msg.content:
                    if isinstance(block, FakeToolUseBlock):
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=block.name,
                                    args=block.input,
                                )
                            )
                        )
                gemini_contents.append(types.Content(role=role, parts=parts))
                continue

            content = msg["content"]
            if isinstance(content, list):
                # Tool result message
                if any(
                    isinstance(item, dict) and item.get("type") == "tool_result" for item in content
                ):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            parts.append(
                                types.Part(
                                    function_response=types.FunctionResponse(
                                        name=item.get("tool_use_id", "unknown"),
                                        response={"result": item.get("content", "")},
                                    )
                                )
                            )
                    gemini_contents.append(types.Content(role="user", parts=parts))
                else:
                    text = "\n".join(
                        block["text"] if isinstance(block, dict) else block.text  # type: ignore[union-attr]
                        for block in content
                        if (isinstance(block, dict) and block.get("type") == "text")
                        or (not isinstance(block, dict) and getattr(block, "type", None) == "text")
                    )
                    gemini_contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
            else:
                gemini_contents.append(types.Content(role=role, parts=[types.Part(text=content)]))

        gemini_tools = cast(ToolListUnion, self._convert_tools(tools)) if tools else None

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=8000,
            stop_sequences=stop_sequences if stop_sequences else None,
            system_instruction=system,
            tools=gemini_tools,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=gemini_contents,
            config=config,
        )

        # Check for function calls in response
        function_calls = []
        for candidate in response.candidates or []:
            for part in candidate.content.parts or []:
                if part.function_call:
                    function_calls.append(part.function_call)

        if function_calls:
            content_blocks: list = []
            for fc in function_calls:
                content_blocks.append(
                    FakeToolUseBlock(
                        tool_use_id=str(uuid.uuid4()),
                        name=fc.name,
                        input=dict(fc.args) if fc.args else {},
                    )
                )
            return FakeMessage(content=content_blocks, stop_reason="tool_use")

        return FakeMessage(text=response.text or "")

    def _convert_tools(self, tools: list) -> list[types.Tool]:
        """Convert Anthropic tool format to Gemini function declarations."""
        declarations = []
        for tool in tools:
            declarations.append(
                types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("input_schema", {}),
                )
            )
        return [types.Tool(function_declarations=declarations)]
