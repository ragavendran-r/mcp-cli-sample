import os
from typing import Sequence, cast
from google import genai
from google.genai import types
from google.genai.types import ToolListUnion
from core.llm_service import LLMMessage


class FakeContentBlock:
    """Mimics Anthropic's ContentBlock so existing code using .type and .text works."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeMessage:
    """Mimics Anthropic's Message so existing code works without changes."""

    def __init__(self, text: str, stop_reason: str = "end_turn"):
        self.content = [FakeContentBlock(text)]
        self.stop_reason = stop_reason
        self.role = "assistant"


class Gemini:
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = model

    def _extract_content(self, message) -> str:
        if isinstance(message, FakeMessage):
            return "\n".join(block.text for block in message.content if block.type == "text")
        if isinstance(message, str):
            return message
        # list of tool result dicts — serialize to string for Gemini
        if isinstance(message, list):
            return "\n".join(
                item.get("content", "") if isinstance(item, dict) else str(item) for item in message
            )
        return str(message)

    def add_user_message(self, messages: list, message) -> None:
        messages.append({"role": "user", "content": self._extract_content(message)})

    def add_assistant_message(self, messages: list, message) -> None:
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
        # Convert messages to Gemini format
        gemini_contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            # Handle Anthropic-style content blocks if present
            if isinstance(content, list):
                text = "\n".join(
                    block["text"] if isinstance(block, dict) else block.text
                    for block in content
                    if (isinstance(block, dict) and block.get("type") == "text")
                    or (not isinstance(block, dict) and getattr(block, "type", None) == "text")
                )
            else:
                text = content
            gemini_contents.append(types.Content(role=role, parts=[types.Part(text=text)]))

        # Map Anthropic-style tools to Gemini function declarations
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
