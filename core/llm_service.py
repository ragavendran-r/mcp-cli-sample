from typing import Protocol, Any, runtime_checkable


@runtime_checkable
class LLMMessage(Protocol):
    """Common interface for Claude's Message and Gemini's FakeMessage."""

    stop_reason: str
    role: str
    content: list


@runtime_checkable
class LLMService(Protocol):
    def add_user_message(self, messages: list, message: Any) -> None: ...
    def add_assistant_message(self, messages: list, message: Any) -> None: ...
    def text_from_message(self, message: LLMMessage) -> str: ...
    def chat(
        self,
        messages: list,
        system: str | None = None,
        temperature: float = 1.0,
        stop_sequences: list = [],
        tools: list | None = None,
        thinking: bool = False,
        thinking_budget: int = 1024,
    ) -> LLMMessage: ...
