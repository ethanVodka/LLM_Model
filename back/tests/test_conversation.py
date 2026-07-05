import pytest

from mini_llm.conversation import DEFAULT_SYSTEM_PROMPT, format_chat_prompt


def test_formats_user_message_for_assistant_generation() -> None:
    prompt = format_chat_prompt(" こんにちは ")

    assert prompt == (
        f"<system>{DEFAULT_SYSTEM_PROMPT}<user>こんにちは<assistant>"
    )


def test_rejects_empty_user_message() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        format_chat_prompt("  ")
