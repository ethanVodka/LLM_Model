"""会話モデルが学習時と推論時に共有するrole形式。"""

DEFAULT_SYSTEM_PROMPT = "あなたは学習用の小型AIです。日本語で短く正確に回答します。"


def format_chat_prompt(user_message: str) -> str:
    """ユーザー入力を、SFTコーパスと同じassistant直前までの形式へ変換する。"""

    normalized = user_message.strip()
    if not normalized:
        raise ValueError("user_message must not be empty")
    return f"<system>{DEFAULT_SYSTEM_PROMPT}<user>{normalized}<assistant>"
