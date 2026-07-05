# LLM Core

小規模なDecoder-only Transformerと、Byte-level BPEトークナイザー、今後追加する学習・評価処理を置くPythonパッケージです。

## トークナイザー

リポジトリルートから実行します。

```powershell
uv run --project back --extra cpu mini-llm-tokenizer train
uv run --project back --extra cpu mini-llm-tokenizer encode "こんにちは"
```

学習済みファイルは `artifacts/tokenizer/tiny.json` に生成され、Git管理対象外です。
