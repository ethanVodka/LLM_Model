# LLM Core

小規模なDecoder-only Transformerと、Byte-level BPEトークナイザー、今後追加する学習・評価処理を置くPythonパッケージです。

## トークナイザー

リポジトリルートから実行します。

```powershell
uv run --project back --extra cpu mini-llm-tokenizer train
uv run --project back --extra cpu mini-llm-tokenizer encode "こんにちは"
```

学習済みファイルは `artifacts/tokenizer/tiny.json` に生成され、Git管理対象外です。

## 学習データ

```powershell
uv run --project back --extra cpu mini-llm-data prepare
uv run --project back --extra cpu mini-llm-data inspect
```

`prepare` は文書単位で分割した `train.npy` と `validation.npy` を作成します。各行は入力64トークンと、1トークン先の正解を作れる65要素です。

## 事前学習

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu
```

`configs/training/tiny.yaml` に従い、AdamW、Cross Entropy、勾配クリッピングを使っ20 stepだけ学習します。チェックポイントにはモデル、Optimizer、設定、最終指標を保存します。

学習を合計40 stepまで再開する例:

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu `
  --resume artifacts/checkpoints/tiny/latest.pt --max-steps 40
```

新形式のチェックポイントはモデルとOptimizerに加え、PyTorchとバッチ抽選の乱数状態も保存します。`--max-steps` は追加step数ではなく、再開後の合計step数です。

## 生成

```powershell
uv run --project back --extra cpu mini-llm-generate "Pythonで" --device cpu
uv run --project back --extra cpu mini-llm-generate "こんにちは" --temperature 0
```

`temperature=0` はgreedy生成、正の値はtop-k samplingで使う確率分布の鋭さを制御します。

会話デモでは `prepare-sft` がsystem／user部分を損失から除外し、assistant回答だけを学習対象にします。

```powershell
uv run --project back --extra cpu mini-llm-data prepare-sft `
  --config configs/data/chat_demo.yaml `
  --model-config configs/model/chat_demo.yaml `
  --tokenizer artifacts/tokenizer/chat_demo.json `
  --corpus data/processed/chat_demo/corpus.jsonl `
  --output-dir artifacts/datasets/chat_demo
```

## API

```powershell
uv run --project back --extra cpu uvicorn mini_llm.api:app --reload --port 8000
```

- `GET /api/health` — モデル読込状態を確認
- `POST /api/generate` — プロンプトと生成設定から文字列を生成

起動時にチェックポイントとトークナイザーを1回だけ読み込みます。パスは `.env` の `CHECKPOINT_PATH` と `TOKENIZER_PATH` で変更できます。

Qwen3-1.7B + LoRAバックエンドは、CUDAと4-bit依存を分離したGPU環境から起動します。

```powershell
$env:UV_PROJECT_ENVIRONMENT = "back/.venv-gpu"
uv sync --project back --frozen --extra gpu --extra qlora
back\.venv-gpu\Scripts\python.exe back/scripts/serve_qwen.py
```

Qwenサービスも同じ `/api/health` と `/api/generate` 契約を使用します。基盤モデルとadapterはGit管理外です。

## 評価

```powershell
uv run --project back --extra cpu mini-llm-evaluate --device cpu
```

validation lossとPerplexityを計算し、`data/samples/evaluation_prompts.jsonl` の固定プロンプトをgreedy生成します。指標、生成結果、入力成果物のハッシュは `artifacts/evaluations/tiny/report.json` へ保存されます。

## v1データ準備

```powershell
uv run --project back --extra cpu mini-llm-corpus
```

`configs/data/corpus_v1.yaml` のmanifestと入力JSONLの出典・ライセンスが一致することを確認し、NFKC正規化、改行統一、完全一致重複除去、文字数フィルタ、主要な秘密情報パターン検査を行います。出力は `data/processed/v1/corpus.jsonl`、監査レポートは `artifacts/data/v1/report.json` です。

固定した日本語Wikipedia記事の本文抜粋、プロジェクトのコード、role付きQAを準備する場合:

```powershell
uv run --project back --extra cpu mini-llm-wikipedia
uv run --project back --extra cpu mini-llm-project-code
uv run --project back --extra cpu mini-llm-qa
```

WikimediaのAPI方針に従うUser-Agentを使い、本文は1記事ずつ間隔を空けて取得します。各レコードにrevision ID、帰属用恒久URL、`CC-BY-SA-4.0` を保存します。QAは自動生成のため、SFT前に品質レビューが必要です。
