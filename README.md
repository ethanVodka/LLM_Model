# Learning LLM

日本語会話とPython／TypeScriptコードを題材に、LLMのデータ準備、事前学習、SFT、評価、API公開までを学ぶプロジェクトです。

本リポジトリには目的の異なる2つのモデルがあります。

| トラック          | 用途                                                    | 実行環境    |
| ----------------- | ------------------------------------------------------- | ----------- |
| `MiniDecoderLM`   | Transformer、トークナイザー、学習処理を実装から理解する | CPU／Docker |
| Qwen3-1.7B + LoRA | 未知の質問を含む実用的な日本語会話を試す                | NVIDIA GPU  |

## 現在できること

- Decoder-only TransformerとByte-level BPEの学習
- 出典・ライセンスを検証したコーパス作成
- 次トークン事前学習とassistant回答のみを対象にしたSFT
- チェックポイント再開、損失、Perplexity、固定プロンプト評価
- Qwen3-1.7Bの4-bit QLoRA学習
- FastAPIとReactによるローカルチャット画面

## クイックスタート：Qwenチャット画面

学習済み基盤モデルとadapterがローカルにある場合、PowerShellを2つ開きます。

### 1. GPU APIを起動

```powershell
cd C:\WorkSpace\LLM_Model
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
back\.venv-gpu\Scripts\python.exe back\scripts\serve_qwen.py
```

モデル読込には数秒かかります。API確認先は以下です。

- Health Check: `http://localhost:8000/api/health`
- OpenAPI UI: `http://localhost:8000/docs`

### 2. Reactを起動

```powershell
cd C:\WorkSpace\LLM_Model
npm.cmd --prefix front run dev
```

ブラウザで `http://localhost:5173` を開きます。

### 生成設定

| 設定        | 意味                               | 推奨初期値 |
| ----------- | ---------------------------------- | ---------- |
| 生成数      | 最大生成トークン数。文字数ではない | `80`       |
| Temperature | `0`は安定、値を上げると多様になる  | `0`        |
| Top-k       | Temperatureが正の場合の候補数      | `20`       |

## 必要環境

- Windows PowerShell
- uv
- Python 3.12.13
- Node.js／npm
- Docker Desktop（CPU再現性確認時）
- NVIDIA GPU（Qwen利用時。本プロジェクトではRTX 3060 Ti 8GBで確認）

## Python環境

CPU環境とGPU環境は混在させず、別々の仮想環境へ構築します。

### CPU環境

```powershell
uv python install 3.12.13
uv sync --project back --frozen --extra cpu
```

生成先は `back/.venv/` です。MiniDecoder、テスト、データ準備ではこちらを使用します。

### GPU／QLoRA環境

```powershell
$env:UV_PROJECT_ENVIRONMENT = "back/.venv-gpu"
uv sync --project back --frozen --extra gpu --extra qlora
```

生成先は `back/.venv-gpu/` です。Qwenの推論とQLoRA学習ではこちらを使用します。

## Qwen3-1.7B QLoRA

Apache 2.0の [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B) を固定revisionで使用します。設定は `configs/training/qwen3_1_7b_qlora.yaml` にあります。

### 1. 会話データを準備

```powershell
back\.venv\Scripts\python.exe -m mini_llm.conversation_prepare_cli

uv run --project back --extra cpu mini-llm-corpus `
  --config configs/data/corpus_sft_v1.yaml `
  --output data/processed/sft_v1/corpus.jsonl `
  --report artifacts/data/sft_v1/report.json

back\.venv\Scripts\python.exe -m mini_llm.qwen_data_cli
```

Qwen用messages JSONLは `data/processed/qwen_sft_v1/conversations.jsonl` に生成されます。

### 2. 基盤モデルを初回取得

```powershell
back\.venv-gpu\Scripts\python.exe back\scripts\generate_qwen.py `
  "プログラミング初心者は何から始めればよいですか？"
```

モデルは `artifacts/models/huggingface/` に保存されます。取得後はオフラインで再利用できます。

### 3. QLoRAを学習

```powershell
back\.venv-gpu\Scripts\python.exe back\scripts\train_qwen_qlora.py
```

既定設定は4-bit NF4、LoRA rank 8、40 stepです。adapterと評価指標は `artifacts/adapters/qwen3_1_7b_ja/` に保存されます。

### 4. adapterを直接確認

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

back\.venv-gpu\Scripts\python.exe back\scripts\generate_qwen.py `
  "PythonとTypeScriptの違いを教えてください" `
  --adapter artifacts/adapters/qwen3_1_7b_ja/step_0040 `
  --temperature 0
```

## MiniDecoderLM学習パイプライン

このトラックはLLM内部の仕組みを理解するための小規模実装です。実用品質を目的としません。

### 1. トークナイザー

```powershell
uv run --project back --extra cpu mini-llm-tokenizer train
uv run --project back --extra cpu mini-llm-tokenizer encode "こんにちは、Pythonで学習します。"
```

### 2. データ準備

```powershell
uv run --project back --extra cpu mini-llm-data prepare
uv run --project back --extra cpu mini-llm-data inspect
```

### 3. 事前学習

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu
```

20 stepから合計40 stepまで再開する例:

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu `
  --resume artifacts/checkpoints/tiny/latest.pt `
  --max-steps 40
```

### 4. 生成と評価

```powershell
uv run --project back --extra cpu mini-llm-generate "Pythonで" `
  --temperature 0 --device cpu
uv run --project back --extra cpu mini-llm-evaluate --device cpu
```

## 外部データの準備

日本語Wikipedia、本リポジトリのPython／TypeScript、Wikipedia由来QAを準備します。

```powershell
uv run --project back --extra cpu mini-llm-wikipedia
uv run --project back --extra cpu mini-llm-project-code
uv run --project back --extra cpu mini-llm-qa

uv run --project back --extra cpu mini-llm-corpus `
  --config configs/data/corpus_knowledge_v1.yaml `
  --output data/processed/knowledge_v1/corpus.jsonl `
  --report artifacts/data/knowledge_v1/report.json
```

Wikipediaレコードにはrevision ID、恒久URL、CC BY-SA 4.0を保存します。自動生成QAは品質保証済みではないため、SFTへ投入する前に内容を確認してください。

## 開発コマンド

### バックエンド

```powershell
uv run --project back --extra cpu pytest
uv run --project back --extra cpu ruff check back
uv run --project back --extra cpu mypy `
  --config-file back/pyproject.toml back/src
```

### フロントエンド

```powershell
npm.cmd --prefix front run test
npm.cmd --prefix front run lint
npm.cmd --prefix front run format:check
npm.cmd --prefix front run build
```

依存関係を変更した場合だけ `uv lock --project back` を実行し、`back/uv.lock` もコミットします。

## Docker

DockerはCPU版MiniDecoderの再現性確認に使用します。Qwen GPU環境は対象外です。

```powershell
Copy-Item .env.example .env
docker compose build
docker compose run --rm trainer pytest
docker compose run --rm trainer mini-llm-info
```

## ディレクトリ構成

```text
back/src/mini_llm/   モデル、データ処理、学習、API
back/scripts/        Qwen QLoRAの学習・推論・API起動
back/tests/          pytestテスト
front/src/           React／TypeScript UI
configs/             モデル、データ、学習設定
data/samples/        Git管理する小規模データ
data/raw/            取得データ（Git管理外）
data/processed/      加工済みデータ（Git管理外）
artifacts/           モデル、adapter、評価結果（Git管理外）
docker/              CPU用Docker環境
```

## 制約と注意事項

- Qwen adapterは未知質問へ回答できますが、事実性を保証しません。
- 最新情報へは接続していないため、ニュース、天気、価格などは別の検索機能が必要です。
- 学習データには出典、取得日、ライセンス、加工内容を記録してください。
- 個人情報、秘密鍵、認証情報、利用許諾のないコードを学習へ混入させないでください。
- `data/raw/`、`data/processed/`、`artifacts/` はGit管理外です。
