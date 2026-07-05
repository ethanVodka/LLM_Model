# Learning LLM

日本語の簡単な会話と短いPython／TypeScriptコード補完を題材に、LLMのデータ準備、トークナイズ、事前学習、評価、SFTを段階的に学ぶプロジェクトです。最初はCPUで動く小型Transformerを使い、パイプラインを検証してからGPUとモデル規模を拡張します。

## ホスト環境（推奨）

Pythonと依存関係はuvで管理します。リポジトリルートから次を実行すると、`back/.venv` にPython 3.12.13とCPU版PyTorchが同期されます。

```powershell
uv python install 3.12.13
uv sync --project back --frozen --extra cpu
back\.venv\Scripts\Activate.ps1
pytest back/tests
```

GPU版（CUDA 13.0）を利用する環境では、CPU版と同時に指定せず次を実行します。

```powershell
uv sync --project back --frozen --extra gpu
```

依存関係を意図的に更新した場合だけ `uv lock --project back` を実行し、`back/uv.lock` をコミットしてください。

## Docker環境

DockerはCPU版PyTorchを使う再現性確認環境です。Docker Desktopを起動して実行します。

```powershell
Copy-Item .env.example .env
docker compose build
docker compose run --rm trainer pytest
docker compose run --rm trainer mini-llm-info
```

## フロントエンド

React、TypeScript、Viteを使用します。

```powershell
npm --prefix front install
npm --prefix front run dev
```

品質チェックは次のコマンドで実行します。

```powershell
npm --prefix front run build
npm --prefix front run test
npm --prefix front run lint
npm --prefix front run format:check
```

## トークナイザー

小規模な合成コーパスから、モデルと同じ1,024語彙のByte-level BPEを学習します。

```powershell
uv run --project back --extra cpu mini-llm-tokenizer train
uv run --project back --extra cpu mini-llm-tokenizer encode "こんにちは、Pythonで学習します。"
```

学習結果はGit管理外の `artifacts/tokenizer/tiny.json` に保存されます。サンプルコーパスはパイプライン検証専用であり、実用モデルの学習データではありません。

## 学習データ準備

コーパスを文書単位でtrainとvalidationへ分割し、次トークン予測用の固定長配列を作ります。

```powershell
uv run --project back --extra cpu mini-llm-data prepare
uv run --project back --extra cpu mini-llm-data inspect
```

生成先はGit管理外の `artifacts/datasets/tiny/` です。分割の再現に必要なシード、文書ID、入力ファイルのSHA-256も `metadata.json` へ記録します。

## 事前学習スモークテスト

データ準備後、CPU用の短い設定で次トークン予測を学習します。

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu
```

学習中はtrain・validation lossと勾配normを表示し、最終状態をGit管理外の `artifacts/checkpoints/tiny/latest.pt` へ保存します。この20 step設定は学習処理の検証用であり、生成品質を得るための学習量ではありません。

保存済みのstep 20から合計40 stepまで続ける場合は次を実行します。

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu `
  --resume artifacts/checkpoints/tiny/latest.pt --max-steps 40
```

`latest.pt` に加えて `step_000040.pt` のような定期チェックポイントを保存します。損失、Perplexity、勾配norm、累積トークン数は `artifacts/experiments/tiny/metrics.jsonl` へ追記されます。

## 文字列生成

学習済みチェックポイントから、temperatureとtop-kを使って1トークンずつ生成します。

```powershell
uv run --project back --extra cpu mini-llm-generate "Pythonで" --device cpu
```

決定的なgreedy生成は `--temperature 0`、生成長の変更は `--max-new-tokens 100` を指定します。20 stepのスモーク学習ではパイプラインの接続だけを確認し、文章の意味や正確さは評価対象にしません。

## APIと生成画面

学習済みチェックポイントを用意し、2つのPowerShellでAPIとViteを起動します。

```powershell
uv run --project back --extra cpu uvicorn mini_llm.api:app --reload --port 8000
npm.cmd --prefix front run dev
```

`http://localhost:5173` の生成フォームはVite proxy経由で `POST /api/generate` を呼び出します。APIの動作確認は `http://localhost:8000/api/health`、OpenAPI UIは `http://localhost:8000/docs` で行えます。Dockerでは `docker compose up api` を使用します。

## 固定評価

同じvalidation配列とプロンプト集で、損失、Perplexity、生成結果を記録します。

```powershell
uv run --project back --extra cpu mini-llm-evaluate --device cpu
```

レポートはGit管理外の `artifacts/evaluations/tiny/report.json` へ保存されます。チェックポイント、トークナイザー、validationデータ、プロンプト集のSHA-256も記録するため、異なる実験を同一条件で比較できます。

## v1コーパスと会話トークナイザー

出典、ライセンス、取得日を `configs/data/corpus_v1.yaml` で検証し、正規化、重複除去、秘密情報パターン検査を実行します。

```powershell
uv run --project back --extra cpu mini-llm-corpus
uv run --project back --extra cpu mini-llm-tokenizer train `
  --config configs/tokenizer/v1.yaml --model-config configs/model/v1.yaml `
  --corpus data/processed/v1/corpus.jsonl --output artifacts/tokenizer/v1.json
uv run --project back --extra cpu mini-llm-data prepare `
  --config configs/data/v1.yaml --model-config configs/model/v1.yaml `
  --tokenizer artifacts/tokenizer/v1.json --corpus data/processed/v1/corpus.jsonl `
  --output-dir artifacts/datasets/v1
```

v1トークナイザーは `<system>`、`<user>`、`<assistant>` を単一トークンとして扱います。現在の38文書はパイプライン検証用であり、会話品質を得るデータ量ではありません。

## 日本語知識・コード・会話データ

`configs/data/wikipedia_ja_v1.yaml` で固定した100記事から最大1,200文字ずつ、Wikimedia APIへ負荷をかけない逐次リクエストで取得します。続けて本リポジトリのPython／TypeScriptと、Wikipedia由来のrole付きQAを準備します。

```powershell
uv run --project back --extra cpu mini-llm-wikipedia
uv run --project back --extra cpu mini-llm-project-code
uv run --project back --extra cpu mini-llm-qa
uv run --project back --extra cpu mini-llm-corpus `
  --config configs/data/corpus_knowledge_v1.yaml `
  --output data/processed/knowledge_v1/corpus.jsonl `
  --report artifacts/data/knowledge_v1/report.json
uv run --project back --extra cpu mini-llm-corpus `
  --config configs/data/corpus_sft_v1.yaml `
  --output data/processed/sft_v1/corpus.jsonl `
  --report artifacts/data/sft_v1/report.json
```

各Wikipediaレコードに固定revision IDと恒久URLを保存し、CC BY-SA 4.0の帰属先を追跡します。QAは自動生成データであり、SFT投入前の品質レビューが必要です。取得データと生成コーパスはGit管理外です。再配布や公開学習に使う場合は、帰属と継承条件を別途確認してください。

## 現在の範囲

`configs/model/tiny.yaml` は実装確認用であり、実用品質のモデルではありません。学習データと生成物はGit管理外です。データセットを追加するときは、出典、ライセンス、利用条件を必ず記録してください。
