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

## 文字列生成

学習済みチェックポイントから、temperatureとtop-kを使って1トークンずつ生成します。

```powershell
uv run --project back --extra cpu mini-llm-generate "Pythonで" --device cpu
```

決定的なgreedy生成は `--temperature 0`、生成長の変更は `--max-new-tokens 100` を指定します。20 stepのスモーク学習ではパイプラインの接続だけを確認し、文章の意味や正確さは評価対象にしません。

## 現在の範囲

`configs/model/tiny.yaml` は実装確認用であり、実用品質のモデルではありません。学習データと生成物はGit管理外です。データセットを追加するときは、出典、ライセンス、利用条件を必ず記録してください。
