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

## 現在の範囲

`configs/model/tiny.yaml` は実装確認用であり、実用品質のモデルではありません。学習データと生成物はGit管理外です。データセットを追加するときは、出典、ライセンス、利用条件を必ず記録してください。
