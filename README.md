# Learning LLM

日本語の簡単な会話と短いPython／TypeScriptコード補完を題材に、LLMのデータ準備、トークナイズ、事前学習、評価、SFTを段階的に学ぶプロジェクトです。最初はCPUで動く小型Transformerを使い、パイプラインを検証してからGPUとモデル規模を拡張します。

## クイックスタート

Docker Desktopを起動し、リポジトリルートで実行します。

```powershell
Copy-Item .env.example .env
docker compose build
docker compose run --rm trainer pytest
docker compose run --rm trainer mini-llm-info
```

ホストにPython 3.10以降がある場合は、Dockerを使わずに開発できます。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".\back[dev]"
pytest back/tests
```

## 現在の範囲

`configs/model/tiny.yaml` は実装確認用であり、実用品質のモデルではありません。学習データと生成物はGit管理外です。データセットを追加するときは、出典、ライセンス、利用条件を必ず記録してください。

