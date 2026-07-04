# Repository Guidelines

## 開発目的

本リポジトリでは、LLMの仕組みを実装しながら学びます。初期目標は、日本語の簡単な会話と短いPython／TypeScriptコード補完を扱う小規模なDecoder-only Transformerです。まずCPUでデータ準備から評価までのパイプラインを完成させ、その後GPU、SFT、より大きなモデルへ段階的に進みます。学習用モデルと、将来の実用コーディングエージェントは別の評価軸で扱ってください。

## プロジェクト構成

- `back/src/mini_llm/` — モデル、設定、今後追加する学習・推論コード
- `back/tests/` — pytestによる単体テスト
- `configs/model/` — モデル構成。`tiny.yaml` はCPUスモークテスト用
- `data/` — 学習データの説明。`raw/` と `processed/` の実データはGit管理外
- `artifacts/` — 評価結果など。チェックポイントはGit管理外
- `docker/`、`compose.yaml` — 再現可能な実行環境
- `front/` — 将来のエージェントUI用。学習基盤の初期段階では使用しない

## 開発・検証コマンド

Dockerを標準環境とします。

- `Copy-Item .env.example .env` — ローカル設定を作成
- `docker compose build` — Python、PyTorch、開発依存を含むイメージを作成
- `docker compose run --rm trainer pytest` — 全テストを実行
- `docker compose run --rm trainer ruff check back` — lintを実行
- `docker compose run --rm trainer mini-llm-info` — モデル規模と利用デバイスを確認

ホスト環境を使う場合は `python -m pip install -e ".\back[dev]"` でインストールします。

## コーディング規約

Python 3.10以上を対象とし、4スペース、UTF-8、型ヒントを使用します。Ruffの行長は100文字、mypyはstrict設定です。モジュールと関数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE` とします。モデルの挙動を暗黙化せず、テンソル形状と制約をコードまたはdocstringで明確にしてください。

## テストと実験

テスト名は `test_<期待する振る舞い>` とし、変更には対応するpytestテストを追加します。最初は形状、因果マスク、設定値検証、再現性を優先します。実験では設定、乱数シード、データ版、依存バージョン、評価結果を記録してください。モデルサイズではなく、固定評価データに対する損失、Perplexity、生成品質で比較します。

## データとセキュリティ

データセットごとに出典、取得日、ライセンス、加工内容を記録します。個人情報、秘密鍵、認証情報、利用許諾のないコードを学習へ混入させないでください。秘密情報は `.env` に置き、`.env.example` には実値を書きません。

## コミットとレビュー

コミットは `feat(model): add causal attention` のようなConventional Commitsを推奨します。PRには目的、設計判断、検証コマンド、実験結果を記載し、モデルやデータの変更では計算量とライセンスへの影響も明記してください。
