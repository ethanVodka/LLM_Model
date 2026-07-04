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

すべてのファイルでUTF-8、LF、末尾改行を使用します。公開APIや複雑な処理には、実装内容ではなく目的・制約・副作用が分かる短いコメントを付けてください。

### Python

Python 3.10以上を対象とし、4スペースと型ヒントを使用します。Ruff（行長100文字）とmypy strictを通してください。モジュール、関数、変数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE` とします。パス操作には `pathlib.Path`、データ構造には型付きdataclassを優先します。`Any`、可変なデフォルト引数、広すぎる `except Exception` は理由なく使用しないでください。テンソルを扱う関数では、入力・出力形状、dtype、device、値域の制約を型、検証処理、またはdocstringで明示します。

### React / TypeScript

TypeScriptはstrictモードと2スペースを使用し、ESLintとPrettierをCIで検証します。`any` は避け、外部入力は `unknown` として検証してから利用します。コンポーネント、型、interfaceは `PascalCase`、関数、変数、hooksは `camelCase`、定数は `UPPER_SNAKE_CASE` とします。コンポーネントファイルは `ChatPanel.tsx`、hooksは `useChatSession.ts`、テストは `ChatPanel.test.tsx` の形式にします。

関数コンポーネントと名前付きexportを基本とし、props型を明示してください。状態と副作用は必要な場所へ局所化し、派生値をstateへ重複保存しません。APIアクセスや業務ロジックを表示コンポーネントへ直接埋め込まず、hooksまたはfeature単位のserviceへ分離します。アクセシブルなHTML要素を優先し、クリック可能な `div` のような代替実装は避けてください。

## テストと実験

テスト名は `test_<期待する振る舞い>` とし、変更には対応するpytestテストを追加します。最初は形状、因果マスク、設定値検証、再現性を優先します。実験では設定、乱数シード、データ版、依存バージョン、評価結果を記録してください。モデルサイズではなく、固定評価データに対する損失、Perplexity、生成品質で比較します。

## データとセキュリティ

データセットごとに出典、取得日、ライセンス、加工内容を記録します。個人情報、秘密鍵、認証情報、利用許諾のないコードを学習へ混入させないでください。秘密情報は `.env` に置き、`.env.example` には実値を書きません。

## MCPの使い分け

Serena MCPはシンボル検索、参照追跡、構造的な編集に使用します。Code Index MCPは横断検索、ファイル要約、大規模化後の永続インデックスに使用し、単純な検索では `rg` を優先してください。Chrome DevTools MCPはReact画面の動作、コンソール、ネットワーク、性能の検証に限定します。Chromeは分離された一時プロファイルで実行し、個人アカウントへのログインや機密情報の入力を避けてください。

## コミットとレビュー

コミットは `feat(model): add causal attention` のようなConventional Commitsを推奨します。PRには目的、設計判断、検証コマンド、実験結果を記載し、モデルやデータの変更では計算量とライセンスへの影響も明記してください。
