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
- `front/src/` — React UI。機能単位のコンポーネント、hooks、services、テスト

## 開発・検証コマンド

ホストではuv管理のPython 3.12.13とCPU版PyTorchを標準とし、Dockerで再現性を確認します。

- `uv sync --project back --frozen --extra cpu` — `back/.venv` をlockfileどおり同期
- `uv run --project back --extra cpu pytest` — ホスト環境で全テストを実行
- `uv run --project back --extra cpu ruff check back` — ホスト環境でlintを実行
- `uv run --project back --extra cpu mypy back/src` — ホスト環境で型検査を実行
- `npm --prefix front run dev` — Vite開発サーバーを起動
- `npm --prefix front run build` — 型検査後に本番用アセットを生成
- `npm --prefix front run test` — Vitestの全テストを実行
- `npm --prefix front run lint` — OxlintでReact／TypeScript規約を検証
- `npm --prefix front run format:check` — Prettierの整形差分を検証
- `Copy-Item .env.example .env` — ローカル設定を作成
- `docker compose build` — Python、PyTorch、開発依存を含むイメージを作成
- `docker compose run --rm trainer pytest` — 全テストを実行
- `docker compose run --rm trainer ruff check back` — lintを実行
- `docker compose run --rm trainer mini-llm-info` — モデル規模と利用デバイスを確認

GPU環境だけ `uv sync --project back --frozen --extra gpu` を使います。`cpu` と `gpu` を同時に指定してはいけません。依存関係を変更した場合は `uv lock --project back` を実行し、`back/uv.lock` も更新してください。

## コーディング規約

すべてのファイルでUTF-8、LF、末尾改行を使用します。公開APIや複雑な処理には、実装内容ではなく目的・制約・副作用が分かる短いコメントを付けてください。

### Python

Python 3.12を対象とし、4スペースと型ヒントを使用します。Ruff（行長100文字）とmypy strictを通してください。モジュール、関数、変数は `snake_case`、クラスは `PascalCase`、定数は `UPPER_SNAKE_CASE` とします。パス操作には `pathlib.Path`、データ構造には型付きdataclassを優先します。`Any`、可変なデフォルト引数、広すぎる `except Exception` は理由なく使用しないでください。テンソルを扱う関数では、入力・出力形状、dtype、device、値域の制約を型、検証処理、またはdocstringで明示します。

### React / TypeScript

TypeScriptはstrictモードと2スペースを使用し、Oxlint、`tsc`、PrettierをCIで検証します。`any` は禁止し、Oxlintの `typescript/no-explicit-any` を無効化してはいけません。外部入力は `unknown` として受け取り、型ガードまたはスキーマで検証してから利用します。型アサーションやnon-null assertion（`!`）で検査を迂回せず、必要な場合は理由をコメントしてください。

コンポーネント、型、interfaceは `PascalCase`、関数、変数、hooksは `camelCase`、定数は `UPPER_SNAKE_CASE` とします。コンポーネントファイルは `ChatPanel.tsx`、hooksは `useChatSession.ts`、テストは `ChatPanel.test.tsx` の形式にします。propsは明示したreadonlyな型で表し、イベント型にはReact提供の `ChangeEventHandler` などを使用します。

関数コンポーネントと名前付きexportを基本とし、コンポーネント関数を直接呼ばずJSXで使用します。Hooksはコンポーネントまたはcustom hookのトップレベルでのみ呼び出します。状態と副作用は必要な場所へ局所化し、計算できる派生値をstateへ重複保存しません。`useEffect` は外部システムとの同期に限定します。

APIアクセスや業務ロジックを表示コンポーネントへ直接埋め込まず、hooksまたはfeature単位のserviceへ分離します。アクセシブルなHTML要素を優先し、クリック可能な `div` のような代替実装は避けてください。ユーザーから見える振る舞いの変更にはTesting Libraryによるテストを追加し、実装詳細ではなくrole、label、表示内容を検証します。

## テストと実験

テスト名は `test_<期待する振る舞い>` とし、変更には対応するpytestテストを追加します。最初は形状、因果マスク、設定値検証、再現性を優先します。実験では設定、乱数シード、データ版、依存バージョン、評価結果を記録してください。モデルサイズではなく、固定評価データに対する損失、Perplexity、生成品質で比較します。

## データとセキュリティ

データセットごとに出典、取得日、ライセンス、加工内容を記録します。個人情報、秘密鍵、認証情報、利用許諾のないコードを学習へ混入させないでください。秘密情報は `.env` に置き、`.env.example` には実値を書きません。

## MCPの使い分け

Serena MCPはシンボル検索、参照追跡、構造的な編集に使用します。Code Index MCPは横断検索、ファイル要約、大規模化後の永続インデックスに使用し、単純な検索では `rg` を優先してください。Chrome DevTools MCPはReact画面の動作、コンソール、ネットワーク、性能の検証に限定します。Chromeは分離された一時プロファイルで実行し、個人アカウントへのログインや機密情報の入力を避けてください。

## コミットとレビュー

コミットは `feat(model): add causal attention` のようなConventional Commitsを推奨します。PRには目的、設計判断、検証コマンド、実験結果を記載し、モデルやデータの変更では計算量とライセンスへの影響も明記してください。
