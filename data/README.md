# Data

- `raw/`: 取得元を保持した未加工データ（Git管理外）
- `processed/`: 正規化、重複除去、分割後のデータ（Git管理外）
- `samples/`: データ準備パイプラインを検証する小さな合成コーパス

データセットごとに出典URL、取得日、ライセンス、加工内容を記録してください。個人情報、認証情報、利用許諾のないコードを学習に使用してはいけません。

入力は1文書1行のJSONLとし、`id`、`text`、`source`、`license`、`language` をすべて必須とします。`id` はコーパス全体で重複させないでください。

`samples/evaluation_prompts.jsonl` は学習に混ぜず、モデル間の生成比較に使う固定入力です。各行に `id`、`prompt`、`category`、`language` を記録します。

`samples/conversation_corpus.jsonl` はrole tokenと会話形式を検証するプロジェクト独自サンプルです。外部データを追加するときは `configs/data/corpus_v1.yaml` にパス、出典、ライセンス、取得日を追加します。

`raw/wikipedia_ja_v1.jsonl` は `mini-llm-wikipedia` で取得するGit管理外データです。各記事のrevision IDとCC BY-SA 4.0帰属URLを削除しないでください。

`raw/project_code_v1.jsonl` は `mini-llm-project-code` で本リポジトリのPython／TypeScriptを変換したデータです。対象globとリポジトリURLは `configs/data/project_code_v1.yaml` で固定します。

`processed/wikipedia_qa_v1.jsonl` は `mini-llm-qa` がWikipedia本文から自動生成するrole付きQAです。自動生成済みであることは品質保証を意味しないため、SFTへ使う前に事実性、質問との対応、回答長を人手または別の評価工程で確認してください。知識事前学習用の `corpus_knowledge_v1.yaml` と会話学習用の `corpus_sft_v1.yaml` は分離します。

`samples/chat_demo_corpus.jsonl` は簡単な会話経路を確認する人手作成データです。学習済みプロンプトには回答できますが、未知の質問への一般化や知識の正確性を保証するデータセットではありません。
