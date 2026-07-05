# Data

- `raw/`: 取得元を保持した未加工データ（Git管理外）
- `processed/`: 正規化、重複除去、分割後のデータ（Git管理外）
- `samples/`: データ準備パイプラインを検証する小さな合成コーパス

データセットごとに出典URL、取得日、ライセンス、加工内容を記録してください。個人情報、認証情報、利用許諾のないコードを学習に使用してはいけません。

入力は1文書1行のJSONLとし、`id`、`text`、`source`、`license`、`language` をすべて必須とします。`id` はコーパス全体で重複させないでください。

`samples/evaluation_prompts.jsonl` は学習に混ぜず、モデル間の生成比較に使う固定入力です。各行に `id`、`prompt`、`category`、`language` を記録します。
