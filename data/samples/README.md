# Tokenizer Sample Corpus

`tokenizer_corpus.jsonl` は、トークナイザーの学習パイプラインを検証するために本プロジェクト内で作成した合成サンプルです。外部文書や既存ソースコードからの転載は含みません。

各行は `id`、`text`、`source`、`license`、`language` を持つJSONです。実用モデルの学習データではなく、1,024語彙で学習とencode／decodeの動作を確認するためだけに使用します。実用コーパスを導入するときは、語彙数とモデル構成を一緒に再評価します。
