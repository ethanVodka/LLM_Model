# LLM Core

小規模なDecoder-only Transformerと、Byte-level BPEトークナイザー、今後追加する学習・評価処理を置くPythonパッケージです。

## トークナイザー

リポジトリルートから実行します。

```powershell
uv run --project back --extra cpu mini-llm-tokenizer train
uv run --project back --extra cpu mini-llm-tokenizer encode "こんにちは"
```

学習済みファイルは `artifacts/tokenizer/tiny.json` に生成され、Git管理対象外です。

## 学習データ

```powershell
uv run --project back --extra cpu mini-llm-data prepare
uv run --project back --extra cpu mini-llm-data inspect
```

`prepare` は文書単位で分割した `train.npy` と `validation.npy` を作成します。各行は入力64トークンと、1トークン先の正解を作れる65要素です。

## 事前学習

```powershell
uv run --project back --extra cpu mini-llm-train --device cpu
```

`configs/training/tiny.yaml` に従い、AdamW、Cross Entropy、勾配クリッピングを使っ20 stepだけ学習します。チェックポイントにはモデル、Optimizer、設定、最終指標を保存します。

## 生成

```powershell
uv run --project back --extra cpu mini-llm-generate "Pythonで" --device cpu
uv run --project back --extra cpu mini-llm-generate "こんにちは" --temperature 0
```

`temperature=0` はgreedy生成、正の値はtop-k samplingで使う確率分布の鋭さを制御します。

## API

```powershell
uv run --project back --extra cpu uvicorn mini_llm.api:app --reload --port 8000
```

- `GET /api/health` — モデル読込状態を確認
- `POST /api/generate` — プロンプトと生成設定から文字列を生成

起動時にチェックポイントとトークナイザーを1回だけ読み込みます。パスは `.env` の `CHECKPOINT_PATH` と `TOKENIZER_PATH` で変更できます。

## 評価

```powershell
uv run --project back --extra cpu mini-llm-evaluate --device cpu
```

validation lossとPerplexityを計算し、`data/samples/evaluation_prompts.jsonl` の固定プロンプトをgreedy生成します。指標、生成結果、入力成果物のハッシュは `artifacts/evaluations/tiny/report.json` へ保存されます。
