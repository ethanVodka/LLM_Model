# Learning LLM Frontend

小規模LLMの学習状況、評価結果、生成結果を表示するReactフロントエンドです。

## コマンド

```powershell
npm install
npm run dev
```

別ターミナルでPython APIをポート8000に起動してください。開発サーバーは `/api` を `http://127.0.0.1:8000` へproxyします。別ホストのAPIを使う場合は `VITE_API_BASE_URL` を設定します。

- `npm run build` — TypeScript型検査と本番ビルド
- `npm run test` — VitestとTesting Libraryによるテスト
- `npm run lint` — Oxlintによる静的検査（`any`禁止を含む）
- `npm run format:check` — Prettierの整形確認

実装規約はリポジトリルートの `AGENTS.md` を参照してください。
