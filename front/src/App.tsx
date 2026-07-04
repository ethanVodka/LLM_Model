import './App.css'

export function App() {
  return (
    <main className="app-shell">
      <section className="hero" aria-labelledby="app-title">
        <p className="eyebrow">Learning LLM</p>
        <h1 id="app-title">小規模LLMを、仕組みから学ぶ</h1>
        <p className="summary">
          日本語とコードを扱うモデルを段階的に実装し、学習・評価・改善の過程を可視化します。
        </p>
        <dl className="status-list" aria-label="現在の開発状況">
          <div>
            <dt>Model</dt>
            <dd>MiniDecoderLM</dd>
          </div>
          <div>
            <dt>Parameters</dt>
            <dd>4,273,664</dd>
          </div>
          <div>
            <dt>Phase</dt>
            <dd>Foundation</dd>
          </div>
        </dl>
      </section>
    </main>
  )
}
