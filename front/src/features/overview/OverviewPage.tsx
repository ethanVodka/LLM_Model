export function OverviewPage() {
  return (
    <section className="overview-page" aria-labelledby="overview-title">
      <div className="overview-content">
        <p className="eyebrow">Learning LLM</p>
        <h1 id="overview-title">小規模LLMを、仕組みから学ぶ</h1>
        <p className="overview-summary">
          日本語とコードを扱うモデルを段階的に実装し、学習・評価・改善の過程を可視化します。
        </p>
        <dl className="status-list" aria-label="現在の開発状況">
          <div>
            <dt>Model</dt>
            <dd>MiniDecoderLM</dd>
          </div>
          <div>
            <dt>Parameters</dt>
            <dd>1,729,291,264</dd>
          </div>
          <div>
            <dt>Phase</dt>
            <dd>Qwen3 QLoRA</dd>
          </div>
        </dl>
        <div className="pipeline-card">
          <h2>現在のパイプライン</h2>
          <ol>
            <li>トークナイザー学習</li>
            <li>次トークン予測データ準備</li>
            <li>Decoder-only Transformer学習</li>
            <li>API経由の自己回帰生成</li>
            <li>assistant回答だけを対象にしたSFT</li>
          </ol>
        </div>
      </div>
    </section>
  )
}
