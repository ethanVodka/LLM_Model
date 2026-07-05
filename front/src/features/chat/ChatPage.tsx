import {
  useRef,
  useState,
  type FormEventHandler,
  type MutableRefObject,
} from 'react'
import { generateText } from '../../services/generationApi'

type ChatMessage = Readonly<{
  id: number
  role: 'user' | 'assistant'
  content: string
  metadata: string | undefined
}>

const SUGGESTIONS = [
  'こんにちは',
  '何ができますか？',
  '言語モデルとは何ですか？',
] as const

export function ChatPage() {
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<readonly ChatMessage[]>([])
  const [maxNewTokens, setMaxNewTokens] = useState(50)
  const [temperature, setTemperature] = useState(0)
  const [topK, setTopK] = useState(50)
  const [error, setError] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const nextMessageId = useRef(1)

  const handleSubmit: FormEventHandler<HTMLFormElement> = async (event) => {
    event.preventDefault()
    const submittedPrompt = prompt.trim()
    if (submittedPrompt.length === 0 || isGenerating) {
      return
    }

    const userMessage = createMessage(nextMessageId, 'user', submittedPrompt)
    setMessages((current) => [...current, userMessage])
    setPrompt('')
    setError(null)
    setIsGenerating(true)
    try {
      const generated = await generateText({
        prompt: submittedPrompt,
        maxNewTokens,
        temperature,
        topK,
        seed: 42,
      })
      const assistantText = removeRepeatedPrompt(
        generated.generatedText,
        submittedPrompt,
      )
      const assistantMessage = createMessage(
        nextMessageId,
        'assistant',
        assistantText,
        `step ${generated.checkpointStep} · ${generated.generatedTokenCount} tokens`,
      )
      setMessages((current) => [...current, assistantMessage])
    } catch (caught: unknown) {
      setError(
        caught instanceof Error
          ? caught.message
          : '生成中に不明なエラーが発生しました',
      )
    } finally {
      setIsGenerating(false)
    }
  }

  return (
    <section className="chat-page" aria-labelledby="chat-title">
      <header className="chat-header">
        <div>
          <h1 id="chat-title">Mini LLM</h1>
          <p>Qwen3-1.7B + LoRA</p>
        </div>
        <span className="model-status">ローカル</span>
      </header>

      <div className="chat-scroll" aria-live="polite">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <div className="model-mark" aria-hidden="true">
              L
            </div>
            <h2>何を話しますか？</h2>
            <p>未知の質問にも対応するローカルQLoRAモデルです。</p>
            <div className="suggestion-grid">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => setPrompt(suggestion)}
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <article
                className="chat-message"
                data-role={message.role}
                key={message.id}
              >
                <div className="message-avatar" aria-hidden="true">
                  {message.role === 'user' ? '私' : 'L'}
                </div>
                <div className="message-body">
                  <p className="message-author">
                    {message.role === 'user' ? 'あなた' : 'Mini LLM'}
                  </p>
                  <pre>{message.content}</pre>
                  {message.metadata !== undefined && (
                    <p className="message-meta">{message.metadata}</p>
                  )}
                </div>
              </article>
            ))}
            {isGenerating && (
              <p className="generating-indicator">Mini LLM が生成中…</p>
            )}
          </div>
        )}
      </div>

      <div className="composer-area">
        {error !== null && <p role="alert">{error}</p>}
        <form className="chat-composer" onSubmit={handleSubmit}>
          <textarea
            aria-label="プロンプト"
            placeholder="Mini LLM にメッセージを送信"
            value={prompt}
            onChange={(event) => setPrompt(event.currentTarget.value)}
            rows={2}
            maxLength={4000}
          />
          <button
            className="send-button"
            type="submit"
            disabled={isGenerating || prompt.trim().length === 0}
            aria-label={isGenerating ? '生成中' : '送信する'}
          >
            <span aria-hidden="true">↑</span>
          </button>
        </form>
        <details className="generation-settings">
          <summary>生成設定</summary>
          <div>
            <label>
              生成数
              <input
                type="number"
                min="1"
                max="256"
                value={maxNewTokens}
                onChange={(event) =>
                  setMaxNewTokens(event.currentTarget.valueAsNumber)
                }
              />
            </label>
            <label>
              Temperature
              <input
                type="number"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(event) =>
                  setTemperature(event.currentTarget.valueAsNumber)
                }
              />
            </label>
            <label>
              Top-k
              <input
                type="number"
                min="1"
                max="1024"
                value={topK}
                onChange={(event) => setTopK(event.currentTarget.valueAsNumber)}
              />
            </label>
          </div>
        </details>
        <p className="composer-note">
          Mini LLMは検証用です。出力内容は正確とは限りません。
        </p>
      </div>
    </section>
  )
}

function createMessage(
  nextMessageId: MutableRefObject<number>,
  role: ChatMessage['role'],
  content: string,
  metadata?: string,
): ChatMessage {
  const message = { id: nextMessageId.current, role, content, metadata }
  nextMessageId.current += 1
  return message
}

function removeRepeatedPrompt(generatedText: string, prompt: string): string {
  if (!generatedText.startsWith(prompt)) {
    return generatedText
  }
  const completion = generatedText.slice(prompt.length).trimStart()
  return completion.length > 0 ? completion : '（続きは生成されませんでした）'
}
