export type GenerationRequest = Readonly<{
  prompt: string
  maxNewTokens: number
  temperature: number
  topK: number
  seed: number
}>

export type GenerationResult = Readonly<{
  generatedText: string
  generatedTokenCount: number
  checkpointStep: number
}>

const API_BASE_URL = import.meta.env['VITE_API_BASE_URL'] ?? ''

export async function generateText(
  request: GenerationRequest,
): Promise<GenerationResult> {
  const response = await fetch(`${API_BASE_URL}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: request.prompt,
      max_new_tokens: request.maxNewTokens,
      temperature: request.temperature,
      top_k: request.topK,
      seed: request.seed,
    }),
  })
  if (!response.ok) {
    throw new Error(`生成APIがエラーを返しました（${response.status}）`)
  }

  const payload: unknown = await response.json()
  if (!isGenerationResponse(payload)) {
    throw new Error('生成APIのレスポンス形式が不正です')
  }
  return {
    generatedText: payload.generated_text,
    generatedTokenCount: payload.generated_token_count,
    checkpointStep: payload.checkpoint_step,
  }
}

function isGenerationResponse(value: unknown): value is {
  generated_text: string
  generated_token_count: number
  checkpoint_step: number
} {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  if (
    !('generated_text' in value) ||
    typeof value.generated_text !== 'string'
  ) {
    return false
  }
  if (
    !('generated_token_count' in value) ||
    typeof value.generated_token_count !== 'number'
  ) {
    return false
  }
  return 'checkpoint_step' in value && typeof value.checkpoint_step === 'number'
}
