import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { App } from './App'

describe('App', () => {
  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  it('左メニューでチャットと概要画面を切り替える', async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(
      screen.getByRole('heading', { name: 'Mini LLM' }),
    ).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'プロジェクト概要' }))

    expect(
      screen.getByRole('heading', { name: '小規模LLMを、仕組みから学ぶ' }),
    ).toBeInTheDocument()
    expect(screen.getByText('MiniDecoderLM')).toBeInTheDocument()
    expect(screen.getByText('1,729,291,264')).toBeInTheDocument()
  })

  it('ハンバーガーボタンでメニューを折りたたむ', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: 'メニューを閉じる' }))

    expect(
      screen.getByRole('button', { name: 'メニューを開く' }),
    ).toHaveAttribute('aria-expanded', 'false')
  })

  it('プロンプトをAPIへ送信して会話として表示する', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(
        JSON.stringify({
          generated_text: 'Pythonでモデルを学習します。',
          generated_token_count: 12,
          checkpoint_step: 20,
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    )
    vi.stubGlobal('fetch', fetchMock)
    render(<App />)

    const prompt = screen.getByRole('textbox', { name: 'プロンプト' })
    await user.type(prompt, 'Pythonで')
    await user.click(screen.getByRole('button', { name: '送信する' }))

    expect(screen.getByText('Pythonで')).toBeInTheDocument()
    expect(await screen.findByText('モデルを学習します。')).toBeInTheDocument()
    expect(screen.getByText('step 20 · 12 tokens')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/generate',
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('APIエラーをチャット画面に表示する', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi
        .fn<typeof fetch>()
        .mockResolvedValue(new Response(null, { status: 503 })),
    )
    render(<App />)

    await user.type(
      screen.getByRole('textbox', { name: 'プロンプト' }),
      'こんにちは',
    )
    await user.click(screen.getByRole('button', { name: '送信する' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      '生成APIがエラーを返しました（503）',
    )
  })
})
