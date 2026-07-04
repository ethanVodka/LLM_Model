import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { App } from './App'

describe('App', () => {
  it('開発目的とモデル情報を表示する', () => {
    render(<App />)

    expect(
      screen.getByRole('heading', { name: '小規模LLMを、仕組みから学ぶ' }),
    ).toBeInTheDocument()
    expect(screen.getByText('MiniDecoderLM')).toBeInTheDocument()
    expect(screen.getByText('4,273,664')).toBeInTheDocument()
  })
})
