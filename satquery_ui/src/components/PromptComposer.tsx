import React from 'react'
import { AIChatInput } from '@/components/ui/ai-chat-input'

interface PromptComposerProps {
  onSubmit: (query: string, files: any[], task: string | null) => void
}

/**
 * Workspace Prompt Composer
 * Uses the AIChatInput expanding composer — dark themed.
 * Only used in WorkspaceView, NOT in HeroSection.
 */
export const PromptComposer: React.FC<PromptComposerProps> = ({ onSubmit }) => {
  return (
    <AIChatInput onSubmit={onSubmit} />
  )
}

export default PromptComposer
