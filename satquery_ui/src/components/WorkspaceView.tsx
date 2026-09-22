/**
 * WorkspaceView — ChatGPT-style application shell
 *
 * Architecture:
 *   <div> 100vw x 100vh, overflow:hidden, flex-row
 *   ├── <ChatSidebar>  fixed-width column, flex flex-col
 *   └── <main>         flex:1, min-w:0, flex flex-col
 *       ├── <scrollable messages>  flex:1, overflow-y:auto
 *       └── <composer>            flex-shrink:0
 */

import React, { useRef, useEffect, useMemo } from 'react'
import ChatThread from './ChatThread'
import PromptComposer from './PromptComposer'
import { ChatSidebar } from './ChatSidebar'
import { useAuth } from './auth/AuthProvider'
import { useAuthModal } from './auth/AuthModalContext'

/* ─────────────────────────────────────────────────────────
   Types
───────────────────────────────────────────────────────── */
interface WorkspaceMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  taskType?: string
  inputType?: string
  modelUsed?: string
  confidence?: number
  opticalPct?: number
  sarPct?: number
  isStreaming?: boolean
}

interface ChatHistoryItem {
  id: string
  title: string
  active?: boolean
}

interface WorkspaceViewProps {
  messages: WorkspaceMessage[]
  isLoading: boolean
  onNewMessage: (query: string, files: any[], task: string | null) => void
  onNewAnalysis: () => void
  onRetry?: () => void
  onSelectConversation?: (id: string) => void
  chatHistory?: Record<string, ChatHistoryItem[]>
  activeConversationId?: string
}

/* ─────────────────────────────────────────────────────────
   (AppSidebar removed — now uses <ChatSidebar /> component)
───────────────────────────────────────────────────────── */

/* ─────────────────────────────────────────────────────────
   WorkspaceView — root shell
───────────────────────────────────────────────────────── */
export const WorkspaceView: React.FC<WorkspaceViewProps> = ({
  messages,
  isLoading,
  onNewMessage,
  onNewAnalysis,
  chatHistory,
  activeConversationId,
  onRetry,
  onSelectConversation,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null)
  const { user, signOut } = useAuth()
  const { openAuthModal } = useAuthModal()

  const sidebarUserInfo = useMemo(() => {
    if (!user) return null

    const userName =
      user.user_metadata?.full_name ||
      user.user_metadata?.name ||
      user.email?.split('@')[0] ||
      'Analyst'

    const userInitials = userName
      .split(' ')
      .filter(Boolean)
      .map((p: string) => p[0])
      .slice(0, 2)
      .join('')
      .toUpperCase() || 'AN'

    return {
      name: userName,
      initials: userInitials,
      subtitle: user.email ? (user.email.length > 22 ? user.email.slice(0, 20) + '…' : user.email) : 'Pro Workspace',
      avatarUrl: user.user_metadata?.avatar_url || user.user_metadata?.picture,
      email: user.email,
    }
  }, [user])

  /* Flatten grouped chat history → flat array for sidebar recents */
  const flatRecentChats = useMemo(() => {
    if (!chatHistory) return []
    return Object.values(chatHistory).flat().map(item => ({
      id: item.id,
      title: item.title,
    }))
  }, [chatHistory])

  /* Auto-scroll to newest message */
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading])

  return (
    /*
      ROOT SHELL
      ┌────────────────────────────────────────────────────────┐
      │  flex-row, 100vw x 100vh, overflow:hidden              │
      │  ┌──────────────┬───────────────────────────────────┐  │
      │  │  AppSidebar  │  main (flex:1, min-w:0)           │  │
      │  │  flex-shrink │  ┌─────────────────────────────┐  │  │
      │  │  :0          │  │ messages (flex:1, scroll)    │  │  │
      │  │              │  ├─────────────────────────────┤  │  │
      │  │              │  │ composer (flex-shrink:0)     │  │  │
      │  │              │  └─────────────────────────────┘  │  │
      │  └──────────────┴───────────────────────────────────┘  │
      └────────────────────────────────────────────────────────┘
    */
    <div
      style={{
        display: 'flex',
        flexDirection: 'row',
        width: '100vw',
        height: '100vh',
        overflow: 'hidden',
        position: 'relative',
        zIndex: 10,
      }}
    >
      {/* SIDEBAR */}
      <ChatSidebar
        onNewChat={onNewAnalysis}
        recentChats={flatRecentChats}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
        userInfo={sidebarUserInfo}
        onSignOut={signOut}
        onSignIn={() => openAuthModal('login')}
      />

      {/* MAIN COLUMN */}
      <main
        style={{
          display: 'flex',
          flexDirection: 'column',
          flex: 1,
          minWidth: 0,
          height: '100%',
          overflow: 'hidden',
        }}
      >
        {/* ── Messages (scrollable) ── */}
        <div
          ref={scrollRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            scrollbarWidth: 'thin',
            scrollbarColor: 'rgba(255,255,255,0.10) transparent',
          }}
        >
          {/* Empty state */}
          {messages.length === 0 && !isLoading && (
            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                gap: 16,
                padding: '0 24px',
              }}
            >
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'rgba(255,255,255,0.05)',
                  border: '1px solid rgba(255,255,255,0.08)',
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="4" fill="rgba(255,255,255,0.6)" />
                  <ellipse cx="12" cy="12" rx="10" ry="4.5" stroke="rgba(255,255,255,0.30)" strokeWidth="1.2" transform="rotate(-30 12 12)" />
                  <ellipse cx="12" cy="12" rx="10" ry="4.5" stroke="rgba(255,255,255,0.15)" strokeWidth="1" transform="rotate(30 12 12)" />
                </svg>
              </div>
              <p style={{ color: 'rgba(255,255,255,0.35)', fontSize: 14, textAlign: 'center' }}>
                Start by asking a question below
              </p>
            </div>
          )}

          {/* Messages — centered max-width */}
          <div style={{ width: '100%', padding: '24px 24px' }}>
            <div style={{ maxWidth: 800, margin: '0 auto' }}>
              <ChatThread messages={messages} isLoading={isLoading} onRetry={onRetry} />
            </div>
          </div>
        </div>

        {/* ── Composer (pinned at bottom) ── */}
        <div
          style={{
            flexShrink: 0,
            width: '100%',
            padding: '16px 24px 24px',
            borderTop: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <div style={{ maxWidth: 800, margin: '0 auto' }}>
            <PromptComposer onSubmit={onNewMessage} />
            <p
              style={{
                textAlign: 'center',
                fontSize: 11,
                marginTop: 10,
                color: 'rgba(255,255,255,0.15)',
              }}
            >
              SatQuery AI may produce inaccurate analysis. Always verify with domain expertise.
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}

export default WorkspaceView
