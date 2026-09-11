/**
 * WorkspaceView — ChatGPT-style application shell
 *
 * Architecture:
 *   <div> 100vw x 100vh, overflow:hidden, flex-row
 *   ├── <AppSidebar>  fixed-width column, flex flex-col
 *   └── <main>        flex:1, min-w:0, flex flex-col
 *       ├── <scrollable messages>  flex:1, overflow-y:auto
 *       └── <composer>            flex-shrink:0
 */

import React, { useRef, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Plus,
  MessageSquare,
  User,
  PanelLeft,
} from 'lucide-react'
import ChatThread from './ChatThread'
import PromptComposer from './PromptComposer'
import { cn } from '@/lib/utils'

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
   Sidebar logo
───────────────────────────────────────────────────────── */
const SatLogo = ({ expanded }: { expanded: boolean }) => (
  <div className="flex items-center overflow-hidden">
    <img
      src="/logo.png"
      alt="SatQuery AI"
      className={cn(
        'object-contain transition-all duration-200 flex-shrink-0',
        expanded ? 'h-9 w-auto max-w-[160px]' : 'h-8 w-8 object-left object-cover rounded-md'
      )}
      style={{ filter: 'drop-shadow(0 2px 6px rgba(0,0,0,0.4))' }}
    />
  </div>
)

/* ─────────────────────────────────────────────────────────
   AppSidebar — 3-tier Flexbox Shell Architecture
───────────────────────────────────────────────────────── */
const AppSidebar = ({
  expanded,
  setExpanded,
  onNewAnalysis,
  chatHistory,
  activeConversationId,
  onSelectConversation,
}: {
  expanded: boolean
  setExpanded: (v: boolean) => void
  onNewAnalysis: () => void
  chatHistory?: Record<string, ChatHistoryItem[]>
  activeConversationId?: string
  onSelectConversation?: (id: string) => void
}) => {
  return (
    <motion.aside
      animate={{ width: expanded ? 280 : 76 }}
      transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
      className="flex-shrink-0 flex flex-col h-full overflow-hidden p-4 select-none"
      style={{
        background: 'rgba(16, 16, 18, 0.94)',
        backdropFilter: 'blur(24px)',
        WebkitBackdropFilter: 'blur(24px)',
        borderRight: '1px solid rgba(255, 255, 255, 0.08)',
        zIndex: 20,
      }}
    >
      {/* ── 1. FIXED TOP SECTION (Header + New Analysis) ── */}
      <div className="flex-shrink-0 flex flex-col gap-4">
        {/* Header: Logo + Collapse Button */}
        <div className="flex items-center justify-between gap-2 px-1">
          <SatLogo expanded={expanded} />
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-10 h-10 flex items-center justify-center rounded-lg text-white/50 hover:text-white hover:bg-white/10 transition-colors cursor-pointer flex-shrink-0"
            title={expanded ? 'Collapse sidebar' : 'Expand sidebar'}
          >
            <PanelLeft className="w-5 h-5" />
          </button>
        </div>

        {/* New Analysis Action Button */}
        <button
          onClick={onNewAnalysis}
          className={cn(
            'flex items-center rounded-xl cursor-pointer transition-all duration-200',
            'text-white hover:bg-white/[0.12] group',
            expanded ? 'gap-3 px-3.5 py-3 w-full' : 'justify-center w-11 h-11 mx-auto',
          )}
          style={{
            background: 'rgba(255, 255, 255, 0.08)',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
          }}
          title="New Analysis"
        >
          <div className="w-6 h-6 rounded-lg bg-white/12 flex items-center justify-center group-hover:bg-white/20 transition-colors flex-shrink-0">
            <Plus className="w-4 h-4 text-white" />
          </div>
          <AnimatePresence>
            {expanded && (
              <motion.span
                key="new-analysis-text"
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.18 }}
                className="text-[14px] font-semibold tracking-tight whitespace-nowrap overflow-hidden text-left"
              >
                New Analysis
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>

      {/* Divider */}
      <div className="h-[1px] w-full bg-white/[0.08] flex-shrink-0 my-4" />

      {/* ── 2. SCROLLABLE MIDDLE SECTION (History Area) ── */}
      <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-4 pr-1">
        {expanded && chatHistory && Object.keys(chatHistory).length > 0 && (
          Object.entries(chatHistory).map(([group, items]) => (
            <div key={group} className="flex flex-col gap-2">
              <p
                className="px-2 pt-1 pb-1 text-[11px] uppercase tracking-[0.16em] font-bold text-white/40"
              >
                {group}
              </p>
              <div className="flex flex-col gap-1.5">
                {items.map((item) => {
                  const isActive = activeConversationId === item.id || item.active
                  return (
                    <button
                      key={item.id}
                      onClick={() => onSelectConversation?.(item.id)}
                      className={cn(
                        'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-left text-[14px]',
                        'transition-all duration-150 cursor-pointer group',
                        isActive
                          ? 'bg-white/10 text-white font-medium shadow-sm'
                          : 'text-white/60 hover:text-white hover:bg-white/[0.06]',
                      )}
                    >
                      <MessageSquare
                        className={cn(
                          'w-[16px] h-[16px] flex-shrink-0 transition-opacity',
                          isActive ? 'text-white opacity-100' : 'text-white/45 group-hover:opacity-85'
                        )}
                      />
                      <span className="truncate leading-snug">{item.title}</span>
                    </button>
                  )
                })}
              </div>
            </div>
          ))
        )}
      </div>

      {/* ── 3. FIXED BOTTOM SECTION (User Profile Area) ── */}
      <div className="flex-shrink-0 pt-3 flex flex-col gap-3" style={{ paddingBottom: 20 }}>
        <div className="h-[1px] w-full bg-white/[0.08]" />

        <button
          className={cn(
            'flex items-center rounded-xl cursor-pointer transition-all duration-150',
            'text-white/80 hover:text-white hover:bg-white/[0.07]',
            expanded ? 'gap-3 px-3.5 py-3 w-full' : 'justify-center w-11 h-11 mx-auto',
          )}
          style={{
            background: 'rgba(255, 255, 255, 0.04)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
          }}
        >
          <div
            className="h-9 w-9 flex-shrink-0 rounded-full flex items-center justify-center bg-white/10 border border-white/15"
          >
            <User className="w-4.5 h-4.5 text-white/90" />
          </div>
          <AnimatePresence>
            {expanded && (
              <motion.div
                key="user-info"
                initial={{ opacity: 0, width: 0 }}
                animate={{ opacity: 1, width: 'auto' }}
                exit={{ opacity: 0, width: 0 }}
                transition={{ duration: 0.18 }}
                className="flex flex-col text-left overflow-hidden gap-0.5"
              >
                <span className="text-[14px] font-semibold text-white/90 whitespace-nowrap leading-tight">
                  Analyst
                </span>
                <span className="text-[11.5px] text-white/40 whitespace-nowrap leading-tight">
                  Pro Workspace
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </button>
      </div>
    </motion.aside>
  )
}

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
  const [sidebarExpanded, setSidebarExpanded] = useState(true)

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
      <AppSidebar
        expanded={sidebarExpanded}
        setExpanded={setSidebarExpanded}
        onNewAnalysis={onNewAnalysis}
        chatHistory={chatHistory}
        activeConversationId={activeConversationId}
        onSelectConversation={onSelectConversation}
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
