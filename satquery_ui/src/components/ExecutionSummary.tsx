import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, CheckCircle2, Circle, Loader2 } from 'lucide-react'

interface ExecutionStep {
  label: string
  detail: string
  done: boolean
  active?: boolean
}

interface ExecutionSummaryProps {
  taskType: string
  inputType: string
  modelUsed: string
  confidence: number
  isStreaming?: boolean
}

const TASK_LABELS: Record<string, string> = {
  vqa: 'Visual Question Answering',
  caption: 'Scene Description',
  change: 'Change Detection',
  fusion: 'Cross-Modal Fusion',
  agentic: 'Agentic Analysis',
}

export const ExecutionSummary: React.FC<ExecutionSummaryProps> = ({
  taskType,
  inputType,
  modelUsed,
  confidence,
  isStreaming = false,
}) => {
  const [open, setOpen] = useState(false)

  const pct = (confidence * 100).toFixed(0)
  const taskLabel = TASK_LABELS[taskType] ?? taskType

  const steps: ExecutionStep[] = [
    {
      label: 'Query classified',
      detail: taskLabel,
      done: true,
    },
    {
      label: 'Satellite input validated',
      detail: inputType,
      done: true,
    },
    {
      label: 'Model selected',
      detail: modelUsed,
      done: true,
    },
    {
      label: 'Analysis completed',
      detail: isStreaming ? 'In progress...' : `Confidence ${pct}%`,
      done: !isStreaming,
      active: isStreaming,
    },
  ]

  return (
    <div
      className="mt-4 overflow-hidden"
      style={{
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '0.75rem',
      }}
    >
      {/* Header */}
      <button
        onClick={() => setOpen(v => !v)}
        className="w-full flex items-center justify-between px-4 py-3 text-left group"
      >
        <div className="flex items-center gap-2.5">
          {isStreaming ? (
            <Loader2 className="w-3 h-3 text-white/40 animate-spin" />
          ) : (
            <CheckCircle2 className="w-3 h-3 text-white/35" />
          )}
          <span
            className="text-[11px] uppercase tracking-[0.10em] font-medium"
            style={{ color: 'rgba(255,255,255,0.35)' }}
          >
            Execution Summary
          </span>
          {!isStreaming && (
            <span
              className="text-[10px] font-medium px-1.5 py-0.5 rounded-full"
              style={{
                background: 'rgba(255,255,255,0.06)',
                color: 'rgba(255,255,255,0.40)',
              }}
            >
              {pct}% confidence
            </span>
          )}
        </div>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.2 }}
        >
          <ChevronDown className="w-3.5 h-3.5 text-white/25 group-hover:text-white/45 transition-colors" />
        </motion.div>
      </button>

      {/* Accordion body */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div
              className="px-4 pb-4 pt-1"
              style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}
            >
              {steps.map((step, i) => (
                <motion.div
                  key={i}
                  initial={{ opacity: 0, x: -6 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05, duration: 0.18 }}
                  className="flex items-start gap-3 py-2.5"
                  style={{ borderBottom: i < steps.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}
                >
                  <div className="flex-shrink-0 mt-0.5">
                    {step.active ? (
                      <Loader2 className="w-3.5 h-3.5 text-white/35 animate-spin" />
                    ) : step.done ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-white/40" />
                    ) : (
                      <Circle className="w-3.5 h-3.5 text-white/15" />
                    )}
                  </div>
                  <div className="min-w-0">
                    <div className="text-[12px] font-medium text-white/70">{step.label}</div>
                    <div className="text-[11px] text-white/35 mt-0.5 truncate">{step.detail}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ExecutionSummary
