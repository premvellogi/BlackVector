import { useEffect, useRef, useCallback, useTransition } from "react";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { SendIcon, Paperclip, X } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import * as React from "react";

/* ─────────────────────────────────────────────
   Auto-resize textarea hook
───────────────────────────────────────────── */
function useAutoResizeTextarea({ minHeight, maxHeight }: { minHeight: number; maxHeight?: number }) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const el = textareaRef.current;
      if (!el) return;
      if (reset) { el.style.height = `${minHeight}px`; return; }
      el.style.height = `${minHeight}px`;
      el.style.height = `${Math.max(minHeight, Math.min(el.scrollHeight, maxHeight ?? Infinity))}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    const el = textareaRef.current;
    if (el) el.style.height = `${minHeight}px`;
  }, [minHeight]);

  return { textareaRef, adjustHeight };
}

/* ─────────────────────────────────────────────
   Typing indicator dots
───────────────────────────────────────────── */
function TypingDots() {
  return (
    <div className="flex items-center gap-1">
      {[0, 1, 2].map((i) => (
        <motion.span
          key={i}
          className="block w-1.5 h-1.5 rounded-full bg-current"
          animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
          transition={{ duration: 0.9, repeat: Infinity, delay: i * 0.18, ease: "easeInOut" }}
        />
      ))}
    </div>
  );
}

/* ─────────────────────────────────────────────
   Props
───────────────────────────────────────────── */
interface AnimatedAIChatProps {
  onSubmit?: (query: string, files: File[]) => void;
  placeholder?: string;
}

/* ─────────────────────────────────────────────
   Composer — textarea + upload pin + send
───────────────────────────────────────────── */
export function AnimatedAIChat({
  onSubmit,
  placeholder = "Describe what you want to analyze...",
}: AnimatedAIChatProps) {
  const [value, setValue] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [, startTransition] = useTransition();
  const { textareaRef, adjustHeight } = useAutoResizeTextarea({ minHeight: 96, maxHeight: 240 });
  const fileInputRef = useRef<HTMLInputElement>(null);

  const canSend = (value.trim().length > 0 || files.length > 0) && !isLoading;

  /* keyboard: Enter = send, Shift+Enter = newline */
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) handleSend();
    }
  };

  const handleSend = () => {
    if (!canSend) return;
    startTransition(() => {
      setIsLoading(true);
      onSubmit?.(value.trim(), files);
      setTimeout(() => {
        setIsLoading(false);
        setValue("");
        setFiles([]);
        adjustHeight(true);
      }, 400);
    });
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFiles(prev => [...prev, ...Array.from(e.target.files!)]);
    }
    e.target.value = "";
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFiles(prev => [...prev, ...Array.from(e.dataTransfer.files)]);
    }
  };

  const removeFile = (i: number) => setFiles(prev => prev.filter((_, idx) => idx !== i));

  return (
    <motion.div
      className="w-full"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1], delay: 0.15 }}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept="image/*,.tif,.tiff,.png,.jpg,.jpeg,.webp,.npz,.nc,.geojson"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* ── Glass card — matches navbar: bg #1f1f1f57, border #333 ── */}
      <div
        className="relative rounded-2xl overflow-hidden transition-colors"
        style={{
          background: isDragging ? "rgba(34, 211, 238, 0.12)" : "rgba(31, 31, 31, 0.34)",
          backdropFilter: "blur(24px)",
          WebkitBackdropFilter: "blur(24px)",
          border: isDragging ? "1px solid rgba(34, 211, 238, 0.6)" : "1px solid #333",
          boxShadow: "0 8px 48px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.04)",
        }}
      >
        {/* ── Textarea area ── */}
        <div style={{ padding: "20px 24px 12px 24px" }}>
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => { setValue(e.target.value); adjustHeight(); }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            className={cn(
              "w-full resize-none bg-transparent border-none outline-none",
              "text-white/90 leading-relaxed",
              "placeholder:text-white/30"
            )}
            style={{
              minHeight: 96,
              fontSize: 15,
              fontFamily: "Inter, system-ui, sans-serif",
              overflow: "hidden",
            }}
          />
        </div>

        {/* ── Attached files ── */}
        <AnimatePresence>
          {files.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              style={{ paddingLeft: 24, paddingRight: 24, paddingBottom: 12 }}
            >
              <div className="flex flex-wrap gap-2">
                {files.map((file, i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.9 }}
                    className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-xs text-white/70"
                    style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}
                  >
                    <Paperclip className="w-3 h-3 opacity-60" />
                    <span className="max-w-[160px] truncate">{file.name}</span>
                    <button
                      onClick={() => removeFile(i)}
                      className="opacity-50 hover:opacity-100 transition-opacity ml-1"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Bottom toolbar ── */}
        <div
          className="flex items-center justify-between"
          style={{
            padding: "12px 20px 18px 20px",
            borderTop: "1px solid #2a2a2a",
          }}
        >
          {/* Left — Upload / pin button */}
          <motion.button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            title="Attach satellite imagery or data files"
            className="flex items-center gap-2 rounded-xl text-white/50 hover:text-white/90 transition-colors duration-200"
            style={{
              padding: "10px 14px",
              background: "rgba(255,255,255,0.06)",
              border: "1px solid #333",
              fontSize: 13,
              fontFamily: "Inter, system-ui, sans-serif",
              cursor: "pointer",
            }}
          >
            <Paperclip className="w-4 h-4" />
            <span>Attach files</span>
          </motion.button>

          {/* Right — Send button */}
          <motion.button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            whileHover={canSend ? { scale: 1.03 } : {}}
            whileTap={canSend ? { scale: 0.97 } : {}}
            className="flex items-center gap-2.5 rounded-xl font-medium transition-all duration-200"
            style={{
              padding: "10px 20px",
              fontSize: 14,
              fontFamily: "Inter, system-ui, sans-serif",
              cursor: canSend ? "pointer" : "not-allowed",
              background: canSend ? "#ffffff" : "rgba(255,255,255,0.07)",
              color: canSend ? "#0a0f0b" : "rgba(255,255,255,0.25)",
              boxShadow: canSend ? "0 4px 24px rgba(255,255,255,0.12)" : "none",
            }}
          >
            <AnimatePresence mode="wait">
              {isLoading ? (
                <motion.span
                  key="loading"
                  className="flex items-center gap-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <TypingDots />
                  <span>Analyzing</span>
                </motion.span>
              ) : (
                <motion.span
                  key="send"
                  className="flex items-center gap-2"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <SendIcon className="w-4 h-4" />
                  <span>Send</span>
                </motion.span>
              )}
            </AnimatePresence>
          </motion.button>
        </div>
      </div>

    </motion.div>
  );
}

export default AnimatedAIChat;
