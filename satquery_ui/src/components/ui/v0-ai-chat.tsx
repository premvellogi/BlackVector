"use client";

import React, { useEffect, useRef, useCallback, useState } from "react";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import {
  ImageIcon,
  Layers,
  Sparkles,
  ArrowUpIcon,
  Paperclip,
  PlusIcon,
  Compass,
  Zap,
} from "lucide-react";

interface UseAutoResizeTextareaProps {
  minHeight: number;
  maxHeight?: number;
}

function useAutoResizeTextarea({
  minHeight,
  maxHeight,
}: UseAutoResizeTextareaProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const adjustHeight = useCallback(
    (reset?: boolean) => {
      const textarea = textareaRef.current;
      if (!textarea) return;

      if (reset) {
        textarea.style.height = `${minHeight}px`;
        return;
      }

      textarea.style.height = `${minHeight}px`;

      const newHeight = Math.max(
        minHeight,
        Math.min(
          textarea.scrollHeight,
          maxHeight ?? Number.POSITIVE_INFINITY
        )
      );

      textarea.style.height = `${newHeight}px`;
    },
    [minHeight, maxHeight]
  );

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = `${minHeight}px`;
    }
  }, [minHeight]);

  useEffect(() => {
    const handleResize = () => adjustHeight();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [adjustHeight]);

  return { textareaRef, adjustHeight };
}

interface VercelV0ChatProps {
  onSubmit?: (query: string, files: any[], task: string | null) => void;
}

export function VercelV0Chat({ onSubmit }: VercelV0ChatProps) {
  const [value, setValue] = useState("");
  const [selectedTask, setSelectedTask] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [uploadedFiles, setUploadedFiles] = useState<{ name: string; type: any }[]>([]);

  const { textareaRef, adjustHeight } = useAutoResizeTextarea({
    minHeight: 60,
    maxHeight: 200,
  });

  const handleSend = () => {
    if (value.trim() || uploadedFiles.length > 0) {
      if (onSubmit) {
        onSubmit(value, uploadedFiles, selectedTask);
      }
      setValue("");
      setUploadedFiles([]);
      setSelectedTask(null);
      adjustHeight(true);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    const mapped = files.map(f => ({ name: f.name, type: 'optical' as const }));
    setUploadedFiles(prev => [...prev, ...mapped]);
  };

  return (
    <div className="flex flex-col items-center justify-center w-full max-w-3xl mx-auto p-4 space-y-6 text-center">
      {/* Hidden File Input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".tif,.tiff,.png,.jpg,.jpeg"
        className="hidden"
        onChange={handleFileUpload}
      />

      {/* Geist Pixel Heading */}
      <h1 className="text-3xl sm:text-4xl md:text-5xl font-bold font-pixel tracking-wider text-white drop-shadow-lg select-none">
        Ask the Earth. See the Answer.
      </h1>

      {/* Main Composer Box */}
      <div className="w-full text-left">
        <div className="relative bg-neutral-900/80 backdrop-blur-xl rounded-2xl border border-neutral-800/80 shadow-2xl overflow-hidden hover:border-neutral-700/80 transition-all">
          {/* File chips */}
          {uploadedFiles.length > 0 && (
            <div className="flex flex-wrap gap-2 p-3 pb-0">
              {uploadedFiles.map((f, idx) => (
                <span
                  key={idx}
                  className="bg-neutral-800 text-white/80 text-xs px-2.5 py-1 rounded-full border border-neutral-700 flex items-center gap-1.5"
                >
                  <span>{f.name}</span>
                  <button
                    onClick={() => setUploadedFiles(prev => prev.filter((_, i) => i !== idx))}
                    className="hover:text-white text-neutral-400"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}

          <div className="overflow-y-auto">
            <Textarea
              ref={textareaRef}
              value={value}
              onChange={(e) => {
                setValue(e.target.value);
                adjustHeight();
              }}
              onKeyDown={handleKeyDown}
              placeholder="Ask anything about your satellite imagery..."
              className={cn(
                "w-full px-4 py-3.5",
                "resize-none",
                "bg-transparent",
                "border-none",
                "text-white text-base",
                "focus:outline-none",
                "focus-visible:ring-0 focus-visible:ring-offset-0",
                "placeholder:text-neutral-500 placeholder:text-sm md:placeholder:text-base",
                "min-h-[60px]"
              )}
              style={{
                overflow: "hidden",
              }}
            />
          </div>

          <div className="flex items-center justify-between p-3 border-t border-neutral-800/50 bg-neutral-900/40">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="group p-2 hover:bg-neutral-800/80 rounded-lg transition-colors flex items-center gap-1.5 text-neutral-400 hover:text-white"
              >
                <Paperclip className="w-4 h-4 text-neutral-400 group-hover:text-white" />
                <span className="text-xs text-zinc-400 hidden sm:group-hover:inline transition-opacity">
                  Attach GeoTIFF / Image
                </span>
              </button>
            </div>

            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setSelectedTask(prev => prev === 'fusion' ? null : 'fusion')}
                className={cn(
                  "px-2.5 py-1.5 rounded-lg text-xs transition-colors border border-dashed flex items-center justify-between gap-1.5",
                  selectedTask
                    ? "border-white/40 bg-white/10 text-white font-medium"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-600 hover:bg-zinc-800"
                )}
              >
                <PlusIcon className="w-3.5 h-3.5" />
                <span>{selectedTask ? selectedTask.toUpperCase() : 'Analysis Mode'}</span>
              </button>

              <button
                type="button"
                onClick={handleSend}
                disabled={!value.trim() && uploadedFiles.length === 0}
                className={cn(
                  "p-2 rounded-lg text-sm transition-colors border flex items-center justify-center",
                  value.trim() || uploadedFiles.length > 0
                    ? "bg-white text-black border-white hover:bg-white/90"
                    : "border-zinc-800 text-zinc-600 bg-neutral-900 cursor-not-allowed"
                )}
              >
                <ArrowUpIcon
                  className={cn(
                    "w-4 h-4",
                    value.trim() || uploadedFiles.length > 0
                      ? "text-black"
                      : "text-zinc-600"
                  )}
                />
                <span className="sr-only">Send</span>
              </button>
            </div>
          </div>
        </div>

        {/* Action Quick Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-2.5 mt-5">
          <ActionButton
            icon={<ImageIcon className="w-3.5 h-3.5 text-emerald-400" />}
            label="Optical Imagery Analysis"
            onClick={() => setValue("Analyze urban land cover and vegetation in this optical image")}
          />
          <ActionButton
            icon={<Layers className="w-3.5 h-3.5 text-cyan-400" />}
            label="SAR Sentinel-1 Processing"
            onClick={() => setValue("Use SAR Sentinel-1 imagery to detect water bodies and built structures")}
          />
          <ActionButton
            icon={<Zap className="w-3.5 h-3.5 text-amber-400" />}
            label="Bi-Temporal Change Detection"
            onClick={() => setValue("Detect structural changes between these two temporal satellite dates")}
          />
          <ActionButton
            icon={<Sparkles className="w-3.5 h-3.5 text-purple-400" />}
            label="Optical + SAR Cross Fusion"
            onClick={() => setValue("Fuse optical and SAR sensors for high-confidence scene classification")}
          />
          <ActionButton
            icon={<Compass className="w-3.5 h-3.5 text-rose-400" />}
            label="Scene Description"
            onClick={() => setValue("Generate a detailed scene description of this remote sensing landscape")}
          />
        </div>
      </div>
    </div>
  );
}

interface ActionButtonProps {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
}

function ActionButton({ icon, label, onClick }: ActionButtonProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center gap-2 px-3.5 py-1.5 bg-neutral-900/90 hover:bg-neutral-800/90 rounded-full border border-neutral-800/80 text-neutral-300 hover:text-white text-xs transition-all hover:scale-105 active:scale-95 shadow-md"
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
