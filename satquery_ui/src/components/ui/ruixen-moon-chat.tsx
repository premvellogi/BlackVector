/**
 * RuixenMoonChat — workspace prompt composer
 *
 * Uses the same AnimatedAIChat glass card that works in the hero,
 * wired to WorkspaceView's onNewMessage handler.
 */
import { AnimatedAIChat } from "@/components/ui/animated-ai-chat";

interface RuixenMoonChatProps {
  onSubmit?: (query: string, files: any[], task: string | null) => void;
  heroMode?: boolean;
}

export default function RuixenMoonChat({ onSubmit }: RuixenMoonChatProps) {
  return (
    <AnimatedAIChat
      placeholder="Ask about satellite imagery, change detection, or geospatial analysis…"
      onSubmit={(query, files) => onSubmit?.(query, files, null)}
    />
  );
}
