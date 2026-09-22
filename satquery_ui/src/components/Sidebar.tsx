import React, { useState } from "react";
import {
  Sidebar as SidebarPrimitive,
  SidebarBody,
  SidebarLink,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  Plus,
  MessageSquare,
  Settings,
  User,
  Satellite,
  Map,
  Layers,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

/* ─────────────────────────────────────────
   Types
───────────────────────────────────────── */
interface ChatHistoryItem {
  id: string;
  title: string;
  active?: boolean;
}

interface SidebarProps {
  onNewAnalysis: () => void;
  onSelectConversation?: (id: string) => void;
  chatHistory?: Record<string, ChatHistoryItem[]>;
  activeConversationId?: string;
}

const DEFAULT_HISTORY: Record<string, ChatHistoryItem[]> = {
  Today: [
    { id: "1", title: "Urban expansion — Delhi NCR" },
    { id: "2", title: "Flood mapping, Kerala coast" },
    { id: "3", title: "Forest change 2023 vs 2024" },
  ],
  Yesterday: [
    { id: "4", title: "Optical + SAR water bodies" },
    { id: "5", title: "Agricultural classification" },
  ],
};

/* ─────────────────────────────────────────
   Navigation Links — large 22px icons
───────────────────────────────────────── */
const NAV_LINKS = [
  {
    label: "Analysis",
    href: "#",
    icon: <Satellite className="text-white/50 h-[22px] w-[22px] flex-shrink-0" />,
  },
  {
    label: "Map View",
    href: "#",
    icon: <Map className="text-white/50 h-[22px] w-[22px] flex-shrink-0" />,
  },
  {
    label: "Layers",
    href: "#",
    icon: <Layers className="text-white/50 h-[22px] w-[22px] flex-shrink-0" />,
  },
  {
    label: "Settings",
    href: "#",
    icon: <Settings className="text-white/50 h-[22px] w-[22px] flex-shrink-0" />,
  },
];

/* ─────────────────────────────────────────
   Logo — expanded (large)
───────────────────────────────────────── */
const Logo = () => {
  return (
    <a
      href="#"
      onClick={(e) => e.preventDefault()}
      className="flex items-center gap-2.5 py-2 relative z-20"
    >
      <img
        src="/satquery-icon.png"
        alt="SatQuery AI"
        className="flex-shrink-0"
        style={{
          height: '28px',
          width: '28px',
          objectFit: 'contain',
          filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.5))',
          imageRendering: 'auto',
        }}
        draggable={false}
      />
      <span
        className="text-white/90 text-[15px] font-semibold tracking-tight whitespace-nowrap"
        style={{ textShadow: '0 1px 3px rgba(0,0,0,0.3)' }}
      >
        SatQuery AI
      </span>
    </a>
  );
};

/* ─────────────────────────────────────────
   Logo — collapsed (icon only, centered)
───────────────────────────────────────── */
const LogoIcon = () => {
  return (
    <a
      href="#"
      onClick={(e) => e.preventDefault()}
      className="flex items-center justify-center py-2 relative z-20"
    >
      <img
        src="/satquery-icon.png"
        alt="SatQuery AI"
        className="flex-shrink-0"
        style={{
          height: '24px',
          width: '24px',
          objectFit: 'contain',
          filter: 'drop-shadow(0 1px 3px rgba(0,0,0,0.5))',
          imageRendering: 'auto',
        }}
        draggable={false}
      />
    </a>
  );
};

/* ─────────────────────────────────────────
   New Analysis Button — big & prominent
───────────────────────────────────────── */
const NewAnalysisButton = ({ onClick }: { onClick: () => void }) => {
  const { open } = useSidebar();

  return (
    <motion.button
      onClick={onClick}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.97 }}
      className={cn(
        "flex items-center rounded-xl cursor-pointer",
        "text-white/70 hover:text-white transition-all duration-200",
        open
          ? "w-full gap-3 px-4 py-4"
          : "w-12 h-12 justify-center"
      )}
      style={{
        background: "rgba(255,255,255,0.06)",
        border: "1px solid rgba(255,255,255,0.10)",
      }}
      title="New Analysis"
    >
      <Plus className="w-[20px] h-[20px] flex-shrink-0" />
      <AnimatePresence>
        {open && (
          <motion.span
            initial={{ opacity: 0, width: 0 }}
            animate={{ opacity: 1, width: "auto" }}
            exit={{ opacity: 0, width: 0 }}
            className="text-[15px] font-medium whitespace-nowrap overflow-hidden"
          >
            New Analysis
          </motion.span>
        )}
      </AnimatePresence>
    </motion.button>
  );
};

/* ─────────────────────────────────────────
   Chat History — well-spaced list
───────────────────────────────────────── */
const ChatHistorySection = ({
  chatHistory,
  activeConversationId,
  onSelectConversation,
}: {
  chatHistory: Record<string, ChatHistoryItem[]>;
  activeConversationId?: string;
  onSelectConversation?: (id: string) => void;
}) => {
  const { open } = useSidebar();

  if (!open) return null;

  return (
    <div className="mt-4 flex flex-col">
      {/* Divider */}
      <div
        className="mb-4"
        style={{ height: "1px", background: "rgba(255,255,255,0.07)" }}
      />

      <div className="overflow-y-auto scrollbar-dark flex-1 pr-1">
        {Object.entries(chatHistory).map(([group, items]) => (
          <div key={group} className="mb-5">
            <AnimatePresence>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="px-3 pb-2 text-[11px] uppercase tracking-[0.14em] font-semibold"
                style={{ color: "rgba(255,255,255,0.25)" }}
              >
                {group}
              </motion.div>
            </AnimatePresence>

            <div className="flex flex-col gap-1.5">
              {items.map((item) => (
                <motion.button
                  key={item.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2 }}
                  onClick={() => onSelectConversation?.(item.id)}
                  className={cn(
                    "w-full flex items-center gap-3 px-3 py-3.5 rounded-xl text-left text-[14px]",
                    "transition-all duration-150 group cursor-pointer",
                    activeConversationId === item.id || item.active
                      ? "bg-white/[0.08] text-white/90"
                      : "text-white/45 hover:text-white/75 hover:bg-white/[0.04]"
                  )}
                >
                  <MessageSquare className="w-[18px] h-[18px] flex-shrink-0 opacity-50" />
                  <span className="truncate leading-snug">{item.title}</span>
                </motion.button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

/* ─────────────────────────────────────────
   Main Sidebar Export
───────────────────────────────────────── */
export const Sidebar: React.FC<SidebarProps> = ({
  onNewAnalysis,
  onSelectConversation,
  chatHistory = DEFAULT_HISTORY,
  activeConversationId,
}) => {
  const [open, setOpen] = useState(false);

  return (
    <SidebarPrimitive open={open} setOpen={setOpen}>
      <SidebarBody
        className="justify-between gap-8"
        style={{
          background: "rgba(8, 8, 8, 0.97)",
          borderRight: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        {/* ── Top section ── */}
        <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
          {/* Logo — large */}
          {open ? <Logo /> : <LogoIcon />}

          {/* New Analysis — spaced from logo */}
          <div className="mt-8">
            <NewAnalysisButton onClick={onNewAnalysis} />
          </div>

          {/* Nav links — spaced from button */}
          <div className="mt-8 flex flex-col gap-2">
            {NAV_LINKS.map((link, idx) => (
              <SidebarLink key={idx} link={link} />
            ))}
          </div>

          {/* Chat history */}
          <ChatHistorySection
            chatHistory={chatHistory}
            activeConversationId={activeConversationId}
            onSelectConversation={onSelectConversation}
          />
        </div>

        {/* ── Bottom: user profile — large avatar ── */}
        <div>
          {/* Divider */}
          <div
            className="mb-4"
            style={{ height: "1px", background: "rgba(255,255,255,0.07)" }}
          />
          <SidebarLink
            link={{
              label: "Analyst",
              href: "#",
              icon: (
                <div
                  className="h-9 w-9 flex-shrink-0 rounded-full flex items-center justify-center"
                  style={{
                    background: "rgba(255,255,255,0.08)",
                    border: "1px solid rgba(255,255,255,0.12)",
                  }}
                >
                  <User className="w-[18px] h-[18px] text-white/50" />
                </div>
              ),
            }}
          />
        </div>
      </SidebarBody>
    </SidebarPrimitive>
  );
};

export default Sidebar;
