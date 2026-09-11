// @ts-nocheck
import * as React from "react"
import {
    motion,
    AnimatePresence,
} from "framer-motion"

// ─────────────────────────────────────────────
// Google Font loader — Hedvig Letters Serif
// ─────────────────────────────────────────────
function useHedvigFont() {
    React.useEffect(() => {
        if (typeof document === "undefined") return
        const id = "hedvig-letters-serif-link"
        if (document.getElementById(id)) return
        const link = document.createElement("link")
        link.id = id
        link.rel = "stylesheet"
        link.href =
            "https://fonts.googleapis.com/css2?family=Hedvig+Letters+Serif:opsz@12..24&display=swap"
        document.head.appendChild(link)
    }, [])
}

// ─────────────────────────────────────────────
// Phosphor Icons (Regular weight — inline SVG)
// ─────────────────────────────────────────────
const PhPlus = ({ size = 20, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M224,128a8,8,0,0,1-8,8H136v80a8,8,0,0,1-16,0V136H40a8,8,0,0,1,0-16h80V40a8,8,0,0,1,16,0v80h80A8,8,0,0,1,224,128Z" />
    </svg>
)
const PhMicrophone = ({ size = 20, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M128,176a48.05,48.05,0,0,0,48-48V80a48,48,0,0,0-96,0v48A48.05,48.05,0,0,0,128,176ZM96,80a32,32,0,0,1,64,0v48a32,32,0,0,1-64,0Zm40,120.26V224a8,8,0,0,1-16,0V200.26A80.11,80.11,0,0,1,48,120a8,8,0,0,1,16,0,64,64,0,0,0,128,0,8,8,0,0,1,16,0A80.11,80.11,0,0,1,136,200.26Z" />
    </svg>
)
const PhArrowUp = ({ size = 20, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M205.66,117.66a8,8,0,0,1-11.32,0L136,59.31V216a8,8,0,0,1-16,0V59.31L61.66,117.66a8,8,0,0,1-11.32-11.32l72-72a8,8,0,0,1,11.32,0l72,72A8,8,0,0,1,205.66,117.66Z" />
    </svg>
)
const PhSlidersHorizontal = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M40,88H73a32,32,0,0,0,62,0H216a8,8,0,0,0,0-16H135a32,32,0,0,0-62,0H40a8,8,0,0,0,0,16Zm64-24A16,16,0,1,1,88,80,16,16,0,0,1,104,64ZM216,168H199a32,32,0,0,0-62,0H40a8,8,0,0,0,0,16H137a32,32,0,0,0,62,0h17a8,8,0,0,0,0-16Zm-48,24a16,16,0,1,1,16-16A16,16,0,0,1,168,192Z" />
    </svg>
)
const PhCode = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M69.12,94.15,28.5,128l40.62,33.85a8,8,0,1,1-10.24,12.29l-48-40a8,8,0,0,1,0-12.29l48-40a8,8,0,0,1,10.24,12.3Zm176,27.7-48-40a8,8,0,1,0-10.24,12.3L227.5,128l-40.62,33.85a8,8,0,1,0,10.24,12.29l48-40a8,8,0,0,0,0-12.29ZM162.73,32.48a8,8,0,0,0-10.25,4.79l-64,176a8,8,0,0,0,4.79,10.26A8.14,8.14,0,0,0,96,224a8,8,0,0,0,7.52-5.27l64-176A8,8,0,0,0,162.73,32.48Z" />
    </svg>
)
const PhPencilSimple = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M227.31,73.37,182.63,28.68a16,16,0,0,0-22.63,0L36.69,152A15.86,15.86,0,0,0,32,163.31V208a16,16,0,0,0,16,16H92.69A15.86,15.86,0,0,0,104,219.31L227.31,96a16,16,0,0,0,0-22.63ZM92.69,208H48V163.31l88-88L180.69,120ZM192,108.68,147.31,64l24-24L216,84.68Z" />
    </svg>
)
const PhGraduationCap = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M251.76,88.94l-120-64a8,8,0,0,0-7.52,0l-120,64a8,8,0,0,0,0,14.12L32,117.87v48.42a15.91,15.91,0,0,0,4.06,10.65C49.16,191.53,78.51,216,128,216a130,130,0,0,0,48-8.76V240a8,8,0,0,0,16,0V199.51a115.63,115.63,0,0,0,27.94-21.57A15.91,15.91,0,0,0,224,167.29V117.87l27.76-14.81a8,8,0,0,0,0-14.12ZM128,200c-43.27,0-68.72-21.14-80-33.71V126.4l76.24,40.67a8,8,0,0,0,7.52,0L208,126.4v39.89C196.72,178.86,171.27,200,128,200Zm88-32.71ZM128,152.57,37.67,96,128,39.43,218.33,96Z" />
    </svg>
)
const PhChatTeardrop = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M132,24A100.11,100.11,0,0,0,32,124v84a16,16,0,0,0,16,16h84a100,100,0,0,0,0-200Zm0,184H48V124a84,84,0,1,1,84,84Z" />
    </svg>
)
const PhLightbulb = ({ size = 18, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M176,232a8,8,0,0,1-8,8H88a8,8,0,0,1,0-16h80A8,8,0,0,1,176,232Zm40-128a87.55,87.55,0,0,1-33.64,69.21A16.24,16.24,0,0,0,176,186v6a16,16,0,0,1-16,16H96a16,16,0,0,1-16-16v-6a16,16,0,0,0-6.23-12.66A87.59,87.59,0,0,1,40,104.49C39.74,56.83,78.26,17.14,125.88,16A88,88,0,0,1,216,104Zm-16,0a72,72,0,0,0-73.74-72c-39,.92-70.47,33.39-70.26,72.39a71.65,71.65,0,0,0,27.64,56.3A32,32,0,0,1,96,186v6h64v-6a32.15,32.15,0,0,1,12.47-25.35A71.65,71.65,0,0,0,200,104Z" />
    </svg>
)
const PhWaveform = ({ size = 20, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M24,128a8,8,0,0,1,8-8H56a8,8,0,0,1,0,16H32A8,8,0,0,1,24,128Zm184-8H192a8,8,0,0,0,0,16h16a8,8,0,0,0,0-16Zm-96-56a8,8,0,0,0-8,8V184a8,8,0,0,0,16,0V72A8,8,0,0,0,112,64Zm-40,24a8,8,0,0,0-8,8V160a8,8,0,0,0,16,0V96A8,8,0,0,0,72,88Zm80,0a8,8,0,0,0-8,8V160a8,8,0,0,0,16,0V96A8,8,0,0,0,152,88Zm40,16a8,8,0,0,0-8,8v32a8,8,0,0,0,16,0V112A8,8,0,0,0,192,104ZM32,104a8,8,0,0,0-8,8v32a8,8,0,0,0,16,0V112A8,8,0,0,0,32,104Z" />
    </svg>
)
const PhCaretDown = ({ size = 12, color = "currentColor" }) => (
    <svg width={size} height={size} viewBox="0 0 256 256" fill={color}>
        <path d="M213.66,101.66l-80,80a8,8,0,0,1-11.32,0l-80-80A8,8,0,0,1,53.66,90.34L128,164.69l74.34-74.35a8,8,0,0,1,11.32,11.32Z" />
    </svg>
)

// ─────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────
function parseCssColor(color: string): [number, number, number] | null {
    if (!color) return null
    const s = color.trim()
    const hex6 = s.match(/^#([0-9a-f]{6})([0-9a-f]{2})?$/i)
    if (hex6)
        return [
            parseInt(hex6[1].slice(0, 2), 16),
            parseInt(hex6[1].slice(2, 4), 16),
            parseInt(hex6[1].slice(4, 6), 16),
        ]
    const hex3 = s.match(/^#([0-9a-f]{3})$/i)
    if (hex3)
        return [
            parseInt(hex3[1][0] + hex3[1][0], 16),
            parseInt(hex3[1][1] + hex3[1][1], 16),
            parseInt(hex3[1][2] + hex3[1][2], 16),
        ]
    const rgb = s.match(/rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/)
    if (rgb) return [parseFloat(rgb[1]), parseFloat(rgb[2]), parseFloat(rgb[3])]
    const hsl = s.match(/hsla?\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%/)
    if (hsl) {
        const h = parseFloat(hsl[1]) / 360,
            sat = parseFloat(hsl[2]) / 100,
            l = parseFloat(hsl[3]) / 100
        if (sat === 0) {
            const v = Math.round(l * 255)
            return [v, v, v]
        }
        const hue2rgb = (p: number, q: number, t: number) => {
            if (t < 0) t += 1
            if (t > 1) t -= 1
            if (t < 1 / 6) return p + (q - p) * 6 * t
            if (t < 1 / 2) return q
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6
            return p
        }
        const q = l < 0.5 ? l * (1 + sat) : l + sat - l * sat,
            p = 2 * l - q
        return [
            Math.round(hue2rgb(p, q, h + 1 / 3) * 255),
            Math.round(hue2rgb(p, q, h) * 255),
            Math.round(hue2rgb(p, q, h - 1 / 3) * 255),
        ]
    }
    return null
}

function contrastIconColor(bgColor: string, fallback: string): string {
    const rgb = parseCssColor(bgColor)
    if (!rgb) return fallback
    const [r, g, b] = rgb
    const toLinear = (c: number) => {
        const n = c / 255
        return n <= 0.03928 ? n / 12.92 : Math.pow((n + 0.055) / 1.055, 2.4)
    }
    const lum =
        0.2126 * toLinear(r) + 0.7152 * toLinear(g) + 0.0722 * toLinear(b)
    return lum > 0.179 ? "#000000" : "#ffffff"
}

// ─────────────────────────────────────────────
// Ellipsis loader — used for all interfaces
// ─────────────────────────────────────────────
const EllipsisLoader = ({ color }: { color: string }) => {
    const dotVariants = {
        animate: (i: number) => ({
            scale: [1, 1.2, 1],
            opacity: [0.2, 0.8, 0.2],
            transition: {
                duration: 1.06,
                repeat: Infinity,
                ease: "easeInOut",
                delay: i * 0.26,
            },
        }),
    }
    return (
        <div
            style={{
                display: "flex",
                gap: "8px",
                alignItems: "center",
                justifyContent: "center",
            }}
        >
            {[0, 1, 2].map((i) => (
                <motion.div
                    key={i}
                    custom={i}
                    variants={dotVariants}
                    animate="animate"
                    style={{
                        width: "12px",
                        height: "12px",
                        backgroundColor: color,
                        borderRadius: "50%",
                    }}
                />
            ))}
        </div>
    )
}

// ─────────────────────────────────────────────
// Label row (Claude) — renamed from Chips
// ─────────────────────────────────────────────
const CLAUDE_LABELS = [
    { Icon: PhCode, label: "Code" },
    { Icon: PhPencilSimple, label: "Write" },
    { Icon: PhGraduationCap, label: "Learn" },
    { Icon: PhChatTeardrop, label: "Personal matters" },
    { Icon: PhLightbulb, label: "Claude's choice" },
]

function ClaudeLabels({
    colors,
    fontFamilyOnly,
}: {
    colors: any
    fontFamilyOnly: string
}) {
    return (
        <div
            style={{
                display: "flex",
                flexWrap: "wrap",
                gap: "8px",
                justifyContent: "center",
                marginTop: "16px",
                maxWidth: "672px",
            }}
        >
            {CLAUDE_LABELS.map(({ Icon, label }) => (
                <button
                    key={label}
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "7px",
                        padding: "9px 16px",
                        borderRadius: "12px",
                        backgroundColor: colors.chipBg,
                        border: `1px solid ${colors.border}`,
                        color: colors.secondaryText,
                        fontSize: "13px",
                        fontFamily: fontFamilyOnly,
                        cursor: "pointer",
                        fontWeight: 500,
                    }}
                >
                    <Icon size={16} color={colors.secondaryText} />
                    <span>{label}</span>
                </button>
            ))}
        </div>
    )
}

// ─────────────────────────────────────────────
// Layout: ChatGPT
// ─────────────────────────────────────────────
function LayoutChatGPT({
    colors,
    fontFamilyOnly,
    text,
    setText,
    handleSend,
    placeholderText,
    showTooltip,
    setShowTooltip,
    resolvedActiveColor,
    resolvedButtonRadius,
    resolvedActiveIconColor,
}: any) {
    const hasText = text.length > 0
    return (
        <div
            style={{
                ...s.inputBox,
                backgroundColor: colors.inputBg,
                border: `1px solid ${colors.border}`,
            }}
        >
            <textarea
                placeholder={placeholderText}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) =>
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    (e.preventDefault(), handleSend())
                }
                style={{
                    ...s.textarea,
                    color: colors.text,
                    fontFamily: fontFamilyOnly,
                }}
            />
            <div style={s.toolbarContainer}>
                <div style={s.toolbarLeft}>
                    <button
                        style={{ ...s.iconButton, color: colors.secondaryText }}
                    >
                        <PhPlus size={20} />
                    </button>
                    <button
                        style={{
                            ...s.toolsButton,
                            color: colors.secondaryText,
                            border: `1px solid ${colors.border}`,
                        }}
                    >
                        <PhSlidersHorizontal size={18} />
                        <span
                            style={{
                                fontSize: "14px",
                                fontWeight: 500,
                                fontFamily: fontFamilyOnly,
                            }}
                        >
                            Tools
                        </span>
                    </button>
                </div>
                <div style={s.toolbarRight}>
                    <div
                        style={{
                            position: "relative",
                            display: "flex",
                            justifyContent: "center",
                        }}
                        onMouseEnter={() => setShowTooltip(true)}
                        onMouseLeave={() => setShowTooltip(false)}
                    >
                        <AnimatePresence>
                            {showTooltip && (
                                <motion.div
                                    initial={{ opacity: 0, y: 5 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0 }}
                                    style={{
                                        ...s.tooltip,
                                        backgroundColor: colors.tooltipBg,
                                        color: colors.tooltipText,
                                        fontFamily: fontFamilyOnly,
                                    }}
                                >
                                    Voice Message
                                </motion.div>
                            )}
                        </AnimatePresence>
                        <button
                            style={{
                                ...s.iconButton,
                                color: colors.secondaryText,
                            }}
                        >
                            <PhMicrophone size={20} />
                        </button>
                    </div>
                    <AnimatePresence>
                        {hasText && (
                            <motion.button
                                onClick={handleSend}
                                initial={{ opacity: 0, scale: 0.7 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.7 }}
                                transition={{ duration: 0.18 }}
                                style={{
                                    ...s.submitButton,
                                    backgroundColor: resolvedActiveColor,
                                    color: resolvedActiveIconColor,
                                    borderRadius: resolvedButtonRadius,
                                }}
                            >
                                <PhArrowUp size={20} />
                            </motion.button>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    )
}

// ─────────────────────────────────────────────
// Layout: Claude
// ─────────────────────────────────────────────
function LayoutClaude({
    colors,
    fontFamilyOnly,
    text,
    setText,
    handleSend,
    placeholderText,
    resolvedActiveColor,
    resolvedButtonRadius,
    resolvedActiveIconColor,
}: any) {
    const hasText = text.length > 0
    // Claude always uses Hedvig Letters Serif
    const claudeFont = "'Hedvig Letters Serif', serif"
    return (
        <div
            style={{
                width: "100%",
                maxWidth: "672px",
                borderRadius: "20px",
                backgroundColor: colors.inputBg,
                border: `1px solid ${colors.border}`,
                padding: "14px 16px",
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                boxSizing: "border-box",
                boxShadow: "0 20px 40px -12px rgba(0,0,0,0.12)",
            }}
        >
            <textarea
                placeholder={placeholderText}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) =>
                    e.key === "Enter" &&
                    !e.shiftKey &&
                    (e.preventDefault(), handleSend())
                }
                style={{
                    ...s.textarea,
                    color: colors.text,
                    // Force Hedvig Letters Serif inside the Claude input
                    fontFamily: claudeFont,
                    minHeight: "52px",
                }}
            />
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                }}
            >
                {/* Left: plus */}
                <button
                    style={{ ...s.iconButton, color: colors.secondaryText }}
                >
                    <PhPlus size={20} />
                </button>
                {/* Right: model label + mic + waveform + send */}
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "8px",
                    }}
                >
                    <div
                        style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "4px",
                            color: colors.secondaryText,
                            fontSize: "13px",
                            fontFamily: claudeFont,
                            cursor: "pointer",
                            padding: "4px 6px",
                        }}
                    >
                        <span>Sonnet 4.6</span>
                        <PhCaretDown size={12} />
                        <span style={{ marginLeft: "2px" }}>Low</span>
                        <PhCaretDown size={12} />
                    </div>
                    <button
                        style={{ ...s.iconButton, color: colors.secondaryText }}
                    >
                        <PhMicrophone size={20} />
                    </button>
                    <button
                        style={{ ...s.iconButton, color: colors.secondaryText }}
                    >
                        <PhWaveform size={20} />
                    </button>
                    {/* Send appears alongside mic, not replacing it */}
                    <AnimatePresence>
                        {hasText && (
                            <motion.button
                                onClick={handleSend}
                                initial={{ opacity: 0, scale: 0.7 }}
                                animate={{ opacity: 1, scale: 1 }}
                                exit={{ opacity: 0, scale: 0.7 }}
                                transition={{ duration: 0.18 }}
                                style={{
                                    ...s.submitButton,
                                    backgroundColor: resolvedActiveColor,
                                    color: resolvedActiveIconColor,
                                    borderRadius: resolvedButtonRadius,
                                }}
                            >
                                <PhArrowUp size={20} />
                            </motion.button>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </div>
    )
}

// ─────────────────────────────────────────────
// Layout: AIGemini — send button appears NEXT TO mic, never replacing it
// ─────────────────────────────────────────────
function LayoutAIGemini({
    colors,
    fontFamilyOnly,
    text,
    setText,
    handleSend,
    placeholderText,
    resolvedActiveColor,
    resolvedButtonRadius,
    resolvedActiveIconColor,
}: any) {
    const hasText = text.length > 0
    return (
        <div
            style={{
                width: "100%",
                maxWidth: "640px",
                borderRadius: "9999px",
                backgroundColor: colors.inputBg,
                border: `1px solid ${colors.border}`,
                padding: "0 16px 0 20px",
                display: "flex",
                alignItems: "center",
                gap: "10px",
                height: "60px",
                boxSizing: "border-box",
                boxShadow: "0 8px 32px rgba(0,0,0,0.18)",
            }}
        >
            <button
                style={{
                    ...s.iconButton,
                    color: colors.secondaryText,
                    padding: "6px",
                    flexShrink: 0,
                }}
            >
                <PhPlus size={20} />
            </button>
            <input
                type="text"
                placeholder={placeholderText}
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                style={{
                    flex: 1,
                    background: "transparent",
                    border: "none",
                    outline: "none",
                    color: colors.text,
                    fontSize: "16px",
                    fontFamily: fontFamilyOnly,
                    minWidth: 0,
                }}
            />
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    flexShrink: 0,
                }}
            >
                <div
                    style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "4px",
                        color: colors.secondaryText,
                        fontSize: "13px",
                        fontFamily: fontFamilyOnly,
                        cursor: "pointer",
                    }}
                >
                    <span>Pro</span>
                    <PhCaretDown size={12} />
                </div>
                {/* Mic always visible; send button appears alongside it when typing */}
                <button
                    style={{ ...s.iconButton, color: colors.secondaryText }}
                >
                    <PhMicrophone size={20} />
                </button>
                <AnimatePresence>
                    {hasText && (
                        <motion.button
                            key="send"
                            onClick={handleSend}
                            initial={{ opacity: 0, scale: 0.7 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.7 }}
                            transition={{ duration: 0.18 }}
                            style={{
                                ...s.submitButton,
                                backgroundColor: resolvedActiveColor,
                                color: resolvedActiveIconColor,
                                borderRadius: resolvedButtonRadius,
                                padding: "8px",
                            }}
                        >
                            <PhArrowUp size={20} />
                        </motion.button>
                    )}
                </AnimatePresence>
            </div>
        </div>
    )
}

// ─────────────────────────────────────────────
// Main Component
// ─────────────────────────────────────────────
/**
 * AIChatPrompt
 * @framerSupportedLayoutWidth any
 * @framerSupportedLayoutHeight any
 * @framerIntrinsicWidth 800
 * @framerIntrinsicHeight 300
 */
export default function AIChatPrompt(props: any) {
    const {
        interfaceStyle = "Claude",
        titleText,
        titleText2,
        placeholderText,
        textEffect,
        answerText,
        font,
        theme,
        submitButtonRadius,
        submitButtonActiveColor,
        glowColor,
        showLabels = true,
        style,
    } = props

    useHedvigFont()

    const [text, setText] = React.useState("")
    const [showTooltip, setShowTooltip] = React.useState(false)
    const [currentTitleIndex, setCurrentTitleIndex] = React.useState(0)
    const [viewState, setViewState] = React.useState<
        "input" | "loading" | "answer"
    >("input")

    // ── Font ───────────────────────────────────────────────────────────────
    const fontStyles = typeof font === "object" ? font : {}
    const isClaudeStyle = interfaceStyle === "Claude"

    // Claude title always uses Hedvig; others fall back to system-ui or the font prop
    const titleFontFamily = isClaudeStyle
        ? "'Hedvig Letters Serif', serif"
        : fontStyles.fontFamily || "system-ui, -apple-system, sans-serif"

    // General font family for non-title text (input toolbar, chips, etc.)
    const fontFamilyOnly = isClaudeStyle
        ? "'Hedvig Letters Serif', serif"
        : fontStyles.fontFamily || "system-ui, -apple-system, sans-serif"

    // ── Theme ──────────────────────────────────────────────────────────────
    const isDark = theme === "dark"
    const colors = {
        text: isDark ? "#ffffff" : "#000000",
        secondaryText: isDark ? "#9ca3af" : "#6b7280",
        inputBg: isDark ? "#1e1e1e" : "#ffffff",
        border: isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)",
        submitBg: isDark ? "rgba(255,255,255,0.1)" : "rgba(0,0,0,0.05)",
        tooltipBg: isDark ? "#ffffff" : "#000000",
        tooltipText: isDark ? "#000000" : "#ffffff",
        chipBg: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.04)",
    }

    // ── Submit button ───────────────────────────────────────────────────────
    const resolvedActiveColor =
        submitButtonActiveColor || (isDark ? "#ffffff" : "#000000")
    const resolvedActiveIconColor = contrastIconColor(
        resolvedActiveColor,
        isDark ? "#000000" : "#ffffff"
    )
    const resolvedButtonRadius =
        submitButtonRadius !== undefined ? `${submitButtonRadius}px` : "50%"

    // ── Glow (AIGemini) ─────────────────────────────────────────────────────
    const resolvedGlowColor = glowColor || "#121A3A"
    const glowRgb = parseCssColor(resolvedGlowColor)
    const glowRgbStr = glowRgb
        ? `${glowRgb[0]},${glowRgb[1]},${glowRgb[2]}`
        : "18,26,58"

    // ── Title cycling ───────────────────────────────────────────────────────
    React.useEffect(() => {
        if (viewState !== "input" || !titleText2) return
        const currentString = currentTitleIndex === 0 ? titleText : titleText2
        const animDuration =
            textEffect === "typing" || textEffect === "slide"
                ? currentString.length * 40 + 400
                : 800
        const timer = setTimeout(
            () => setCurrentTitleIndex((p) => (p === 0 ? 1 : 0)),
            animDuration + 2500
        )
        return () => clearTimeout(timer)
    }, [currentTitleIndex, titleText, titleText2, textEffect, viewState])

    // ── Reset after answer ──────────────────────────────────────────────────
    React.useEffect(() => {
        if (viewState !== "answer") return
        const t = setTimeout(() => {
            setViewState("input")
            setText("")
            setCurrentTitleIndex(0)
        }, 5000)
        return () => clearTimeout(t)
    }, [viewState])

    const handleSend = () => {
        if (text.trim().length > 0) {
            setViewState("loading")
            setTimeout(() => setViewState("answer"), 2500)
        }
    }

    // ── Title renderer ──────────────────────────────────────────────────────
    const renderTitle = () => {
        const currentString = currentTitleIndex === 0 ? titleText : titleText2
        const titleStyle: React.CSSProperties = {
            ...s.title,
            color: colors.text,
            ...fontStyles,
            fontFamily: titleFontFamily,
        }
        if (textEffect === "fade" || textEffect === "none") {
            return (
                <motion.h1
                    key={currentTitleIndex}
                    initial={{ opacity: 0, y: textEffect === "none" ? 0 : 5 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: textEffect === "none" ? 0 : -5 }}
                    transition={{ duration: 0.6 }}
                    style={titleStyle}
                >
                    {currentString}
                </motion.h1>
            )
        }
        const isSlide = textEffect === "slide"
        return (
            <motion.h1
                key={currentTitleIndex}
                exit={{ opacity: 0, y: -10 }}
                style={titleStyle}
            >
                {currentString.split("").map((char: string, i: number) => (
                    <motion.span
                        key={i}
                        initial={{ opacity: 0, y: isSlide ? 20 : 0 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{
                            delay: i * 0.04,
                            duration: isSlide ? 0.4 : 0.2,
                        }}
                        style={{
                            display: "inline-block",
                            whiteSpace: char === " " ? "pre" : "normal",
                        }}
                    >
                        {char}
                    </motion.span>
                ))}
            </motion.h1>
        )
    }

    // ── Shared props for all layout components ──────────────────────────────
    const sharedInputProps = {
        colors,
        fontFamilyOnly,
        text,
        setText,
        handleSend,
        placeholderText,
        showTooltip,
        setShowTooltip,
        resolvedActiveColor,
        resolvedButtonRadius,
        resolvedActiveIconColor,
    }

    // ── Outer bg (only AIGemini) ────────────────────────────────────────────
    const containerBg =
        interfaceStyle === "AIGemini"
            ? `radial-gradient(ellipse 70% 60% at 50% 60%, rgba(${glowRgbStr},0.55) 0%, rgba(${glowRgbStr},0.18) 40%, transparent 75%)`
            : "transparent"

    return (
        <div
            style={{
                ...s.container,
                fontFamily: fontFamilyOnly,
                background: containerBg,
                ...style,
            }}
        >
            <AnimatePresence mode="wait">
                {viewState === "input" && (
                    <motion.div
                        key="input"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0, filter: "blur(4px)" }}
                        style={s.fullWidthWrapper}
                    >
                        {/* Title */}
                        <div style={s.titleWrapper}>
                            <AnimatePresence mode="wait">
                                {renderTitle()}
                            </AnimatePresence>
                        </div>

                        {/* Input layout */}
                        {interfaceStyle === "ChatGPT" && (
                            <LayoutChatGPT {...sharedInputProps} />
                        )}
                        {interfaceStyle === "Claude" && (
                            <LayoutClaude {...sharedInputProps} />
                        )}
                        {interfaceStyle === "AIGemini" && (
                            <LayoutAIGemini {...sharedInputProps} />
                        )}

                        {/* Labels — Claude only, controlled by prop */}
                        {interfaceStyle === "Claude" && showLabels && (
                            <ClaudeLabels
                                colors={colors}
                                fontFamilyOnly={fontFamilyOnly}
                            />
                        )}
                    </motion.div>
                )}

                {viewState === "loading" && (
                    <motion.div
                        key="loading"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        style={s.centerWrapper}
                    >
                        {/* Unified ellipsis loader for all interfaces */}
                        <EllipsisLoader color={colors.text} />
                    </motion.div>
                )}

                {viewState === "answer" && (
                    <motion.div
                        key="answer"
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, filter: "blur(4px)" }}
                        style={s.centerWrapper}
                    >
                        <h1
                            style={{
                                ...s.title,
                                color: colors.text,
                                ...fontStyles,
                                fontFamily: titleFontFamily,
                            }}
                        >
                            {answerText}
                        </h1>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}

// ─────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────
const s: Record<string, React.CSSProperties> = {
    container: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        padding: "16px",
        boxSizing: "border-box",
    },
    fullWidthWrapper: {
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        width: "100%",
    },
    centerWrapper: {
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        minHeight: "200px",
    },
    titleWrapper: {
        height: "60px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "40px",
        overflow: "visible",
    },
    title: { margin: 0, textAlign: "center" },
    inputBox: {
        width: "100%",
        maxWidth: "672px",
        borderRadius: "32px",
        padding: "16px",
        display: "flex",
        flexDirection: "column",
        gap: "8px",
        boxShadow: "0 25px 50px -12px rgba(0,0,0,0.10)",
        boxSizing: "border-box",
    },
    textarea: {
        backgroundColor: "transparent",
        border: "none",
        outline: "none",
        resize: "none",
        width: "100%",
        minHeight: "60px",
        fontSize: "18px",
        padding: "8px",
        boxSizing: "border-box",
    },
    toolbarContainer: {
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        marginTop: "8px",
    },
    toolbarLeft: { display: "flex", alignItems: "center", gap: "12px" },
    toolbarRight: { display: "flex", alignItems: "center", gap: "12px" },
    iconButton: {
        background: "none",
        border: "none",
        padding: "8px",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
    },
    toolsButton: {
        display: "flex",
        alignItems: "center",
        gap: "8px",
        padding: "6px 12px",
        backgroundColor: "transparent",
        borderRadius: "9999px",
        cursor: "pointer",
    },
    submitButton: {
        padding: "10px",
        border: "none",
        cursor: "pointer",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        transition: "background 0.2s",
    },
    tooltip: {
        position: "absolute",
        bottom: "100%",
        marginBottom: "8px",
        padding: "6px 12px",
        borderRadius: "6px",
        fontSize: "12px",
        fontWeight: 500,
        whiteSpace: "nowrap",
        zIndex: 50,
        pointerEvents: "none",
    },
}

// ─────────────────────────────────────────────
// Property Controls
// ─────────────────────────────────────────────
AIChatPrompt.defaultProps = {
    interfaceStyle: "Claude",
    titleText: "How Can I Help You?",
    titleText2: "What are we building today?",
    placeholderText: "Message...",
    textEffect: "typing",
    answerText: "Here is your generated layout!",
    theme: "dark",
    submitButtonRadius: 50,
    submitButtonActiveColor: "",
    glowColor: "#121A3A",
}

if (typeof addPropertyControls === "function") {
addPropertyControls(AIChatPrompt, {
    interfaceStyle: {
        type: ControlType?.Enum || "Enum",
        title: "Interface",
        options: ["ChatGPT", "Claude", "AIGemini"],
        optionTitles: ["ChatGPT", "Claude", "AI Gemini"],
        description: "Choose which AI chat interface style to render.",
        defaultValue: "Claude",
    },
    theme: {
        type: ControlType?.Enum || "Enum",
        title: "Theme",
        options: ["dark", "light"],
        optionTitles: ["🌙 Dark", "☀️ Light"],
        displaySegmentedControl: true,
        defaultValue: "dark",
    },
})
}
