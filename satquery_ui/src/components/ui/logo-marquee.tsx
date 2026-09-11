import * as React from "react";
import { SiPytorch, SiPython, SiFastapi } from "react-icons/si";
import { MdSatelliteAlt } from "react-icons/md";

/* ─────────────────────────────────────────────────────────
   Logo definitions — react-icons + styled text for others
───────────────────────────────────────────────────────── */
interface LogoItem {
  icon: React.ReactNode;
  label: string;
  href: string;
  iconColor?: string;
}

const LOGOS: LogoItem[] = [
  {
    icon: <SiPytorch size={32} />,
    label: "PyTorch",
    href: "https://pytorch.org",
    iconColor: "#ee4c2c",
  },
  {
    icon: (
      /* Rasterio — custom grid-square SVG matching their mark */
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="1.5" y="1.5" width="8.5" height="8.5" rx="1.2" stroke="currentColor" strokeWidth="1.8"/>
        <rect x="12" y="1.5" width="8.5" height="8.5" rx="1.2" stroke="currentColor" strokeWidth="1.8"/>
        <rect x="1.5" y="12" width="8.5" height="8.5" rx="1.2" stroke="currentColor" strokeWidth="1.8"/>
        <rect x="12" y="12" width="8.5" height="8.5" rx="1.2" stroke="currentColor" strokeWidth="1.8"/>
      </svg>
    ),
    label: "rasterio",
    href: "https://rasterio.readthedocs.io",
  },
  {
    icon: <SiFastapi size={32} />,
    label: "FastAPI",
    href: "https://fastapi.tiangolo.com",
    iconColor: "#009688",
  },
  {
    icon: (
      /* InternVL — layered-diamond mark matching their logo */
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="12,2 22,12 12,22 2,12" stroke="currentColor" strokeWidth="1.8" fill="none"/>
        <polygon points="12,6 18,12 12,18 6,12" stroke="currentColor" strokeWidth="1.4" fill="none" opacity="0.6"/>
        <circle cx="12" cy="12" r="2" fill="currentColor" opacity="0.9"/>
      </svg>
    ),
    label: "InternVL 2.5",
    href: "https://github.com/OpenGVLab/InternVL",
  },
  {
    icon: <SiPython size={32} />,
    label: "Python",
    href: "https://python.org",
    iconColor: "#3776ab",
  },
  {
    icon: <MdSatelliteAlt size={32} />,
    label: "BigEarth",
    href: "https://bigearth.net",
  },
];

/* Triple for seamless full-width infinite loop */
const TRACK = [...LOGOS, ...LOGOS, ...LOGOS];

/* ─────────────────────────────────────────────────────────
   LogoMarquee — full-width, CSS-keyframe scroll
───────────────────────────────────────────────────────── */
export function LogoMarquee() {
  return (
    <div className="w-full" style={{ fontFamily: "Inter, sans-serif" }}>
      {/* "Made With" label */}
      <p
        className="text-center uppercase tracking-[0.22em] text-[10px]"
        style={{ color: "rgba(255,255,255,0.22)", marginBottom: 40 }}
      >
        Made With
      </p>

      {/* Scrolling strip — full-width with fade edges */}
      <div
        className="relative w-full overflow-hidden"
        style={{
          maskImage:
            "linear-gradient(to right, transparent 0%, black 8%, black 92%, transparent 100%)",
          WebkitMaskImage:
            "linear-gradient(to right, transparent 0%, black 8%, black 92%, transparent 100%)",
        }}
      >
        <div
          className="flex items-center"
          style={{
            width: "max-content",
            animation: "marquee-scroll 28s linear infinite",
            gap: 96,
            padding: "4px 0",
          }}
        >
          {TRACK.map((logo, i) => (
            <a
              key={i}
              href={logo.href}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2.5 shrink-0"
              style={{
                textDecoration: "none",
                color: "rgba(255,255,255,0.4)",
                transition: "color 0.25s ease",
              }}
              onMouseEnter={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.85)")}
              onMouseLeave={(e) => (e.currentTarget.style.color = "rgba(255,255,255,0.4)")}
            >
              {/* Icon — inherits color from parent <a> */}
              <span style={{ lineHeight: 0, display: "flex", alignItems: "center" }}>
                {logo.icon}
              </span>

              {/* Label */}
              <span
                style={{
                  fontSize: 18,
                  fontWeight: 500,
                  letterSpacing: "-0.01em",
                  whiteSpace: "nowrap",
                }}
              >
                {logo.label}
              </span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}

export default LogoMarquee;
