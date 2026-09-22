import React from 'react'
import { motion } from 'framer-motion'
import { Navbar } from './ui/mini-navbar'
import { Typewriter } from './ui/typewriter-text'
import { AnimatedAIChat } from './ui/animated-ai-chat'
import { LogoMarquee } from './ui/logo-marquee'

/**
 * HeroSection
 * Layout (top → bottom, all centered):
 *   1. Navbar (floating, fixed)
 *   2. Typewriter heading — cycles both hero strings in Geist Pixel
 *   3. AnimatedAIChat composer — input + send + file upload
 *   4. Logo marquee (bottom)
 */

const HERO_FONT_STYLE: React.CSSProperties = {
  fontFamily: "'Geist Pixel', monospace",
  fontSize: 'clamp(1.5rem, 3.5vw, 2.5rem)',
  fontWeight: 600,
  color: 'rgba(255,255,255,0.96)',
  lineHeight: 1.35,
  letterSpacing: '0.01em',
  textAlign: 'center',
}

interface HeroSectionProps {
  onSubmit: (query: string, files: any[], task: string | null) => void
  githubUrl?: string
  onLoginClick?: () => void
}

export const HeroSection: React.FC<HeroSectionProps> = ({ onSubmit, githubUrl: _githubUrl, onLoginClick }) => {
  return (
    <div className="relative min-h-screen w-full flex flex-col overflow-x-hidden">
      {/*
        MotionSite background lives in App.tsx.
        This gradient is a legibility overlay only.
      */}
      <div
        className="absolute inset-0 pointer-events-none z-[1]"
        style={{
          background:
            'linear-gradient(to bottom, rgba(0,0,0,0.55) 0%, rgba(0,0,0,0.20) 40%, rgba(0,0,0,0.65) 100%)',
        }}
      />

      {/* Floating Navbar */}
      <Navbar githubUrl={_githubUrl} onLoginClick={onLoginClick} />

      {/* ── Center column: heading + composer ── */}
      <main
        className="relative z-10 flex-1 flex flex-col items-center justify-center px-4 gap-8"
        style={{ paddingTop: '80px', paddingBottom: '48px' }}
      >
        {/* Typewriter heading — single <h1>, no duplicates */}
        <div
          className="flex items-center justify-center text-center w-full"
          style={{ maxWidth: '820px', minHeight: '5rem' }}
        >
          <h1 style={HERO_FONT_STYLE}>
            <Typewriter
              text={[
                'Ask the Earth. See the Answer.',
                'Upload your satellite imagery',
              ]}
              speed={75}
              deleteSpeed={40}
              delay={2200}
              loop={true}
              cursor="|"
              style={HERO_FONT_STYLE}
            />
          </h1>
        </div>

        {/* AI Composer — input + send + file upload */}
        <div className="w-full" style={{ maxWidth: '680px' }}>
          <AnimatedAIChat
            placeholder="Describe what you want to analyze..."
            onSubmit={(query, files) => onSubmit(query, files, null)}
          />
        </div>
      </main>

      {/* ── Logo marquee — pinned full-width at bottom of hero ── */}
      <motion.div
        className="relative z-10 w-full"
        style={{ paddingBottom: 28, paddingTop: 8 }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.8, delay: 0.5, ease: 'easeOut' }}
      >
        <LogoMarquee />
      </motion.div>

    </div>
  )
}

export default HeroSection
