import React, { useState, useEffect, useRef } from 'react';
import { useAuth } from '../auth/AuthProvider';
import { useAuthModal } from '../auth/AuthModalContext';

/* ─────────────────────────────────────────────────────────────
   Animated Nav Link — slide-up on hover
───────────────────────────────────────────────────────────── */
const AnimatedNavLink = ({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) => (
  <a
    href={href}
    className="group"
    style={{
      display: 'inline-block',
      height: 20,
      overflow: 'hidden',
      position: 'relative',
      textDecoration: 'none',
    }}
  >
    <div
      className="group-hover:-translate-y-1/2"
      style={{
        display: 'flex',
        flexDirection: 'column',
        transition: 'transform 0.3s ease-out',
      }}
    >
      <span style={{ display: 'block', height: 20, lineHeight: '20px', fontSize: 14, color: '#9ca3af', whiteSpace: 'nowrap' }}>{children}</span>
      <span style={{ display: 'block', height: 20, lineHeight: '20px', fontSize: 14, color: '#ffffff', whiteSpace: 'nowrap' }}>{children}</span>
    </div>
  </a>
);

/* ─────────────────────────────────────────────────────────────
   Navbar
───────────────────────────────────────────────────────────── */
interface NavbarProps {
  githubUrl?: string;
  onLoginClick?: () => void;
}

export function Navbar({ githubUrl = 'https://github.com', onLoginClick }: NavbarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [borderRadius, setBorderRadius] = useState('9999px');
  const shapeTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { user, signOut } = useAuth();
  const { openAuthModal } = useAuthModal();

  const userName =
    user?.user_metadata?.full_name ||
    user?.user_metadata?.name ||
    user?.email?.split('@')[0] ||
    'User';

  const userInitials = userName
    .split(' ')
    .filter(Boolean)
    .map((p: string) => p[0])
    .slice(0, 2)
    .join('')
    .toUpperCase() || 'U';

  useEffect(() => {
    if (shapeTimeoutRef.current) clearTimeout(shapeTimeoutRef.current);
    if (isOpen) {
      setBorderRadius('16px');
    } else {
      shapeTimeoutRef.current = setTimeout(() => setBorderRadius('9999px'), 300);
    }
    return () => { if (shapeTimeoutRef.current) clearTimeout(shapeTimeoutRef.current); };
  }, [isOpen]);

  const navLinks = [
    { label: 'Home',         href: '#' },
    { label: 'Features',     href: '#features' },
    { label: 'How It Works', href: '#how-it-works' },
    { label: 'Why Us',       href: '#why-us' },
  ];

  /* ── Logo ── */
  const logo = (
    <div style={{ display: 'flex', alignItems: 'center' }}>
      <img
        src="/logo.png"
        alt="SatQuery AI"
        style={{
          height: 46,
          width: 'auto',
          objectFit: 'contain',
          display: 'block',
        }}
        draggable={false}
      />
    </div>
  );

  /* ── Auth button / User badge ── */
  const authBtn = user ? (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 7,
          padding: '6px 12px',
          borderRadius: 9999,
          background: 'rgba(255,255,255,0.06)',
          border: '1px solid rgba(255,255,255,0.12)',
        }}
      >
        <div
          style={{
            width: 22,
            height: 22,
            borderRadius: '50%',
            background: 'linear-gradient(135deg, #6366f1, #a855f7)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            fontWeight: 700,
            color: '#fff',
          }}
        >
          {userInitials}
        </div>
        <span
          style={{
            fontSize: 12.5,
            fontWeight: 500,
            color: 'rgba(255,255,255,0.9)',
            fontFamily: 'Inter, system-ui, sans-serif',
            maxWidth: 110,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {userName}
        </span>
      </div>
      <button
        onClick={() => signOut()}
        style={{
          padding: '8px 14px',
          fontSize: 12,
          border: '1px solid rgba(255,255,255,0.14)',
          background: 'rgba(255,255,255,0.04)',
          color: 'rgba(255,255,255,0.65)',
          borderRadius: 9999,
          cursor: 'pointer',
          whiteSpace: 'nowrap',
          transition: 'all 0.2s',
          fontFamily: 'Inter, system-ui, sans-serif',
        }}
        onMouseEnter={e => {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(239,68,68,0.4)';
          (e.currentTarget as HTMLButtonElement).style.color = '#fca5a5';
        }}
        onMouseLeave={e => {
          (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.14)';
          (e.currentTarget as HTMLButtonElement).style.color = 'rgba(255,255,255,0.65)';
        }}
      >
        Sign Out
      </button>
    </div>
  ) : (
    <button
      onClick={() => {
        if (onLoginClick) onLoginClick();
        else openAuthModal('login');
      }}
      style={{
        padding: '10px 20px',
        fontSize: 13,
        border: '1px solid #333',
        background: 'rgba(31,31,31,0.62)',
        color: '#d1d5db',
        borderRadius: 9999,
        cursor: 'pointer',
        whiteSpace: 'nowrap',
        transition: 'border-color 0.2s, color 0.2s',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
      onMouseEnter={e => {
        (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.5)';
        (e.currentTarget as HTMLButtonElement).style.color = '#fff';
      }}
      onMouseLeave={e => {
        (e.currentTarget as HTMLButtonElement).style.borderColor = '#333';
        (e.currentTarget as HTMLButtonElement).style.color = '#d1d5db';
      }}
    >
      Log In
    </button>
  );

  /* ── GitHub button — white gradient + blur glow ── */
  const githubBtn = (
    <div style={{ position: 'relative' }} className="group">
      {/* Glow */}
      <div style={{
        position: 'absolute',
        inset: 0,
        margin: -8,
        borderRadius: 9999,
        background: '#f3f4f6',
        opacity: 0.4,
        filter: 'blur(12px)',
        pointerEvents: 'none',
        transition: 'opacity 0.3s, margin 0.3s',
      }} className="hidden sm:block group-hover:opacity-60" />
      <a
        href={githubUrl}
        target="_blank"
        rel="noopener noreferrer"
        style={{
          position: 'relative',
          zIndex: 10,
          display: 'inline-flex',
          alignItems: 'center',
          gap: 5,
          padding: '10px 20px',
          fontSize: 13,
          fontWeight: 600,
          color: '#000',
          background: 'linear-gradient(135deg, #f3f4f6, #d1d5db)',
          borderRadius: 9999,
          textDecoration: 'none',
          whiteSpace: 'nowrap',
          transition: 'background 0.2s',
          fontFamily: 'Inter, system-ui, sans-serif',
        }}
      >
        GitHub
        <svg width="11" height="11" viewBox="0 0 11 11" fill="none">
          <path d="M2.5 8.5L8.5 2.5M8.5 2.5H4M8.5 2.5V7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </a>
    </div>
  );

  return (
    <header
      style={{
        position: 'fixed',
        top: 24,
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 50,
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        paddingTop: 16,
        paddingBottom: 16,
        paddingLeft: 28,
        paddingRight: 28,
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        borderRadius,
        border: '1px solid #333',
        background: '#1f1f1f57',
        transition: 'border-radius 0s ease-in-out',
      }}
      /* mobile: full width */
      className="w-[calc(100%-2rem)] sm:w-auto"
    >
      {/* ── Main row ── */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: 36 }}>

        {/* Logo */}
        {logo}

        {/* Desktop nav links */}
        <nav style={{ display: 'flex', alignItems: 'center', gap: 28 }} className="hidden sm:flex">
          {navLinks.map(link => (
            <AnimatedNavLink key={link.href} href={link.href}>
              {link.label}
            </AnimatedNavLink>
          ))}
        </nav>

        {/* Desktop CTAs */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }} className="hidden sm:flex">
          {authBtn}
          {githubBtn}
        </div>

        {/* Mobile hamburger */}
        <button
          className="sm:hidden"
          onClick={() => setIsOpen(v => !v)}
          style={{ background: 'none', border: 'none', color: '#d1d5db', cursor: 'pointer', padding: 4 }}
          aria-label={isOpen ? 'Close Menu' : 'Open Menu'}
        >
          {isOpen ? (
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg width="24" height="24" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* ── Mobile dropdown ── */}
      <div
        className={`sm:hidden flex flex-col items-center w-full overflow-hidden transition-all duration-300 ease-in-out
          ${isOpen ? 'max-h-[500px] opacity-100 pt-5' : 'max-h-0 opacity-0 pt-0 pointer-events-none'}`}
      >
        <nav style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16, width: '100%' }}>
          {navLinks.map(link => (
            <a
              key={link.href}
              href={link.href}
              style={{ color: '#d1d5db', textDecoration: 'none', fontSize: 15, textAlign: 'center', width: '100%' }}
            >
              {link.label}
            </a>
          ))}
        </nav>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12, marginTop: 16, width: '100%' }}>
          {authBtn}
          {githubBtn}
        </div>
      </div>
    </header>
  );
}

export default Navbar;
