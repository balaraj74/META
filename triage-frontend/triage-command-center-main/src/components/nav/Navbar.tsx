import { Link, useLocation } from "@tanstack/react-router";
import { LiveBadge } from "@/components/ui/LiveBadge";
import { ThemeToggle } from "@/components/ui/ThemeToggle";
import { ExportPanel } from "@/components/ui/ExportPanel";
import { ArrowRight } from "lucide-react";

function RedCross({ className = "h-5 w-5" }: { className?: string }) {
  return (
    <svg viewBox="0 0 24 24" className={className} aria-hidden="true">
      <rect x="9.5" y="3" width="5" height="18" fill="var(--emergency-red)" />
      <rect x="3" y="9.5" width="18" height="5" fill="var(--emergency-red)" />
    </svg>
  );
}

const LINKS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/visualizer", label: "Visualizer" },
  { to: "/command", label: "Command Center" },
  { to: "/training", label: "Training" },
  { to: "/sponsors", label: "Sponsors" },
  { to: "/replay", label: "Replay" },
] as const;

export function Navbar() {
  const loc = useLocation();
  const showPitchBtn = loc.pathname === "/dashboard" || loc.pathname === "/visualizer";

  function openPitch() {
    if (typeof document !== "undefined") {
      document.documentElement.requestFullscreen?.().catch(() => {});
    }
    window.location.href = "/pitch";
  }

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-surface/95 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1600px] items-center justify-between px-6">
        <Link to="/" className="flex items-center gap-2">
          <RedCross />
          <span className="font-display text-2xl leading-none text-text-primary">TRIAGE</span>
          <span className="ml-2 hidden font-mono text-[10px] uppercase tracking-widest text-text-muted md:inline">
            v0.1 · hackathon build
          </span>
        </Link>

        <nav className="flex items-center gap-5 text-sm">
          {LINKS.map((l) => (
            <NavLink key={l.to} to={l.to}>
              {l.label}
            </NavLink>
          ))}
          <a
            href="https://github.com"
            target="_blank"
            rel="noreferrer"
            className="hidden text-text-secondary hover:text-text-primary lg:inline"
          >
            GitHub
          </a>
          <a
            href="https://huggingface.co"
            target="_blank"
            rel="noreferrer"
            className="hidden text-text-secondary hover:text-text-primary lg:inline"
          >
            HuggingFace
          </a>
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <LiveBadge />
          <ExportPanel />
          {showPitchBtn && (
            <button
              onClick={openPitch}
              className="hidden items-center gap-1.5 bg-primary px-3 py-1.5 font-mono text-[10px] uppercase tracking-wider text-primary-foreground hover:bg-primary-dark md:inline-flex"
              style={{ borderRadius: 4 }}
            >
              Pitch Mode <ArrowRight className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

function NavLink({
  to,
  children,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  to: string;
  children: React.ReactNode;
}) {
  return (
    <Link
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      to={to as any}
      className="relative py-1 text-text-secondary hover:text-text-primary"
      activeProps={{
        className:
          "relative py-1 text-text-primary after:absolute after:left-0 after:right-0 after:-bottom-[15px] after:h-[2px] after:bg-primary",
      }}
    >
      {children}
    </Link>
  );
}
