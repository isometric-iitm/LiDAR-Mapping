"use client";
/**
 * Pre-download device gate. Rendered by page.tsx when the device verdict
 * fails; page.tsx decides WHEN to render it and when to release the engine.
 * This component is purely presentational:
 *   - viewport fail: needs >= 700px width (dense HUD + sidebar); mobile gets
 *     a "widen / use desktop" panel
 *   - WebGPU fail: browser-specific help card (Chrome/Edge/Safari/Firefox)
 * Every panel offers "Continue anyway" for anyone who insists.
 */
import { useMemo } from "react";
import type { ReactNode } from "react";

const MIN_W = 700;
const MIN_H = 480;

type Browser = "chrome" | "safari" | "firefox" | "other";

export type GateVerdict =
  | { ok: true }
  | { ok: false; why: "viewport" | "webgpu"; vw: number; vh: number; browser: Browser };

/** Runs once per page load, before any download. */
export function checkDevice(): GateVerdict {
  if (typeof window === "undefined") return { ok: true };
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  if (vw < MIN_W || vh < MIN_H) {
    return { ok: false, why: "viewport", vw, vh, browser: detectBrowser(navigator.userAgent) };
  }
  // baked demo: no WebGPU required
  return { ok: true };
}

function detectBrowser(ua: string): Browser {
  if (/firefox\//i.test(ua)) return "firefox";
  // Edge/Brave/Opera/etc all report "chrome" in the UA; that is the right
  // guidance for all of them (Chromium docs apply).
  if (/chrome|chromium|crios|edg\//i.test(ua)) return "chrome";
  if (/safari/i.test(ua)) return "safari";
  return "other";
}

const HELP: Record<Browser, { name: string; link: string; text: string }> = {
  chrome: {
    name: "Chrome or Edge 113+ (desktop)",
    link: "https://developer.chrome.com/docs/web-platform/webgpu",
    text:
      "Update to 113 or newer and make sure hardware acceleration is on: " +
      'Settings > System > "Use graphics acceleration when available", then relaunch.',
  },
  safari: {
    name: "Safari 26+ (macOS Tahoe / iOS 26)",
    link: "https://developer.apple.com/documentation/safari-release-notes/safari-26-release-notes",
    text: "WebGPU ships in Safari 26. Update your OS, or use Chrome/Edge 113+ on this device instead.",
  },
  firefox: {
    name: "Firefox 141+ (Windows) or 145+ (macOS)",
    link: "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API",
    text:
      "WebGPU is enabled by default on Windows (141+) and macOS (145+). On Linux, use Nightly and set " +
      "gfx.webgpu.ignore-blocklist in about:config.",
  },
  other: {
    name: "a WebGPU-capable browser",
    link: "https://webgpureport.org",
    text: "Visit webgpureport.org to check this browser. Chrome or Edge 113+ (desktop) is the safest choice.",
  },
};

function Panel({ title, children, onProceed }: { title: string; children: ReactNode; onProceed: () => void }) {
  return (
    <div className="absolute inset-0 z-30 flex items-center justify-center bg-black/85 p-4 backdrop-blur-[2px]">
      <div className="frost flex w-[min(460px,92vw)] flex-col gap-3 px-6 py-6 text-center">
        <div className="text-base font-semibold text-zinc-100">{title}</div>
        <div className="text-sm leading-relaxed text-zinc-400">{children}</div>
        <button
          onClick={onProceed}
          className="mx-auto mt-1 rounded-[3px] px-3 py-1.5 text-xs text-zinc-400 underline decoration-zinc-600 underline-offset-4 transition-colors hover:text-zinc-200"
        >
          Continue anyway
        </button>
      </div>
    </div>
  );
}

export default function DeviceGate({ verdict, onPass }: { verdict: GateVerdict; onPass: () => void }) {
  const help = useMemo(() => HELP[verdict.ok ? "other" : verdict.browser], [verdict]);
  if (verdict.ok) return null;
  if (verdict.why === "viewport") {
    return (
      <Panel title="Best viewed on a larger screen" onProceed={onPass}>
        This demo renders a dense live 3D map and needs about {MIN_W} pixels of width ({verdict.vw} available).
        Open it on a desktop browser, widen this window, or request the desktop site in your browser menu.
      </Panel>
    );
  }
  return (
    <Panel title="WebGPU required" onProceed={onPass}>
      This baked demo does not require WebGPU. Please continue.
    </Panel>
  );
}
