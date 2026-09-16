import { useCallback, useEffect, useState } from "react";

const KEY = "formfit.voice";

/** Spoken cues via the Web Speech API (optional, remembered per browser). */
export function useSpeech() {
  const supported = typeof window !== "undefined" && "speechSynthesis" in window;
  const [enabled, setEnabled] = useState(() => {
    try { return localStorage.getItem(KEY) === "1"; } catch { return false; }
  });
  useEffect(() => {
    try { localStorage.setItem(KEY, enabled ? "1" : "0"); } catch { /* ignore */ }
    if (!enabled && supported) window.speechSynthesis.cancel();
  }, [enabled, supported]);

  const say = useCallback((text: string) => {
    if (!enabled || !supported) return;
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 1.05;
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }, [enabled, supported]);

  return { supported, enabled, setEnabled, say };
}
