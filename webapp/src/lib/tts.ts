// Web Speech API TTS for the dispatcher location tail. Replaces flutter_tts.
// Picks a US English female-ish voice when available; falls back to default.

export function speak(text: string, volume = 1.0): Promise<void> {
  return new Promise((resolve) => {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) {
      resolve();
      return;
    }
    try {
      const synth = window.speechSynthesis;
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 0.9;
      u.pitch = 1.0;
      u.volume = Math.max(0, Math.min(1, volume));

      const voices = synth.getVoices();
      const preferred = voices.find((v) => {
        const n = v.name.toLowerCase();
        const isUs = v.lang.toLowerCase().startsWith("en-us");
        const isFemale =
          n.includes("female") ||
          n.includes("zira") ||
          n.includes("aria") ||
          n.includes("jenny") ||
          n.includes("samantha") ||
          n.includes("google us english");
        return isUs && isFemale;
      });
      if (preferred) u.voice = preferred;

      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        resolve();
      };
      u.onend = finish;
      u.onerror = finish;
      // Safety timeout so the dispatcher chain never hangs.
      setTimeout(finish, 12000);
      synth.speak(u);
    } catch {
      resolve();
    }
  });
}

// Voice list loads async on some browsers; warm it up early.
export function warmUpVoices() {
  if (typeof window !== "undefined" && "speechSynthesis" in window) {
    window.speechSynthesis.getVoices();
  }
}
