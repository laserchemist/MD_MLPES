import React, { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { SLIDES } from "./slides.jsx";

const W = 1920, H = 1080;

// Slide number lives in the URL hash (#3), so links and reloads keep your place.
const fromHash = () => {
  const n = parseInt(window.location.hash.replace("#", ""), 10);
  return Number.isFinite(n) ? Math.min(Math.max(n - 1, 0), SLIDES.length - 1) : 0;
};

function useScale() {
  const [scale, setScale] = useState(1);
  useLayoutEffect(() => {
    const fit = () => setScale(Math.min(window.innerWidth / W, window.innerHeight / H));
    fit();
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, []);
  return scale;
}

export default function App() {
  const [index, setIndex] = useState(fromHash);
  const [notes, setNotes] = useState(false);
  const scale = useScale();

  const go = useCallback((i) => {
    const n = Math.min(Math.max(i, 0), SLIDES.length - 1);
    setIndex(n);
    window.history.replaceState(null, "", `#${n + 1}`);
  }, []);

  useEffect(() => {
    const onKey = (e) => {
      if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); go(index + 1); }
      else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); go(index - 1); }
      else if (e.key === "Home") go(0);
      else if (e.key === "End") go(SLIDES.length - 1);
      else if (e.key === "n" || e.key === "N") setNotes((v) => !v);
      else if (e.key === "f" || e.key === "F") {
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen?.();
      }
    };
    const onHash = () => setIndex(fromHash());
    window.addEventListener("keydown", onKey);
    window.addEventListener("hashchange", onHash);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("hashchange", onHash); };
  }, [index, go]);

  const slide = SLIDES[index];
  const { Component } = slide;

  return (
    <>
      <div className="progress" style={{ width: `${((index + 1) / SLIDES.length) * 100}%` }} />
      <main className="viewer">
        <div className="stage-box" style={{ width: W * scale, height: H * scale }}>
          {/* key forces a fresh mount, so the live animation restarts on its slide */}
          <div className="stage" style={{ transform: `scale(${scale})` }} key={slide.id}>
            <Component />
          </div>
        </div>
      </main>
      {notes && (
        <aside className="notes">
          <h4>Speaker notes · {slide.title}</h4>
          <p>{slide.notes}</p>
        </aside>
      )}
      <nav className="controls" aria-label="Slide controls">
        <button onClick={() => go(index - 1)} aria-label="Previous slide">‹</button>
        <span>{index + 1} / {SLIDES.length}</span>
        <button onClick={() => go(index + 1)} aria-label="Next slide">›</button>
        <button onClick={() => setNotes((v) => !v)}>Notes</button>
        <span className="help">← → keys · N notes · F full screen</span>
      </nav>
    </>
  );
}
