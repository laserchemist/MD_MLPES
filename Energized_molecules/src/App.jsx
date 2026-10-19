import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import deck from './slides.json'

const W = 1920
const H = 1080
const slides = deck.slides
const N = slides.length

/** One slide at its native 1920×1080 size, scaled to `scale`. */
function SlideFrame({ slide, scale, className = '' }) {
  return (
    <div className={`frame ${className}`} style={{ width: W * scale, height: H * scale }}>
      <div
        className="slide-html"
        style={{ transform: `scale(${scale})` }}
        dangerouslySetInnerHTML={{ __html: slide.html }}
      />
    </div>
  )
}

function useElementSize(ref) {
  const [size, setSize] = useState({ width: 0, height: 0 })
  useLayoutEffect(() => {
    if (!ref.current) return
    const ro = new ResizeObserver(([e]) => setSize({ width: e.contentRect.width, height: e.contentRect.height }))
    ro.observe(ref.current)
    return () => ro.disconnect()
  }, [ref])
  return size
}

function readHash() {
  const m = window.location.hash.match(/^#\/(\d+)/)
  const i = m ? parseInt(m[1], 10) - 1 : 0
  return Math.min(Math.max(i, 0), N - 1)
}

export default function App() {
  const [index, setIndex] = useState(readHash)
  const [showNotes, setShowNotes] = useState(false)
  const [overview, setOverview] = useState(false)
  const stageRef = useRef(null)
  const { width, height } = useElementSize(stageRef)
  const scale = width && height ? Math.min(width / W, height / H) : 0
  const slide = slides[index]

  const go = useCallback((i) => setIndex(Math.min(Math.max(i, 0), N - 1)), [])

  // keep the URL hash in step with the current slide, and follow back/forward
  useEffect(() => {
    const target = `#/${index + 1}`
    if (window.location.hash !== target) window.history.replaceState(null, '', target)
    document.title = `${index + 1}. ${slide.title} · ${deck.title}`
  }, [index, slide.title])
  useEffect(() => {
    const onHash = () => setIndex(readHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const k = e.key
      if (['ArrowRight', 'PageDown', ' ', 'ArrowDown'].includes(k)) { e.preventDefault(); go(index + 1) }
      else if (['ArrowLeft', 'PageUp', 'ArrowUp'].includes(k)) { e.preventDefault(); go(index - 1) }
      else if (k === 'Home') go(0)
      else if (k === 'End') go(N - 1)
      else if (k === 'n' || k === 'N') setShowNotes((v) => !v)
      else if (k === 'o' || k === 'O' || (k === 'Escape' && overview)) setOverview((v) => (k === 'Escape' ? false : !v))
      else if (k === 'f' || k === 'F') {
        if (document.fullscreenElement) document.exitFullscreen()
        else document.documentElement.requestFullscreen?.()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, overview, go])

  // swipe on touch screens
  const touch = useRef(null)
  const onTouchStart = (e) => { touch.current = e.touches[0].clientX }
  const onTouchEnd = (e) => {
    if (touch.current == null) return
    const dx = e.changedTouches[0].clientX - touch.current
    if (Math.abs(dx) > 50) go(index + (dx < 0 ? 1 : -1))
    touch.current = null
  }

  const sectionStarts = useMemo(() => {
    const out = []
    slides.forEach((s, i) => { if (i === 0 || s.section !== slides[i - 1].section) out.push(i) })
    return out
  }, [])

  return (
    <div className="app">
      <main className="viewer">
        <div className="stage" ref={stageRef} onTouchStart={onTouchStart} onTouchEnd={onTouchEnd}>
          {scale > 0 && <SlideFrame slide={slide} scale={scale} className="current" />}
          <button className="hit prev" aria-label="Previous slide" onClick={() => go(index - 1)} disabled={index === 0} />
          <button className="hit next" aria-label="Next slide" onClick={() => go(index + 1)} disabled={index === N - 1} />
        </div>
        {showNotes && (
          <aside className="notes" aria-label="Speaker notes">
            <p className="notes-label">Speaker notes · slide {index + 1}</p>
            <p>{slide.notes || 'No notes for this slide.'}</p>
          </aside>
        )}
      </main>

      <footer className="toolbar">
        <div className="progress" aria-hidden="true">
          <div style={{ width: `${((index + 1) / N) * 100}%` }} />
          {sectionStarts.map((i) => (
            <span key={i} className="tick" style={{ left: `${(i / N) * 100}%` }} />
          ))}
        </div>
        <div className="controls">
          <button onClick={() => go(index - 1)} disabled={index === 0} aria-label="Previous">←</button>
          <span className="counter">{index + 1} / {N}</span>
          <button onClick={() => go(index + 1)} disabled={index === N - 1} aria-label="Next">→</button>
          <span className="where" title={slide.section}>{slide.title}</span>
          <span className="spacer" />
          <button className={showNotes ? 'on' : ''} onClick={() => setShowNotes((v) => !v)} title="Speaker notes (N)">Notes</button>
          <button onClick={() => setOverview(true)} title="All slides (O)">Overview</button>
          <button onClick={() => window.print()} title="Print or save as PDF">PDF</button>
          <button onClick={() => (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.())} title="Full screen (F)">Full screen</button>
        </div>
      </footer>

      {overview && (
        <div className="overview" role="dialog" aria-label="All slides">
          <div className="overview-head">
            <h1>{deck.title}</h1>
            <button onClick={() => setOverview(false)}>Close (Esc)</button>
          </div>
          <div className="grid">
            {slides.map((s, i) => (
              <button
                key={s.id}
                className={`thumb ${i === index ? 'active' : ''}`}
                onClick={() => { go(i); setOverview(false) }}
              >
                <SlideFrame slide={s} scale={0.16} />
                <span className="thumb-label">{i + 1}. {s.title}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* every slide at full size, shown only when printing */}
      <div className="print-deck" aria-hidden="true">
        {slides.map((s) => <SlideFrame key={s.id} slide={s} scale={1} className="print-page" />)}
      </div>
    </div>
  )
}
