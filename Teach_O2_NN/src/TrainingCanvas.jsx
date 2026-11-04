import React, { useEffect, useRef } from "react";

/*
 * Live training of a 1 → 8 → 8 → 1 GELU network on 36 Morse points for O2,
 * written in plain JavaScript (forward pass, backprop and Adam by hand) so it
 * runs in any browser with no ML library. Trains ~4,800 epochs, pauses, then
 * restarts from new random weights.
 */
const De = 5.21, re = 1.208, a = 2.66, H = 8, N = 36;
const morse = (r) => { const e = Math.exp(-a * (r - re)); return De * (1 - e) * (1 - e) - De; };

export default function TrainingCanvas({ width = 1100, height = 640 }) {
  const ref = useRef(null);

  useEffect(() => {
    // ---- data, in standard units ----
    const R = [], Y = [];
    for (let i = 0; i < N; i++) { const t = i / (N - 1), r = 0.95 + 2.05 * Math.pow(t, 1.4); R.push(r); Y.push(morse(r)); }
    const xm = 1.9, xs = 0.6;
    const ym = Y.reduce((s, v) => s + v, 0) / N;
    const ys = Math.sqrt(Y.reduce((s, v) => s + (v - ym) ** 2, 0) / N);

    // ---- GELU (tanh form) and its derivative ----
    const K = 0.7978845608, C = 0.044715;
    const ge = (x) => 0.5 * x * (1 + Math.tanh(K * (x + C * x * x * x)));
    const dge = (x) => { const t = Math.tanh(K * (x + C * x * x * x)); return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * K * (1 + 3 * C * x * x); };

    // ---- seeded random numbers so every restart is reproducible ----
    let seed = 7;
    const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
    const gauss = () => { const u = rnd() || 1e-9, v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    const mat = (n, m, s) => Array.from({ length: n }, () => Array.from({ length: m }, () => gauss() * s));
    const zer = (n, m) => mat(n, m, 0);

    let P, M, V, step, ep, hold, loss;
    const init = () => {
      P = { W1: mat(H, 1, 1.5), b1: mat(H, 1, 0.8), W2: mat(H, H, Math.sqrt(2 / H)), b2: zer(H, 1), W3: mat(1, H, Math.sqrt(1 / H)), b3: zer(1, 1) };
      M = {}; V = {};
      for (const k in P) { M[k] = zer(P[k].length, P[k][0].length); V[k] = zer(P[k].length, P[k][0].length); }
      step = 0; ep = 0; hold = 0; loss = [];
    };

    const fwd = (x) => {
      const z1 = [], a1 = [], z2 = [], a2 = [];
      for (let i = 0; i < H; i++) { z1[i] = P.W1[i][0] * x + P.b1[i][0]; a1[i] = ge(z1[i]); }
      for (let i = 0; i < H; i++) { let s = P.b2[i][0]; for (let j = 0; j < H; j++) s += P.W2[i][j] * a1[j]; z2[i] = s; a2[i] = ge(s); }
      let o = P.b3[0][0]; for (let j = 0; j < H; j++) o += P.W3[0][j] * a2[j];
      return { z1, a1, z2, a2, o };
    };

    // one full-batch epoch: forward, backprop (chain rule), Adam update
    const train = () => {
      const G = {}; let L = 0;
      for (const k in P) G[k] = zer(P[k].length, P[k][0].length);
      for (let n = 0; n < N; n++) {
        const x = (R[n] - xm) / xs, y = (Y[n] - ym) / ys, f = fwd(x), e = f.o - y;
        L += e * e; const go = 2 * e / N; G.b3[0][0] += go;
        const g2 = [];
        for (let j = 0; j < H; j++) { G.W3[0][j] += go * f.a2[j]; g2[j] = go * P.W3[0][j] * dge(f.z2[j]); }
        const g1 = new Array(H).fill(0);
        for (let i = 0; i < H; i++) { G.b2[i][0] += g2[i]; for (let j = 0; j < H; j++) { G.W2[i][j] += g2[i] * f.a1[j]; g1[j] += g2[i] * P.W2[i][j]; } }
        for (let j = 0; j < H; j++) { const g = g1[j] * dge(f.z1[j]); G.W1[j][0] += g * x; G.b1[j][0] += g; }
      }
      step++; const lr = 0.01, B1 = 0.9, B2 = 0.999;
      for (const k in P) for (let i = 0; i < P[k].length; i++) for (let j = 0; j < P[k][0].length; j++) {
        const gg = G[k][i][j];
        M[k][i][j] = B1 * M[k][i][j] + (1 - B1) * gg;
        V[k][i][j] = B2 * V[k][i][j] + (1 - B2) * gg * gg;
        P[k][i][j] -= lr * (M[k][i][j] / (1 - B1 ** step)) / (Math.sqrt(V[k][i][j] / (1 - B2 ** step)) + 1e-8);
      }
      return L / N;
    };

    // ---- drawing ----
    const cv = ref.current, g = cv.getContext("2d");
    const dpr = 2; cv.width = width * dpr; cv.height = height * dpr; g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = width, h = height, Lm = 96, Rt = 16, T = 16, B = 80, ylo = -6, yhi = 4, rl = 0.9, rh = 3.05;
    const X = (r) => Lm + (r - rl) / (rh - rl) * (w - Lm - Rt);
    const Yp = (v) => T + (yhi - v) / (yhi - ylo) * (h - T - B);
    const font = "system-ui, -apple-system, Segoe UI, Arial, sans-serif";

    const draw = () => {
      g.fillStyle = "#F7F5EF"; g.fillRect(0, 0, w, h);
      g.font = `24px ${font}`; g.fillStyle = "#5A6475"; g.strokeStyle = "#E1DED4"; g.lineWidth = 1.5;
      for (let v = -6; v <= 4; v += 2) { g.beginPath(); g.moveTo(Lm, Yp(v)); g.lineTo(w - Rt, Yp(v)); g.stroke(); g.textAlign = "right"; g.fillText(v, Lm - 12, Yp(v) + 8); }
      g.textAlign = "center";
      for (let r = 1; r <= 3.001; r += 0.5) g.fillText(r.toFixed(1), X(r), h - B + 34);
      g.fillText("O–O distance r (Å)", Lm + (w - Lm) / 2, h - 8);
      g.save(); g.translate(24, T + (h - T - B) / 2); g.rotate(-Math.PI / 2); g.fillText("V (eV)", 0, 0); g.restore();

      g.save(); g.beginPath(); g.rect(Lm, T, w - Lm - Rt, h - T - B); g.clip();
      const S = 180, rs = [], fs = [];
      for (let i = 0; i <= S; i++) { rs.push(rl + (rh - rl) * i / S); fs.push(fwd((rs[i] - xm) / xs)); }
      // each last-layer neuron's weighted contribution ("pieces")
      g.strokeStyle = "#1BAF7A"; g.globalAlpha = 0.5; g.lineWidth = 2;
      for (let j = 0; j < H; j++) { g.beginPath(); for (let i = 0; i <= S; i++) { const pv = P.W3[0][j] * fs[i].a2[j] * ys; i ? g.lineTo(X(rs[i]), Yp(pv)) : g.moveTo(X(rs[i]), Yp(pv)); } g.stroke(); }
      // Morse reference
      g.globalAlpha = 1; g.setLineDash([10, 8]); g.strokeStyle = "#5A6475"; g.lineWidth = 3; g.beginPath();
      for (let i = 0; i <= S; i++) { i ? g.lineTo(X(rs[i]), Yp(morse(rs[i]))) : g.moveTo(X(rs[i]), Yp(morse(rs[i]))); } g.stroke(); g.setLineDash([]);
      // network prediction
      g.strokeStyle = "#2A78D6"; g.lineWidth = 5; g.lineJoin = "round"; g.beginPath();
      for (let i = 0; i <= S; i++) { const nv = fs[i].o * ys + ym; i ? g.lineTo(X(rs[i]), Yp(nv)) : g.moveTo(X(rs[i]), Yp(nv)); } g.stroke();
      // training data
      g.fillStyle = "#E07A2F"; for (let i = 0; i < N; i++) { g.beginPath(); g.arc(X(R[i]), Yp(Y[i]), 6, 0, 2 * Math.PI); g.fill(); }
      g.restore();

      const rm = loss.length ? Math.sqrt(loss[loss.length - 1]) * ys : 0;
      g.textAlign = "right"; g.fillStyle = "#14213D"; g.font = `600 28px ${font}`;
      g.fillText("epoch " + ep.toLocaleString(), w - Rt - 16, T + 40);
      g.font = `26px ${font}`; g.fillStyle = "#5A6475";
      g.fillText(loss.length ? `RMSE ${rm.toFixed(3)} eV` : "random weights", w - Rt - 16, T + 78);
      if (hold > 0) { g.fillStyle = "#2A78D6"; g.fillText("trained ✓", w - Rt - 16, T + 116); }
    };

    let raf;
    const loop = () => {
      if (hold > 0) { hold++; if (hold > 200) { seed += 101; init(); } }
      else { let l; for (let k = 0; k < 12; k++) { l = train(); ep++; } loss.push(l); if (ep >= 4800) hold = 1; }
      draw(); raf = requestAnimationFrame(loop);
    };
    init(); loop();
    return () => cancelAnimationFrame(raf);
  }, [width, height]);

  return (
    <canvas
      ref={ref}
      style={{ width, height, display: "block" }}
      role="img"
      aria-label="Live animation: a GELU neural network trains on 36 Morse points for O2; the blue prediction converges onto the dashed Morse curve while green neuron contributions rearrange."
    />
  );
}
