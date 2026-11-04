import React from "react";
import TrainingCanvas from "./TrainingCanvas.jsx";

const fig = (name) => `${import.meta.env.BASE_URL}figures/${name}`;

const Head = ({ eyebrow, title }) => (
  <div className="head">
    <p className="eyebrow">{eyebrow}</p>
    <h2 className="title">{title}</h2>
  </div>
);

const GELU_PATH =
  "M40.0 268.9 L48.8 268.9 L57.5 268.9 L66.2 268.9 L75.0 268.9 L83.8 269.0 L92.5 269.0 L101.3 269.0 L110.0 269.0 L118.7 269.1 L127.5 269.2 L136.2 269.2 L145.0 269.3 L153.8 269.5 L162.5 269.7 L171.2 269.9 L180.0 270.1 L188.8 270.4 L197.5 270.8 L206.3 271.2 L215.0 271.7 L223.8 272.3 L232.5 272.9 L241.3 273.6 L250.0 274.4 L258.8 275.1 L267.5 275.9 L276.2 276.7 L285.0 277.5 L293.8 278.2 L302.5 278.8 L311.2 279.2 L320.0 279.4 L328.8 279.4 L337.5 279.1 L346.2 278.5 L355.0 277.5 L363.8 276.0 L372.5 274.1 L381.3 271.8 L390.0 268.9 L398.8 265.5 L407.5 261.7 L416.2 257.4 L425.0 252.6 L433.8 247.4 L442.5 241.8 L451.2 235.9 L460.0 229.7 L468.8 223.2 L477.5 216.6 L486.3 209.8 L495.0 202.8 L503.8 195.9 L512.5 188.9 L521.2 181.8 L530.0 174.8 L538.8 167.9 L547.5 161.0 L556.2 154.1 L565.0 147.3 L573.8 140.6 L582.5 134.0 L591.3 127.4 L600.0 120.8 L608.8 114.4 L617.5 107.9 L626.2 101.5 L635.0 95.2 L643.8 88.9 L652.5 82.6 L661.2 76.3 L670.0 70.0 L678.8 63.7 L687.5 57.5 L696.2 51.3 L705.0 45.0 L713.8 38.8 L722.5 32.6 L731.2 26.3 L740.0 20.1";

/* ------------------------------------------------------------------ */
/* 1. Cover                                                            */
/* ------------------------------------------------------------------ */
function Cover() {
  return (
    <section className="slide dark" style={{ justifyContent: "center", paddingBottom: 160 }}>
      <svg width="760" height="520" viewBox="0 0 760 520" style={{ position: "absolute", right: 96, top: 200, opacity: 0.9 }} aria-hidden="true">
        <path d="M40 10 C 70 300, 95 470, 150 478 C 210 486, 260 330, 360 230 C 460 140, 600 115, 740 110" fill="none" stroke="#2A78D6" strokeWidth="10" strokeLinecap="round" />
        <g fill="#E07A2F">
          {[[58, 140], [82, 330], [112, 440], [150, 478], [196, 460], [250, 350], [320, 265], [410, 185], [520, 138], [640, 118]].map(([x, y]) => (
            <circle key={x} cx={x} cy={y} r="11" />
          ))}
        </g>
      </svg>
      <p style={{ fontSize: 30, fontWeight: 600, letterSpacing: 3, textTransform: "uppercase", color: "var(--orange)", width: 1000 }}>Machine learning for chemistry</p>
      <h1 style={{ fontFamily: "var(--head)", fontSize: 112, fontWeight: 700, lineHeight: 1.05, width: 1000 }}>Teaching a neural network the O–O bond</h1>
      <p style={{ fontSize: 40, lineHeight: 1.35, color: "var(--on-dark)", width: 960 }}>
        Neural-network potential energy surfaces, from a single neuron to forces and vibrational frequencies
      </p>
      <p style={{ position: "absolute", left: 128, bottom: 64, fontSize: 24, color: "var(--on-dark)" }}>
        Jonathan M. Smith · Temple University&nbsp;&nbsp;&nbsp;<span style={{ color: "var(--orange)" }}>|</span>&nbsp;&nbsp;&nbsp;Slides and worked example built in collaboration with Claude (Anthropic)
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 2. The problem                                                      */
/* ------------------------------------------------------------------ */
function Problem() {
  const cards = [
    ["Quantum chemistry is slow", "One energy at one geometry: minutes to days."],
    ["Dynamics needs millions of calls", "MD needs V and F = −dV/dr at every time step."],
    ["So learn V from a few samples", "Fit a fast, smooth V(r); differentiate for forces."],
  ];
  return (
    <section className="slide" style={{ gap: 56 }}>
      <Head eyebrow="The problem" title="A potential energy surface is expensive to compute and needed everywhere" />
      <div className="row flex1">
        <div className="col flex1">
          {cards.map(([h, p]) => (
            <div className="card" key={h}>
              <h3>{h}</h3>
              <p className="p30 muted">{p}</p>
            </div>
          ))}
        </div>
        <div className="panel-dark" style={{ width: 640, padding: 48, gap: 24 }}>
          <p className="eyebrow" style={{ color: "var(--orange)", textTransform: "none" }}>OUR TEST CASE: O₂, X ³Σg⁻</p>
          <p style={{ fontFamily: "var(--head)", fontSize: 36, lineHeight: 1.3 }}>V(r) = Dₑ[1 − exp(−a(r − rₑ))]² − Dₑ</p>
          <table className="t" style={{ fontSize: 30 }}>
            <tbody>
              <tr><th>Parameter</th><th>Value</th></tr>
              <tr><td>Well depth Dₑ</td><td>5.21 eV</td></tr>
              <tr><td>Bond length rₑ</td><td>1.208 Å</td></tr>
              <tr><td>Width a</td><td>2.66 Å⁻¹</td></tr>
              <tr><td>Training points</td><td>36, 0.95–3.0 Å</td></tr>
            </tbody>
          </table>
          <p className="p24" style={{ color: "var(--on-dark)" }}>A Morse curve stands in for quantum-chemistry data, so we know the right answer.</p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 3. The network                                                      */
/* ------------------------------------------------------------------ */
function Network() {
  return (
    <section className="slide" style={{ gap: 48 }}>
      <Head eyebrow="The model" title="Inside the network: weigh, add, then bend" />
      <div className="row flex1">
        <div className="card" style={{ width: 856, padding: "36px 48px", gap: 20 }}>
          <h3>One neuron</h3>
          <p style={{ fontSize: 32, lineHeight: 1.4 }}>
            z = <b>w</b>·<b>x</b> + b <span style={{ color: "var(--muted2)" }}>(weigh and add)</span>
            <br />a = GELU(z) = z·Φ(z) <span style={{ color: "var(--muted2)" }}>(bend)</span>
          </p>
          <svg width="760" height="320" viewBox="0 0 760 320" role="img" aria-label="GELU is smooth with a small dip below zero; ReLU has a sharp corner at zero">
            <line x1="40" y1="268.9" x2="740" y2="268.9" stroke="#B9B4A6" strokeWidth="2" />
            <line x1="390" y1="10" x2="390" y2="310" stroke="#B9B4A6" strokeWidth="2" />
            <path d="M40 268.9 L390 268.9 L740 20.1" fill="none" stroke="#E07A2F" strokeWidth="5" strokeDasharray="14 10" strokeLinecap="round" />
            <path d={GELU_PATH} fill="none" stroke="#2A78D6" strokeWidth="7" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div style={{ display: "flex", gap: 40 }}>
            <p className="p24" style={{ color: "var(--blue)", fontWeight: 600 }}>━ GELU: smooth everywhere</p>
            <p className="p24" style={{ color: "var(--orange-ink)", fontWeight: 600 }}>╌ ReLU: sharp corner at 0</p>
          </div>
        </div>
        <div className="col flex1" style={{ gap: 20 }}>
          <h3>The network</h3>
          <div className="col" style={{ gap: 12 }}>
            <p className="pill" style={{ background: "var(--white)", border: "1px solid var(--rule)" }}>Input: bond length r (standardized)</p>
            <p className="pill" style={{ background: "var(--blue-tint)" }}>Hidden layer 1: 16 GELU neurons</p>
            <p className="pill" style={{ background: "var(--blue-tint)" }}>Hidden layer 2: 16 GELU neurons</p>
            <p className="pill" style={{ background: "var(--ink)", color: "var(--paper)" }}>Output: V(r) in eV (linear)</p>
          </div>
          <p className="p30 muted">
            <b style={{ color: "var(--ink)" }}>Why GELU for a PES?</b> Forces are −dV/dr and frequencies need d²V/dr². GELU keeps both smooth; ReLU gives a piecewise-linear V with jumping forces.
          </p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 4. Live training                                                    */
/* ------------------------------------------------------------------ */
function Learn() {
  return (
    <section className="slide" style={{ gap: 16 }}>
      <Head eyebrow="Training, live" title="Watch the network learn the O–O curve" />
      <div style={{ position: "absolute", left: 128, top: 300 }}>
        <TrainingCanvas width={1100} height={640} />
      </div>
      <div className="col" style={{ position: "absolute", left: 1280, top: 300, width: 512, gap: 20 }}>
        <p className="p24" style={{ color: "var(--muted2)" }}>
          <b style={{ color: "var(--blue)" }}>━ network</b>&nbsp;&nbsp;&nbsp;<b style={{ color: "var(--orange-ink)" }}>● data</b>
          <br /><b>╌ Morse</b>&nbsp;&nbsp;&nbsp;<b style={{ color: "#14866A" }}>━ neuron pieces</b>
        </p>
        <p className="p30"><b>Pieces add up.</b> Each green curve is one last-layer neuron times its weight; their sum is the blue curve.</p>
        <p className="p30"><b>Wall first, tail last.</b> Big errors at the repulsive wall get fixed first; the flat dissociation tail is where pieces must cancel.</p>
        <p className="p30"><b>8 neurons per layer,</b> trained in your browser with plain JavaScript and Adam.</p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 5. Loss and optimizer                                               */
/* ------------------------------------------------------------------ */
function Loss() {
  return (
    <section className="slide">
      <Head eyebrow="The loss and the optimizer" title="Training: what we minimize, and how" />
      <div className="row">
        <div className="col" style={{ width: 760, gap: 16 }}>
          <img className="fig" src={fig("loss_curve.png")} style={{ width: 760, height: 523 }}
            alt="Training error on a log scale versus epoch for energy and force, with spikes until the learning rate is cut at epoch 2000." />
          <p className="p24 muted">Spikes mean the steps are too big. Cutting the learning rate at epoch 2,000 lets training settle into the minimum.</p>
        </div>
        <div className="col flex1">
          <div className="panel-dark" style={{ padding: "24px 32px", gap: 8 }}>
            <p style={{ fontFamily: "var(--head)", fontSize: 32, lineHeight: 1.3 }}>
              L = mean[(V − V<span style={{ color: "var(--on-dark)" }}>ref</span>)²] + λ · mean[(F − F<span style={{ color: "var(--on-dark)" }}>ref</span>)²]
            </p>
            <p className="p24" style={{ color: "var(--on-dark)" }}>Mean squared error on energies and forces, with λ = 0.1 Å² balancing the two.</p>
          </div>
          <ol className="p26" style={{ paddingLeft: 36 }}>
            <li><b>Forward:</b> predict V and F = −dV/dr at all 36 points</li>
            <li><b>Loss:</b> score both against the reference</li>
            <li><b>Backprop:</b> the chain rule gives ∂L/∂w for all 321 weights</li>
            <li><b>Adam:</b> each weight takes its own adaptive step</li>
          </ol>
          <table className="t" style={{ fontSize: 26 }}>
            <tbody>
              <tr><th>Loss uses</th><th className="r">Energy only</th><th className="r">+ Forces</th></tr>
              <tr><td>Force RMSE</td><td className="r">0.40 eV/Å</td><td className="r">0.023 eV/Å</td></tr>
              <tr><td>Energy RMSE</td><td className="r">0.0049 eV</td><td className="r">0.0009 eV</td></tr>
              <tr><td>ωₑ (expt ≈ 1580)</td><td className="r">1636 cm⁻¹</td><td className="r">1588 cm⁻¹</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 6. Data flow                                                        */
/* ------------------------------------------------------------------ */
const Arrow = () => (
  <svg className="arrow" viewBox="0 0 56 28" aria-hidden="true">
    <path d="M0 9 H34 V0 L56 14 L34 28 V19 H0 Z" fill="#B9B4A6" />
  </svg>
);

function FlowRow({ label, sub, color, boxes }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
      <p style={{ width: 200, fontFamily: "var(--head)", fontSize: 32, fontWeight: 700, color }}>
        {label}<br /><span style={{ color: "var(--muted2)" }}>{sub}</span>
      </p>
      {boxes.map(([h, p, bg, fg, sub2], i) => (
        <React.Fragment key={h}>
          {i > 0 && <Arrow />}
          <div className="flow-box" style={{ background: bg, border: bg === "var(--white)" ? "1px solid var(--rule)" : "none" }}>
            <p className="h3s" style={{ color: fg }}>{h}</p>
            <p className="p24" style={{ color: sub2, lineHeight: 1.35 }}>{p}</p>
          </div>
        </React.Fragment>
      ))}
    </div>
  );
}

function Batch() {
  const W = "var(--white)";
  return (
    <section className="slide" style={{ gap: 36 }}>
      <Head eyebrow="How data flows" title="One array in, one array out" />
      <FlowRow label="Training" sub="every epoch" color="var(--blue)" boxes={[
        ["r: 36 × 1", "all 36 distances in one array", W, "var(--ink)", "var(--muted)"],
        ["Network", "same 321 weights applied to each row", "var(--blue-tint)", "var(--ink)", "var(--blue-ink)"],
        ["V, F: 36 × 1", "energies, plus forces from one autograd call", W, "var(--ink)", "var(--muted)"],
        ["1 loss → 1 step", "average over all 36, then one Adam update", "var(--ink)", "var(--paper)", "var(--on-dark)"],
      ]} />
      <FlowRow label="Using it" sub="after training" color="var(--orange-ink)" boxes={[
        ["r: 400 × 1", "any grid, any length", W, "var(--ink)", "var(--muted)"],
        ["Trained network", "weights now fixed", "var(--blue-tint)", "var(--ink)", "var(--blue-ink)"],
        ["V, F: 400 × 1", "one forward pass, one autograd call", W, "var(--ink)", "var(--muted)"],
        ["Use them", "plots, MD time steps, ωₑ from d²V/dr²", "#FBE7D6", "var(--ink)", "#7A3D12"],
      ]} />
      <div className="row" style={{ gap: 32 }}>
        <div className="ruled flex1">
          <p className="h3s">Why one autograd call is enough</p>
          <p className="p24 muted">Each Vᵢ depends only on its own rᵢ, so the gradient of ΣVᵢ gives every dVᵢ/drᵢ at once.</p>
        </div>
        <div className="ruled orange flex1">
          <p className="h3s">Full batch vs. mini-batch</p>
          <p className="p24 muted">36 points fit in one batch. Large data sets use a random mini-batch per step (stochastic gradient descent).</p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 7. Annotated code                                                   */
/* ------------------------------------------------------------------ */
const Code = ({ lines }) => (
  <pre className="mono">
    {lines.map(([code, comment], i) => (
      <React.Fragment key={i}>
        {code}
        {comment && <span className="c">{comment}</span>}
        {"\n"}
      </React.Fragment>
    ))}
  </pre>
);

const TRAIN = [
  ["r = R.reshape(-1, 1)            ", "# (36,1): all points at once"],
  ["model = PES()                   ", "# 1→16→16→1 GELU + scaling"],
  ["opt = torch.optim.Adam(model.parameters(), lr=0.01)"],
  ["for epoch in range(6000):"],
  ["    opt.zero_grad()             ", "# clear last step’s gradients"],
  ["    r.requires_grad_(True)      ", "# track r so we can take dV/dr"],
  ["    V = model(r)                ", "# (36,1) energies"],
  ["    dVdr, = torch.autograd.grad(V.sum(), r,"],
  ["                                create_graph=True)"],
  ["    F = -dVdr                   ", "# (36,1) forces"],
  ["    loss = ((V - V_ref)**2).mean() \\"],
  ["         + 0.1 * ((F - F_ref)**2).mean()  ", "# 1 number"],
  ["    loss.backward()             ", "# ∂loss/∂w for all 321 weights"],
  ["    opt.step()                  ", "# one Adam update"],
];
const READ = [
  ["model.eval()"],
  ["r = torch.linspace(0.9, 3.1, 400)"],
  ["r = r.reshape(-1, 1)  ", "# (400,1)"],
  ["r.requires_grad_(True)"],
  ["V = model(r)          ", "# (400,1)"],
  ["dVdr, = torch.autograd.grad("],
  ["            V.sum(), r)"],
  ["F = -dVdr             ", "# (400,1)"],
];

function CodeSlide() {
  return (
    <section className="slide" style={{ gap: 36 }}>
      <Head eyebrow="The code, annotated" title="Training and reading out in PyTorch" />
      <div className="row" style={{ gap: 32, alignItems: "flex-start" }}>
        <div className="panel-dark" style={{ width: 1040 }}>
          <p className="eyebrow" style={{ color: "var(--orange)" }}>① Train: 36 points per step</p>
          <Code lines={TRAIN} />
        </div>
        <div className="col flex1">
          <div className="panel-dark">
            <p className="eyebrow" style={{ color: "var(--orange)" }}>② Read out: any grid, no loop</p>
            <Code lines={READ} />
          </div>
          <div className="card" style={{ border: "2px solid var(--blue)", padding: "24px 32px", gap: 12 }}>
            <p className="eyebrow" style={{ color: "var(--blue)" }}>Standard units inside PES()</p>
            <pre className="mono" style={{ color: "var(--ink)" }}>{"x = (r − 1.81 Å) / 0.64 Å\nV = 1.83 eV · ŷ − 2.15 eV"}</pre>
            <p className="p24 muted">Training mean and SD of r and V. The net sees numbers near 1; autograd applies the chain rule, so F comes out in eV/Å.</p>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 8. Results                                                          */
/* ------------------------------------------------------------------ */
function Results() {
  const stats = [
    ["0.0009 eV", "energy RMSE inside 0.95–3.0 Å", ""],
    ["0.023 eV/Å", "force RMSE, from −dV/dr", ""],
    ["1.209 Å", "network rₑ (Morse: 1.208 Å)", ""],
    ["1588 cm⁻¹", "ωₑ from d²V/dr² (expt ≈ 1580)", "orange"],
  ];
  return (
    <section className="slide">
      <Head eyebrow="Results in PyTorch" title="Energies, forces, and a vibrational frequency" />
      <div className="row">
        <div className="col" style={{ width: 1120, gap: 16 }}>
          <img className="fig" src={fig("fit_force.png")} style={{ width: 1120, height: 448 }}
            alt="The network's V(r) and autograd forces lie on top of the Morse reference." />
          <p className="p24 muted">
            1 → 16 → 16 → 1 GELU network, 6,000 Adam epochs, loss = energy error + 0.1 × force error. Forces come from{" "}
            <b style={{ color: "var(--ink)" }}>torch.autograd.grad</b>, not finite differences.
          </p>
        </div>
        <div className="col flex1">
          {stats.map(([n, l, c]) => (
            <div className={`ruled ${c}`} key={n}>
              <p className="stat">{n}</p>
              <p className="p24 muted">{l}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 9. Weights                                                          */
/* ------------------------------------------------------------------ */
function Weights() {
  return (
    <section className="slide">
      <Head eyebrow="Inside the trained model" title="The fitted O–O network, color-coded" />
      <div className="row">
        <img className="fig" src={fig("network_weights.png")} style={{ width: 1100, height: 602 }}
          alt="Trained 1-16-16-1 GELU network: edges red for positive and blue for negative weights, thicker for larger; nodes colored by bias." />
        <div className="col flex1" style={{ gap: 28 }}>
          <div className="ruled" style={{ gap: 6 }}>
            <p className="stat">321</p>
            <p className="p24 muted">learned numbers: 288 weights and 33 biases</p>
          </div>
          <p className="p28">
            <b>Lines</b> are weights: <b style={{ color: "#B2302B" }}>red +</b>, <b style={{ color: "var(--blue)" }}>blue −</b>, thicker = larger.
            <br /><b>Circles</b> are colored by each neuron’s bias.
          </p>
          <p className="p28">The first layer sets <b>where</b> each neuron bends along r; the output layer mixes signs so pieces <b>add up</b> at the wall and <b>cancel</b> in the flat tail.</p>
          <p className="p28 muted">No single neuron “is” the bond: the curve is spread across all the weights.</p>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 10. GELU vs ReLU                                                    */
/* ------------------------------------------------------------------ */
function Gelu() {
  const pts = [
    ["Smooth gradient → real forces", "ReLU’s derivative is a step, so forces jump. Force error: 1.5 vs 0.023 eV/Å."],
    ["Curvature exists → spectroscopy", "ReLU’s d²V/dr² is zero almost everywhere: no ωₑ. GELU gives ≈ 1580 cm⁻¹."],
    ["No saturation → trains well", "Unlike tanh, GELU passes large inputs, keeping gradients alive in deep nets."],
  ];
  return (
    <section className="slide" style={{ gap: 32 }}>
      <Head eyebrow="Same network, different activation" title="Why GELU suits molecular surfaces and their gradients" />
      <img className="fig" src={fig("gelu_vs_relu.png")} style={{ width: 1664, height: 518 }}
        alt="ReLU vs GELU networks: ReLU force is a staircase and its curvature is zero; GELU follows the Morse force and curvature." />
      <div className="row" style={{ gap: 32 }}>
        {pts.map(([h, p]) => (
          <div className="ruled flex1" key={h}>
            <p className="h3s">{h}</p>
            <p className="p24 muted">{p}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 11. Outside the training data                                       */
/* ------------------------------------------------------------------ */
function Outside() {
  const pts = [
    ["Not random, just unconstrained", "Far from data, GELU nets become straight lines. At 0.3 Å: 35–56 eV vs. 536 eV.", ""],
    ["Disagreement flags danger", "Seeds agree to 0.00 eV inside the data, but differ by 23 eV at 10 Å.", "violet"],
    ["The fixes", "Sample where you simulate, build physics in, and flag out-of-range inputs.", "orange"],
  ];
  return (
    <section className="slide" style={{ gap: 32 }}>
      <Head eyebrow="Beyond the training range" title="Outside the data: confident and wrong" />
      <img className="fig" src={fig("outside.png")} style={{ width: 1664, height: 480 }}
        alt="Three networks trained with different seeds agree inside 0.95–3.0 Å but undershoot the short-range wall and climb at long range." />
      <div className="row" style={{ gap: 32 }}>
        {pts.map(([h, p, c]) => (
          <div className={`ruled ${c} flex1`} key={h}>
            <p className="h3s">{h}</p>
            <p className="p24 muted">{p}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 12. Physics in the architecture                                     */
/* ------------------------------------------------------------------ */
function Physics() {
  return (
    <section className="slide" style={{ paddingBottom: 160 }}>
      <Head eyebrow="Physics in the architecture" title="Build the physics in: V(∞) = 0 by design" />
      <div className="row">
        <img className="fig" src={fig("dissociation.png")} style={{ width: 900, height: 540 }}
          alt="Out to 8 Å: the plain network drifts above zero while V = g(y) − g(0) stays at zero like Morse." />
        <div className="col flex1">
          <div className="panel-dark" style={{ padding: "28px 32px", gap: 8 }}>
            <p style={{ fontFamily: "var(--head)", fontSize: 32, lineHeight: 1.4 }}>y = exp(−(r − rₑ)/λ)<br />V(r) = g(y) − g(0)</p>
            <p className="p24" style={{ color: "var(--on-dark)" }}>As r → ∞, y → 0, so V → g(0) − g(0) = 0 exactly.</p>
          </div>
          <table className="t" style={{ fontSize: 28 }}>
            <tbody>
              <tr><th>r</th><th className="r">Morse</th><th className="r">Plain</th><th className="r">Built-in</th></tr>
              <tr><td>5 Å</td><td className="r">0.00</td><td className="r">+0.75</td><td className="r">0.00</td></tr>
              <tr><td>8 Å</td><td className="r">0.00</td><td className="r">+1.29</td><td className="r">0.00</td></tr>
            </tbody>
          </table>
          <p className="p30 muted"><b style={{ color: "var(--ink)" }}>The same idea scales up:</b> symmetry-aware inputs and smooth cutoffs let per-atom networks model whole molecules and materials.</p>
        </div>
      </div>
      <p className="p24" style={{ position: "absolute", left: 128, bottom: 64, width: 1664, color: "var(--muted2)" }}>
        Behler &amp; Parrinello, <i>Phys. Rev. Lett.</i> 98, 146401 (2007) · Smith, Isayev &amp; Roitberg, <i>Chem. Sci.</i> 8, 3192 (2017) · Hendrycks &amp; Gimpel, arXiv:1606.08415 (2016)
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* 13. Low-lying electronic states                                     */
/* ------------------------------------------------------------------ */
function States() {
  const pts = [
    ["The ground state is a triplet", "X ³Σg⁻ has two unpaired π* electrons, which is why liquid O₂ is paramagnetic.", ""],
    ["Singlets sit just above", "a ¹Δg at 0.98 eV (1.27 µm airglow) and b ¹Σg⁺ at 1.63 eV (762 nm A-band): singlet oxygen.", "orange"],
    ["One network, one surface", "All meet at O(³P) + O(³P). Several states need several outputs, e.g. learned gaps on top of the ground state.", "violet"],
  ];
  return (
    <section className="slide" style={{ paddingBottom: 160 }}>
      <Head eyebrow="Looking ahead" title="One surface of many: O₂’s low-lying states" />
      <div className="row">
        <img className="fig" src={fig("o2_states.png")} style={{ width: 1000, height: 573 }}
          alt="O2 potential curves: triplet ground state X, singlet a 0.98 eV higher, singlet b 1.63 eV higher, all reaching the same O(3P) + O(3P) limit." />
        <div className="col flex1" style={{ gap: 28 }}>
          {pts.map(([h, p, c]) => (
            <div className={`ruled ${c}`} key={h}>
              <p className="h3s">{h}</p>
              <p className="p24 muted">{p}</p>
            </div>
          ))}
        </div>
      </div>
      <p className="p24" style={{ position: "absolute", left: 128, bottom: 64, width: 1664, color: "var(--muted2)" }}>
        Morse curves from spectroscopic constants: K. P. Huber &amp; G. Herzberg, <i>Constants of Diatomic Molecules</i> (1979)
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Deck order + speaker notes                                          */
/* ------------------------------------------------------------------ */
export const SLIDES = [
  { id: "cover", title: "Teaching a neural network the O–O bond", Component: Cover,
    notes: "Today’s goal: show how a small neural network can learn a molecular potential energy surface, using the simplest interesting case, the O–O bond in O2. The orange dots behind the title are “quantum chemistry” samples; the blue curve is what the network learns to draw through them. Audience: chemistry and CS undergraduates, so we’ll keep one foot in each world." },
  { id: "problem", title: "The problem", Component: Problem,
    notes: "A potential energy surface (PES) maps nuclear geometry to electronic energy. For a diatomic it is a 1-D curve V(r); for a molecule with N atoms it has 3N−6 dimensions, which is why sampling it with ab initio methods gets expensive fast. Machine-learned potentials replace the expensive call with a cheap, differentiable function. To keep the focus on the method, we use a Morse fit to the O2 ground state (X ³Σg⁻) as our “quantum chemistry”: Dₑ ≈ 5.21 eV, rₑ ≈ 1.208 Å, a ≈ 2.66 Å⁻¹. 36 points, denser near the well, are the training set. Note that this ground state is a triplet (two unpaired π* electrons); singlet states lie only 0.98 and 1.63 eV above it, which the last slide returns to." },
  { id: "network", title: "Inside the network", Component: Network,
    notes: "Each neuron computes a weighted sum plus a bias, the same thing as linear regression, and then passes it through an activation. Without the activation, stacked layers collapse into one big linear map. GELU (Hendrycks and Gimpel, 2016) multiplies z by the probability Φ(z) that a standard normal variable is below z, so it is a smooth “soft gate”. For chemistry that smoothness matters: we take first derivatives for forces and second derivatives for the harmonic frequency. A ReLU network’s V(r) is piecewise linear, so its force is a step function and its curvature is zero almost everywhere. Architecture: 1 → 16 → 16 → 1, about 320 parameters, trained with Adam." },
  { id: "learn", title: "Watch it learn", Component: Learn,
    notes: "This slide runs a real training loop in the browser: a 1 → 8 → 8 → 1 GELU network, Adam, learning rate 0.01, fitting the 36 Morse points in standard units. It trains to about 4,800 epochs, pauses, then restarts from new random weights. Point out: (1) the green curves are individual last-layer neurons and the blue prediction is their weighted sum plus a bias; (2) the repulsive wall is learned first, then the well, then the tail; (3) RMSE falls toward a few hundredths of an eV." },
  { id: "loss", title: "Loss and optimizer", Component: Loss,
    notes: "The loss is the single number training makes small. Mean squared error is smooth, punishes large errors strongly, and its square root is the RMSE we report. The second term is force matching: autograd gives F = −dV/dr, compared to reference forces; λ = 0.1 Å² weights it. Forces are cheap extra labels (3N per geometry vs one energy) and teach the slope directly: here the force error dropped about 17× and ωₑ moved from 1636 to 1588 cm⁻¹. Backprop goes through a derivative of the network (create_graph=True). Adam uses running averages of each weight’s gradient and squared gradient. Learning rate 0.01, ×0.3 every 2,000 epochs; the spikes are overshooting steps." },
  { id: "batch", title: "One array in, one array out", Component: Batch,
    notes: "Training is full batch: all 36 distances go in as a 36 × 1 array; the same weights act on each row; one autograd call gives 36 forces; everything becomes one loss, one gradient for each of the 321 weights, and one Adam update per epoch. Large data sets use random mini-batches (stochastic gradient descent). After training, pass any array (400 points here) for 400 energies in one forward pass; the gradient of the summed energies gives all 400 forces because each energy depends only on its own distance. In MD you call the model once per time step." },
  { id: "code", title: "The code", Component: CodeSlide,
    notes: "Left, training: reshape to a 36 × 1 column; PES is the 1 → 16 → 16 → 1 GELU network wrapped with standard-units scaling (ångströms in, eV out). Each loop pass is one epoch: zero_grad, requires_grad on r, forward pass, autograd forces with create_graph=True, energy + 0.1 × force MSE, backward, Adam step. The full script also lowers the learning rate at epochs 2,000 and 4,000. Right: eval mode, any grid, one call for energies, one autograd call for forces. This exact code reaches about 0.0008 eV and 0.026 eV/Å RMSE. Scaling box: x = (r − r̄)/s_r and V = s_V·ŷ + V̄ from training statistics only." },
  { id: "results", title: "Results", Component: Results,
    notes: "Forces come from automatic differentiation of the network output with respect to the input coordinate. Differentiating twice at the minimum gives the force constant k, and ω = (1/2πc)·√(k/μ) with μ = 8 amu gives about 1588 cm⁻¹, right at the experimental ~1580 cm⁻¹. That second derivative only exists because GELU is smooth. The dashed Morse curves are hidden under the network curves because the fit is that close." },
  { id: "weights", title: "The fitted network", Component: Weights,
    notes: "The actual trained model drawn from its final parameters. Edge color and thickness show each weight; node color shows each neuron’s bias. The input layer spreads the neurons’ bend points across bond lengths; the middle layer combines them; the output weights have both signs, building a steep wall while cancelling to a flat tail. Ask the audience: would a different random seed give the same picture? No: many weight patterns give the same curve." },
  { id: "gelu", title: "Why GELU", Component: Gelu,
    notes: "Identical networks and training; only the activation differs. dV/dr is a sum of products of weights and σ′(z). For ReLU σ′ is 0 or 1, so forces are a staircase and curvature is zero except at kinks: discontinuous forces break energy conservation in MD, and zero curvature means no frequencies. ReLU here: energy RMSE ≈ 0.09 eV, force RMSE ≈ 1.5 eV/Å, ωₑ undefined. GELU has smooth first and second derivatives and doesn’t saturate like tanh. Production ML potentials use smooth activations such as shifted softplus (SchNet) and SiLU." },
  { id: "outside", title: "Outside the training data", Component: Outside,
    notes: "A trained network is a fixed smooth function: not random, but unconstrained outside the data. Three seeds agree within 0.01 eV from 0.95 to 3.0 Å. Short range: Morse reaches 536 eV at 0.3 Å; the networks give 35–56 eV, a wall about ten times too soft. Long range: the networks climb to 2–25 eV at 10 Å instead of staying at zero. Ensemble disagreement is a cheap uncertainty estimate and drives active learning. Fixes: data where the simulation goes, physics in the model, and range checks." },
  { id: "physics", title: "Physics in the architecture", Component: Physics,
    notes: "Feed the network a Morse-like variable y = exp(−(r − rₑ)/λ), λ = 0.5 Å, which maps r → ∞ to y → 0, and subtract the network’s value at y = 0: the dissociation limit is zero by construction, and the long-range force vanishes too. Behler–Parrinello (2007) and ANI (2017) use symmetry functions and smooth cutoffs for the same reason. Take-home: a small smooth network learns a PES to sub-meV accuracy; autograd gives forces and curvatures for free; build known physics into the architecture and be careful outside the training data." },
  { id: "states", title: "One surface of many", Component: States,
    notes: "Everything so far fit one curve, the ground state of O2, and that ground state is itself a triplet: X ³Σg⁻, with two unpaired electrons in the degenerate π* orbitals, which is why liquid oxygen sticks to a magnet. Just above are two singlets from the same π*² configuration: a ¹Δg at 0.98 eV (7918 cm⁻¹) and b ¹Σg⁺ at 1.63 eV (13195 cm⁻¹), together “singlet oxygen”: a → X emission is the 1.27 µm airglow, b ← X absorption the 762 nm atmospheric A-band; both are spin-forbidden, so the singlets are long-lived. Curves are Morse models from spectroscopic constants sharing the O(³P) + O(³P) limit. ML lesson: one output learns one surface. Excited-state dynamics, photochemistry or intersystem crossing need one output per state, or a ground-state model plus learned gaps (Δ-learning), plus couplings where curves approach or cross: the multi-surface S0/S1/T1 strategy used for Criegee intermediates." },
];
