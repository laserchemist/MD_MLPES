# Teach_O2_NN: Teaching a Neural Network the O–O Bond

A 13-slide React presentation on neural-network potential energy surfaces (PES), built around the simplest
interesting case: the O–O bond in O₂. It goes from a single neuron to forces, vibrational frequencies,
extrapolation failure, and building physics into the architecture. Audience: undergraduate chemistry and CS.

*Jonathan M. Smith, Temple University. Slides and worked example built in collaboration with Claude (Anthropic).*

## View it

The `site/` folder is the finished, static presentation (no install needed):

- **GitHub Pages:** if Pages serves this repository from `main` / root, open
  `https://laserchemist.github.io/MD_MLPES/Teach_O2_NN/site/`
- **Locally:** `cd Teach_O2_NN/site && python3 -m http.server` and open <http://localhost:8000>
  (opening `index.html` straight from disk can be blocked by the browser's module security).

**Controls:** ← / → (or space) to move, **N** for speaker notes, **F** for full screen. The slide number is in the
URL (`#4` jumps to the live training slide).

## Slides

| # | Slide | Content |
|---|-------|---------|
| 1 | Cover | Title, presenter |
| 2 | The problem | Why learn a PES; Morse model of O₂ (Dₑ 5.21 eV, rₑ 1.208 Å, a 2.66 Å⁻¹) |
| 3 | Inside the network | Neuron = weigh, add, bend; GELU vs ReLU; 1 → 16 → 16 → 1 architecture |
| 4 | Watch it learn | **Live** in-browser training of a 1 → 8 → 8 → 1 GELU net (plain JS, hand-written backprop + Adam) |
| 5 | Loss and optimizer | Energy + force MSE, backprop, Adam; energy-only vs force-matched comparison |
| 6 | One array in, one array out | Full-batch training, batched read-out, why one autograd call gives every force |
| 7 | Annotated code | PyTorch training loop and read-out; standard-units scaling |
| 8 | Results | Energy RMSE 0.0009 eV, force RMSE 0.023 eV/Å, rₑ 1.209 Å, ωₑ 1588 cm⁻¹ (expt ≈ 1580) |
| 9 | The fitted network | Color-coded weights and biases of the trained model |
| 10 | Why GELU | Same net with ReLU vs GELU: energy, force, curvature |
| 11 | Outside the data | Three seeds agree inside 0.95–3.0 Å, diverge outside |
| 12 | Physics in the architecture | V(∞) = 0 by construction with V = g(y) − g(0) |
| 13 | One surface of many | O₂’s triplet ground state and low-lying singlets (a ¹Δg, b ¹Σg⁺); multi-state outlook |

## Edit and rebuild

```bash
cd Teach_O2_NN
npm install
npm run dev      # live-reloading preview at http://localhost:5173
npm run build    # writes the static presentation to site/
```

- Slide content and speaker notes: `src/slides.jsx` (one component per slide, order in `SLIDES`)
- Viewer (scaling, keys, notes): `src/App.jsx`
- Live training animation: `src/TrainingCanvas.jsx`
- Styles and palette: `src/styles.css`

The 1920 × 1080 slide canvas is scaled to fit any window.

## Figures and the Python behind them

All figures in `public/figures/` come from real PyTorch runs in `scripts/` (PyTorch + matplotlib):

| Script | Produces |
|--------|----------|
| `figs.py` | `fit_force.png` (energy + force fit), `dissociation.png` (plain vs V(∞)=0 model) |
| `figs2.py` | `gelu_vs_relu.png`, `network_weights.png` |
| `lossfig.py` | `loss_curve.png`, energy-only vs force-matched numbers |
| `outside.py` | `outside.png` (three seeds outside the training range) |
| `slidecode.py` | the exact code shown on slide 7, with RMSE check and extrapolation table |
| `states.py` | `o2_states.png` (X, a, b states of O₂ as Morse curves) |
| `anim.py` | `public/media/O2_GELU_training.gif` / `.mp4` (training animation for PowerPoint or Keynote) |

Units here are eV and Å (a teaching example on a Morse curve), not the Hartree conventions of the main
MD_MLPES pipeline.

## References

- W. S. McCulloch & W. Pitts, *Bull. Math. Biophys.* **5**, 115 (1943)
- J. Behler & M. Parrinello, *Phys. Rev. Lett.* **98**, 146401 (2007)
- J. S. Smith, O. Isayev & A. E. Roitberg, *Chem. Sci.* **8**, 3192 (2017)
- D. Hendrycks & K. Gimpel, arXiv:1606.08415 (2016)
- K. P. Huber & G. Herzberg, *Constants of Diatomic Molecules* (1979)
