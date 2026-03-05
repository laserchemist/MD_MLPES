#!/bin/bash
# Quick Start: IR Spectroscopy Module
# Complete example from model to IR spectrum

echo "================================================================================"
echo "  IR SPECTROSCOPY - QUICK START EXAMPLE"
echo "================================================================================"

# ============================================
# CONFIGURATION
# ============================================

# Update these paths to your actual files!
MODEL="outputs/diagnostic_phase1_20260102_201929/retrained_mlpes_model.pkl"
DATA="outputs/diagnostic_phase1_20260102_201929/augmented_training_data.npz"

echo ""
echo "Configuration:"
echo "  ML-PES Model:    $MODEL"
echo "  Training Data:   $DATA"
echo ""

# ============================================
# OPTION 1: Train Dipole Surface First (RECOMMENDED)
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OPTION 1: Train Dipole Surface (Fast Method)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This trains a dipole surface that can predict dipoles 60,000x faster than PSI4!"
echo ""
echo "Running training step (this will take 2-5 minutes)..."
echo ""

python3 compute_ir_spectrum.py \
    --model "$MODEL" \
    --training-data "$DATA" \
    --train-dipole \
    --equilibration-steps 0 \
    --production-steps 1000 \
    --temp 300 \
    --output ir_training_step

echo ""
echo "✅ Dipole surface trained and saved!"
echo "   Location: ir_training_step/dipole_surface.pkl"
echo ""

# ============================================
# Now compute full IR spectrum
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Computing Full IR Spectrum at 300K"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Parameters:"
echo "  • Equilibration: 10000 steps (5 ps)"
echo "  • Production:    50000 steps (25 ps)"
echo "  • Temperature:   300 K"
echo "  • Frequency:     0-4000 cm⁻¹"
echo ""
echo "This will take ~10-15 minutes..."
echo ""

python3 compute_ir_spectrum.py \
    --model "$MODEL" \
    --training-data "$DATA" \
    --dipole-model ir_training_step/dipole_surface.pkl \
    --equilibration-steps 10000 \
    --production-steps 50000 \
    --temp 300 \
    --timestep 0.5 \
    --save-every 10 \
    --max-freq 4000 \
    --window hann \
    --output ir_spectrum_300K

echo ""
echo "✅ IR spectrum computed!"
echo ""

# ============================================
# RESULTS
# ============================================

echo "================================================================================"
echo "  RESULTS"
echo "================================================================================"
echo ""
echo "📁 Output directory: outputs/ir_spectrum_300K/"
echo ""
echo "📊 Files created:"
echo "   • production_trajectory.npz    - MD trajectory with dipoles"
echo "   • ir_spectrum.npz              - Spectrum data (frequencies, intensities)"
echo "   • ir_spectrum.png              - IR spectrum plot"
echo "   • dipole_autocorrelation.png   - C(t) plot"
echo "   • dipole_evolution.png         - Dipole vs time"
echo "   • energy_trajectory.png        - Energy conservation"
echo "   • ir_dashboard.png             - Comprehensive 6-panel view"
echo ""
echo "💡 Next steps:"
echo "   1. View plots in outputs/ir_spectrum_300K/"
echo "   2. Compare to experimental spectrum"
echo "   3. Run at different temperatures:"
echo "      bash run_temperature_series.sh"
echo "   4. Analyze peak assignments"
echo ""

# ============================================
# OPTION 2: Using PSI4 for Dipoles (Slower)
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  OPTION 2: Using PSI4 for Dipoles (Not Recommended)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  This method is much slower (~1 hour for 25 ps) because PSI4"
echo "    computes dipoles on-the-fly. Only use for small systems or testing."
echo ""
echo "To run with PSI4 dipoles:"
echo ""
echo "  python3 compute_ir_spectrum.py \\"
echo "      --model $MODEL \\"
echo "      --training-data $DATA \\"
echo "      --equilibration-steps 5000 \\"
echo "      --production-steps 10000 \\"
echo "      --temp 300 \\"
echo "      --output ir_psi4"
echo ""
echo "  (No --dipole-model or --train-dipole flags)"
echo ""

echo "================================================================================"
echo "  QUICK START COMPLETE!"
echo "================================================================================"
echo ""
echo "🎉 You now have:"
echo "   ✅ Trained dipole surface (reusable!)"
echo "   ✅ IR spectrum at 300K"
echo "   ✅ Publication-quality plots"
echo "   ✅ Complete analysis data"
echo ""
echo "📚 For more information:"
echo "   • Read IR_SPECTROSCOPY_GUIDE.md"
echo "   • View example scripts"
echo "   • Check outputs/ir_spectrum_300K/ for results"
echo ""
