#!/bin/bash
# Fixed IR Spectroscopy Workflow
# Finds available files automatically

echo "================================================================================"
echo "  IR SPECTROSCOPY WORKFLOW - AUTO-DETECTION"
echo "================================================================================"

# ============================================
# AUTO-DETECT FILES
# ============================================

echo ""
echo "🔍 Searching for ML-PES models and training data..."
echo ""

# Find model files
MODEL_FILES=$(find outputs -name "*mlpes*.pkl" -o -name "*model*.pkl" 2>/dev/null | grep -v "__pycache__")

if [ -z "$MODEL_FILES" ]; then
    echo "❌ No ML-PES model files found!"
    echo ""
    echo "Please ensure you have trained a model first using:"
    echo "  python3 complete_workflow_v2.2.py"
    echo ""
    exit 1
fi

echo "📁 Found ML-PES models:"
echo "$MODEL_FILES" | nl
echo ""

# Find training data
DATA_FILES=$(find outputs -name "*training*.npz" -o -name "*augmented*.npz" 2>/dev/null)

if [ -z "$DATA_FILES" ]; then
    echo "❌ No training data files found!"
    echo ""
    echo "Please ensure you have training data from:"
    echo "  python3 complete_workflow_v2.2.py"
    echo ""
    exit 1
fi

echo "📁 Found training data:"
echo "$DATA_FILES" | nl
echo ""

# ============================================
# SELECT FILES
# ============================================

# Use most recent files by default
MODEL=$(echo "$MODEL_FILES" | tail -1)
DATA=$(echo "$DATA_FILES" | tail -1)

echo "🎯 Auto-selected (most recent):"
echo "  Model: $MODEL"
echo "  Data:  $DATA"
echo ""

# Check if files exist
if [ ! -f "$MODEL" ]; then
    echo "❌ Model file not found: $MODEL"
    exit 1
fi

if [ ! -f "$DATA" ]; then
    echo "❌ Data file not found: $DATA"
    exit 1
fi

# ============================================
# MANUAL OVERRIDE (optional)
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Manual Override (optional)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To use different files, you can:"
echo "  1. Edit this script and set MODEL and DATA manually"
echo "  2. Or press Enter to continue with auto-selected files"
echo ""
read -p "Press Enter to continue or Ctrl+C to abort... "

# ============================================
# STEP 1: Train Dipole Surface
# ============================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Training Dipole Surface"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "This step trains an ML model to predict dipole moments quickly."
echo "Time: ~2-5 minutes"
echo ""

python3 compute_ir_spectrum.py \
    --model "$MODEL" \
    --training-data "$DATA" \
    --train-dipole \
    --equilibration-steps 0 \
    --production-steps 1000 \
    --temp 300 \
    --output outputs/ir_dipole_training

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Dipole training failed!"
    echo ""
    echo "Common issues:"
    echo "  1. Training data doesn't have dipole moments"
    echo "  2. Missing dependencies (scikit-learn, scipy)"
    echo "  3. Incompatible file formats"
    echo ""
    echo "Try running with PSI4 dipoles instead (slower):"
    echo "  python3 compute_ir_spectrum.py --model $MODEL --training-data $DATA --temp 300 --output outputs/ir_test"
    echo ""
    exit 1
fi

echo ""
echo "✅ Dipole surface trained successfully!"
echo "   Location: outputs/ir_dipole_training/dipole_surface.pkl"
echo ""

DIPOLE_MODEL="outputs/ir_dipole_training/dipole_surface.pkl"

# ============================================
# STEP 2: Compute IR Spectrum
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Computing IR Spectrum at 300K"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  • Equilibration: 10000 steps (5 ps)"
echo "  • Production:    50000 steps (25 ps)"
echo "  • Temperature:   300 K"
echo "  • Frequency:     0-4000 cm⁻¹"
echo ""
echo "Time: ~10-15 minutes"
echo ""

python3 compute_ir_spectrum.py \
    --model "$MODEL" \
    --training-data "$DATA" \
    --dipole-model "$DIPOLE_MODEL" \
    --equilibration-steps 10000 \
    --production-steps 50000 \
    --temp 300 \
    --timestep 0.5 \
    --save-every 10 \
    --max-freq 4000 \
    --window hann \
    --output outputs/ir_spectrum_300K

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ IR spectrum calculation failed!"
    echo ""
    echo "Check the error messages above for details."
    echo ""
    exit 1
fi

echo ""
echo "✅ IR spectrum computed successfully!"
echo ""

# ============================================
# RESULTS
# ============================================

echo "================================================================================"
echo "  CALCULATION COMPLETE!"
echo "================================================================================"
echo ""
echo "📁 Output directories:"
echo "   • outputs/ir_dipole_training/    - Dipole surface training"
echo "   • outputs/ir_spectrum_300K/      - IR spectrum results"
echo ""
echo "📊 Visualization files (in ir_spectrum_300K/):"
echo "   • ir_spectrum.png              - Main IR spectrum"
echo "   • dipole_autocorrelation.png   - Autocorrelation function"
echo "   • dipole_evolution.png         - Dipole moment dynamics"
echo "   • energy_trajectory.png        - Energy conservation"
echo "   • ir_dashboard.png             - Comprehensive 6-panel view"
echo ""
echo "📈 Data files:"
echo "   • production_trajectory.npz    - Full MD trajectory with dipoles"
echo "   • ir_spectrum.npz              - Spectrum data (frequencies, intensities)"
echo ""
echo "💡 Next steps:"
echo "   1. View plots: open outputs/ir_spectrum_300K/*.png"
echo "   2. Run at different temperatures: bash run_temperature_series.sh"
echo "   3. Compare to experimental spectrum"
echo "   4. Analyze peak assignments"
echo ""
echo "🎉 Success! Your IR spectrum is ready for analysis."
echo ""
