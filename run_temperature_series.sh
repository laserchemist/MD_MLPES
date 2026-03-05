#!/bin/bash
# Run IR Spectroscopy at Multiple Temperatures

echo "================================================================================"
echo "  IR SPECTROSCOPY - TEMPERATURE SERIES"
echo "================================================================================"

# ============================================
# CONFIGURATION
# ============================================

# Update these paths!
MODEL="outputs/diagnostic_phase1_20260102_201929/retrained_mlpes_model.pkl"
DATA="outputs/diagnostic_phase1_20260102_201929/augmented_training_data.npz"
DIPOLE_MODEL="ir_training_step/dipole_surface.pkl"

# Temperature range
TEMPS=(100 200 300 400 500)

echo ""
echo "Configuration:"
echo "  ML-PES Model:     $MODEL"
echo "  Training Data:    $DATA"
echo "  Dipole Surface:   $DIPOLE_MODEL"
echo "  Temperatures:     ${TEMPS[@]} K"
echo ""

# Check if dipole model exists
if [ ! -f "$DIPOLE_MODEL" ]; then
    echo "⚠️  Dipole surface not found: $DIPOLE_MODEL"
    echo ""
    echo "Training dipole surface first..."
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
    echo "✅ Dipole surface trained!"
    echo ""
fi

# ============================================
# RUN TEMPERATURE SERIES
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Running IR Calculations"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for TEMP in "${TEMPS[@]}"; do
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo "  Temperature: $TEMP K"
    echo "─────────────────────────────────────────────────────────────────────────────"
    echo ""
    
    python3 compute_ir_spectrum.py \
        --model "$MODEL" \
        --training-data "$DATA" \
        --dipole-model "$DIPOLE_MODEL" \
        --equilibration-steps 10000 \
        --production-steps 50000 \
        --temp $TEMP \
        --timestep 0.5 \
        --save-every 10 \
        --max-freq 4000 \
        --window hann \
        --output ir_${TEMP}K
    
    echo ""
    echo "✅ Completed: $TEMP K"
    echo ""
done

# ============================================
# CREATE COMPARISON PLOT
# ============================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Creating Temperature Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create comparison plot
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

fig, ax = plt.subplots(figsize=(12, 8))

temps = [100, 200, 300, 400, 500]
colors = plt.cm.coolwarm(np.linspace(0, 1, len(temps)))

for temp, color in zip(temps, colors):
    spectrum_file = Path(f'outputs/ir_{temp}K/ir_spectrum.npz')
    
    if spectrum_file.exists():
        data = np.load(spectrum_file)
        frequencies = data['frequencies']
        intensities = data['intensities']
        
        # Normalize for comparison
        intensities = intensities / intensities.max()
        
        # Offset for clarity
        offset = temps.index(temp) * 0.2
        ax.plot(frequencies, intensities + offset, 
               label=f'{temp} K', color=color, linewidth=1.5, alpha=0.8)

ax.set_xlabel('Frequency (cm⁻¹)', fontsize=14, fontweight='bold')
ax.set_ylabel('Intensity (offset for clarity)', fontsize=14, fontweight='bold')
ax.set_title('IR Spectra - Temperature Dependence', fontsize=16, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4000)

plt.tight_layout()
plt.savefig('outputs/ir_temperature_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Comparison plot saved: outputs/ir_temperature_comparison.png")
EOF

# ============================================
# SUMMARY
# ============================================

echo ""
echo "================================================================================"
echo "  TEMPERATURE SERIES COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Results:"
for TEMP in "${TEMPS[@]}"; do
    echo "   • outputs/ir_${TEMP}K/"
done
echo ""
echo "📈 Comparison plot:"
echo "   • outputs/ir_temperature_comparison.png"
echo ""
echo "💡 Analysis suggestions:"
echo "   1. Compare peak positions across temperatures"
echo "   2. Look for temperature-dependent peak broadening"
echo "   3. Identify hot bands (higher T) vs cold bands (lower T)"
echo "   4. Check for conformational changes with temperature"
echo ""
echo "📚 Each directory contains:"
echo "   • ir_spectrum.png              - Individual spectrum"
echo "   • dipole_autocorrelation.png   - Autocorrelation function"
echo "   • dipole_evolution.png         - Dipole dynamics"
echo "   • energy_trajectory.png        - Energy conservation"
echo "   • ir_dashboard.png             - Complete analysis"
echo ""
echo "🎊 Temperature study complete! Compare spectra to observe thermal effects."
echo ""
