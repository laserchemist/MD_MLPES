# ============================================
# STEP 1: Generate Training Data with Dipoles
# ============================================
# Modify direct_md.py to save dipoles, or:
# Use existing training data if it has dipoles

# Check if your training data has dipoles:
python3 -c "
from modules.data_formats import load_trajectory
data = load_trajectory('training_data.npz')
print('Has dipoles:', 'dipoles' in data.metadata)
"

# If no dipoles, need to add them (see Section below)

# ============================================
# STEP 2: Train Dipole Surface (2-5 minutes)
# ============================================
python3 compute_ir_spectrum.py \
    --model ml_model.pkl \
    --training-data training_data.npz \
    --train-dipole \
    --equilibration-steps 0 \
    --production-steps 1000 \
    --temp 300

# This creates: dipole_surface.pkl

# ============================================
# STEP 3: Full IR Calculation (10-20 minutes)
# ============================================
python3 compute_ir_spectrum.py \
    --model ml_model.pkl \
    --training-data training_data.npz \
    --dipole-model dipole_surface.pkl \
    --equilibration-steps 10000 \
    --production-steps 50000 \
    --temp 300 \
    --timestep 0.5 \
    --max-freq 4000 \
    --output ir_300K

# Output:
# - ir_300K/production_trajectory.npz
# - ir_300K/ir_spectrum.npz
# - ir_300K/ir_spectrum.png
# - ir_300K/dipole_autocorrelation.png
# - ir_300K/dipole_evolution.png
# - ir_300K/energy_trajectory.png
# - ir_300K/ir_dashboard.png

# ============================================
# DONE! ✅
# ============================================
