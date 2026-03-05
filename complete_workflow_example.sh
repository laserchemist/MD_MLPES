#!/bin/bash
# Complete ML-PES Workflow Example
# From listing models to production MD to visualization

echo "================================================================================"
echo "  COMPLETE ML-PES WORKFLOW - FROM LISTING TO VISUALIZATION"
echo "================================================================================"

# ============================================
# STEP 1: List available models and data
# ============================================
echo ""
echo "STEP 1: Listing available models and training data..."
echo "--------------------------------------------------------------------------------"
python3 list_models.py

# This shows you:
# - All ML-PES models with RMSE values
# - All training datasets with config counts
# - Refinement status (original vs refined)
# - Recommended model/data combination

# ============================================
# STEP 2: Run production MD
# ============================================
echo ""
echo "STEP 2: Running production MD with best model..."
echo "--------------------------------------------------------------------------------"

# Use the recommended model and data from Step 1
# Update these paths based on list_models.py output!
MODEL="outputs/diagnostic_phase1_20260102_201929/retrained_mlpes_model.pkl"
DATA="outputs/diagnostic_phase1_20260102_201929/augmented_training_data.npz"

python3 simple_production_md.py \
    --model "$MODEL" \
    --training-data "$DATA" \
    --temp 300 \
    --steps 20000 \
    --timestep 0.5 \
    --output production_md_300K.npz

# This creates:
# - production_md_300K.npz (trajectory file)
# - 10 ps of MD at 300K
# - Using your refined ML-PES model

# ============================================
# STEP 3: Visualize results
# ============================================
echo ""
echo "STEP 3: Creating visualizations..."
echo "--------------------------------------------------------------------------------"

python3 visualize_trajectory.py production_md_300K.npz --all

# This creates:
# - production_md_300K_analysis/ directory
# - Energy trajectory plots
# - Temperature evolution
# - Force distributions
# - 3D structure snapshots
# - Summary dashboard

# ============================================
# STEP 4: View statistics
# ============================================
echo ""
echo "STEP 4: Displaying trajectory statistics..."
echo "--------------------------------------------------------------------------------"

python3 visualize_trajectory.py production_md_300K.npz

# This prints:
# - Number of frames
# - Energy statistics
# - Temperature statistics
# - Force statistics

# ============================================
# COMPLETE!
# ============================================
echo ""
echo "================================================================================"
echo "  WORKFLOW COMPLETE!"
echo "================================================================================"
echo ""
echo "📁 Results:"
echo "   • MD trajectory: production_md_300K.npz"
echo "   • Visualizations: production_md_300K_analysis/"
echo ""
echo "💡 Next steps:"
echo "   • View plots in production_md_300K_analysis/"
echo "   • Run additional MD at different temperatures"
echo "   • Further analyze trajectory data"
echo "   • Register in database (optional)"
echo ""
