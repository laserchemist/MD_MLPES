"""
PSI4-MD Framework - Core Modules

This package contains the core functionality for:
- Test molecule library (test_molecules.py)
- Data format handling (data_formats.py)
- Direct molecular dynamics (direct_md.py)
- Machine learning PES (ml_pes.py)
- Visualization (visualization.py)
- Dashboard integration (dashboard_integration.py)
"""

__version__ = '1.0.0'
__author__ = 'PSI4-MD Framework'

# Import main components for easy access
from .test_molecules import (
    TestMolecule,
    get_molecule,
    get_all_molecules,
    add_random_displacement
)

from .data_formats import (
    TrajectoryData,
    save_trajectory,
    load_trajectory,
    convert_format
)

from .direct_md import (
    DirectMDConfig,
    DirectMDRunner,
    run_direct_md,
    initialize_velocities
)

# Optional imports (may not have dependencies)
try:
    from .ml_pes import (
        MLPESConfig,
        MLPESTrainer,
        train_pes,
        evaluate_model
    )
except ImportError:
    pass

try:
    from .visualization import (
        TrajectoryVisualizer,
        plot_training_curves
    )
except ImportError:
    pass

try:
    from .dashboard_integration import (
        create_live_dashboard,
        generate_dashboard_json
    )
except ImportError:
    pass

__all__ = [
    # Test molecules
    'TestMolecule',
    'get_molecule',
    'get_all_molecules',
    'add_random_displacement',
    
    # Data formats
    'TrajectoryData',
    'save_trajectory',
    'load_trajectory',
    'convert_format',
    
    # Direct MD
    'DirectMDConfig',
    'DirectMDRunner',
    'run_direct_md',
    'initialize_velocities',
    
    # ML-PES (if available)
    'MLPESConfig',
    'MLPESTrainer',
    'train_pes',
    'evaluate_model',
    
    # Visualization (if available)
    'TrajectoryVisualizer',
    'plot_training_curves',
    
    # Dashboard (if available)
    'create_live_dashboard',
    'generate_dashboard_json',
]
