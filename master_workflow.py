#!/usr/bin/env python3
"""
ML-PES IR Spectrum Master Workflow
===================================
Single menu-driven script that guides you through the complete workflow
and keeps track of all files.

Author: Jonathan
Date: 2026-01-17
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import subprocess


class WorkflowTracker:
    """Track workflow state and files."""
    
    def __init__(self, state_file='workflow_state.json'):
        self.state_file = Path(state_file)
        self.state = self.load_state()
    
    def load_state(self):
        """Load workflow state from file."""
        if self.state_file.exists():
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'trajectory_dir': None,
            'training_data': None,
            'ml_pes_model': None,
            'dipole_model': None,
            'ir_output_dir': None,
            'completed_steps': [],
            'last_updated': None
        }
    
    def save_state(self):
        """Save workflow state to file."""
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    def mark_complete(self, step):
        """Mark a step as complete."""
        if step not in self.state['completed_steps']:
            self.state['completed_steps'].append(step)
        self.save_state()
    
    def is_complete(self, step):
        """Check if a step is complete."""
        return step in self.state['completed_steps']


def print_header(title):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def print_status(tracker):
    """Print current workflow status."""
    print_header("CURRENT STATUS")
    
    steps = [
        ('step1', 'Combine trajectories & compute dipoles', tracker.state.get('training_data')),
        ('step2', 'Train ML-dipole model', tracker.state.get('dipole_model')),
        ('step3', 'Train ML-PES model', tracker.state.get('ml_pes_model')),
        ('step4', 'Compute IR spectrum', tracker.state.get('ir_output_dir'))
    ]
    
    print("\nWorkflow Progress:")
    for step_id, step_name, file_path in steps:
        status = "✅" if tracker.is_complete(step_id) else "⏳"
        print(f"  {status} {step_name}")
        if file_path:
            print(f"     → {file_path}")
    
    print()


def find_trajectory_directories():
    """Find directories containing trajectory files."""
    dirs = []
    for path in Path('outputs').glob('training_with_dipoles_*'):
        if path.is_dir():
            # Check if it has trajectory files
            traj_files = list(path.glob('trajectory_*K.npz'))
            if traj_files:
                dirs.append(path)
    return sorted(dirs, key=lambda x: x.stat().st_mtime, reverse=True)


def find_training_data_files():
    """Find combined training_data.npz files."""
    files = []
    for path in Path('outputs').rglob('training_data.npz'):
        if path.exists():
            files.append(path)
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def find_ml_pes_models():
    """Find trained ML-PES models."""
    files = []
    for pattern in ['mlpes_model*.pkl', 'ml_pes_model*.pkl']:
        for path in Path('outputs').rglob(pattern):
            files.append(path)
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def find_dipole_models():
    """Find trained dipole models."""
    files = list(Path('.').glob('dipole_surface*.pkl'))
    files.extend(Path('outputs').rglob('dipole_surface*.pkl'))
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)


def select_from_list(items, prompt, show_func=None):
    """Interactive selection from a list."""
    if not items:
        print("\n❌ No items found!")
        return None
    
    print(f"\n{prompt}")
    print("-" * 80)
    
    for i, item in enumerate(items, 1):
        if show_func:
            print(f"  [{i}] {show_func(item)}")
        else:
            print(f"  [{i}] {item}")
    
    print(f"  [0] Cancel")
    
    while True:
        try:
            choice = input(f"\nSelect [0-{len(items)}]: ").strip()
            if not choice:
                continue
            choice = int(choice)
            if choice == 0:
                return None
            if 1 <= choice <= len(items):
                return items[choice - 1]
            print(f"Invalid choice. Please enter 0-{len(items)}")
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            return None


def step1_combine_and_compute_dipoles(tracker):
    """Step 1: Combine trajectories and compute dipoles."""
    print_header("STEP 1: Combine Trajectories & Compute Dipoles")
    
    # Find trajectory directories
    traj_dirs = find_trajectory_directories()
    
    if not traj_dirs:
        print("\n❌ No trajectory directories found!")
        print("   Run generate_training_data_COMPLETE_WORKING.py first")
        return False
    
    # Select directory
    traj_dir = select_from_list(
        traj_dirs,
        "Select trajectory directory:",
        lambda x: f"{x.name} ({len(list(x.glob('trajectory_*K.npz')))} trajectories)"
    )
    
    if not traj_dir:
        return False
    
    # Check if already combined
    combined_file = traj_dir / 'training_data.npz'
    if combined_file.exists():
        print(f"\n✓ Found existing training_data.npz")
        use_existing = input("Use existing file? [Y/n]: ").strip().lower()
        if use_existing != 'n':
            tracker.state['trajectory_dir'] = str(traj_dir)
            tracker.state['training_data'] = str(combined_file)
            tracker.mark_complete('step1')
            return True
    
    # Run combine script
    print(f"\n🚀 Running combine_and_compute_dipoles_WORKING.py...")
    print(f"   This will take ~10 minutes")
    
    cmd = [
        'python3',
        'combine_and_compute_dipoles_WORKING.py',
        str(traj_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        
        # Check if file was created
        if combined_file.exists():
            print("\n✅ Training data with dipoles created!")
            tracker.state['trajectory_dir'] = str(traj_dir)
            tracker.state['training_data'] = str(combined_file)
            tracker.mark_complete('step1')
            return True
        else:
            print("\n❌ training_data.npz not created!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running combine script: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ combine_and_compute_dipoles_WORKING.py not found!")
        print("   Make sure it's in the current directory")
        return False


def step2_train_dipole_model(tracker):
    """Step 2: Train ML-dipole model."""
    print_header("STEP 2: Train ML-Dipole Model")
    
    # Check for training data
    if not tracker.state.get('training_data'):
        print("\n⚠️  No training data found!")
        
        # Try to find existing files
        training_files = find_training_data_files()
        if training_files:
            training_data = select_from_list(
                training_files,
                "Select training data file:",
                lambda x: f"{x} ({x.stat().st_size // 1024} KB)"
            )
            if training_data:
                tracker.state['training_data'] = str(training_data)
            else:
                print("\n❌ No training data selected")
                print("   Complete Step 1 first")
                return False
        else:
            print("   Complete Step 1 first")
            return False
    
    # Output filename
    output_file = 'dipole_surface_ammonia.pkl'
    if Path(output_file).exists():
        print(f"\n✓ Found existing {output_file}")
        use_existing = input("Use existing file? [Y/n]: ").strip().lower()
        if use_existing != 'n':
            tracker.state['dipole_model'] = output_file
            tracker.mark_complete('step2')
            return True
    
    # Run training
    print(f"\n🚀 Training ML-dipole model...")
    print(f"   This will take ~1-2 minutes")
    
    cmd = [
        'python3',
        'compute_ir_workflow_FIXED.py',
        '--train-dipole',
        '--training-data', tracker.state['training_data'],
        '--dipole-output', output_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        
        if Path(output_file).exists():
            print("\n✅ ML-dipole model trained!")
            tracker.state['dipole_model'] = output_file
            tracker.mark_complete('step2')
            return True
        else:
            print(f"\n❌ {output_file} not created!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error training model: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ compute_ir_workflow_FIXED.py not found!")
        return False


def step3_train_ml_pes(tracker):
    """Step 3: Train ML-PES model."""
    print_header("STEP 3: Train ML-PES Model")
    
    # Check for existing models
    existing_models = find_ml_pes_models()
    if existing_models:
        print("\n✓ Found existing ML-PES models:")
        ml_pes = select_from_list(
            existing_models,
            "Select ML-PES model (or 0 to train new):",
            lambda x: f"{x} ({x.stat().st_size // 1024} KB)"
        )
        
        if ml_pes:
            tracker.state['ml_pes_model'] = str(ml_pes)
            tracker.mark_complete('step3')
            return True
    
    # Train new model
    print("\n🚀 Launching ML-PES training...")
    print("   Use complete_workflow_v2.2.py manually:")
    print(f"\n   python3 complete_workflow_v2.2.py")
    print(f"\n   Then:")
    print(f"   1. Select [1] Work with existing training data")
    print(f"   2. Select [2] Scan directories")
    print(f"   3. Select your training_data.npz")
    print(f"   4. Select [2] Energy + forces training")
    
    input("\n   Press Enter when ML-PES training is complete...")
    
    # Find the newly created model
    models = find_ml_pes_models()
    if models:
        print("\n✓ Found ML-PES models:")
        model = select_from_list(
            models,
            "Select the model you just trained:",
            lambda x: f"{x}"
        )
        
        if model:
            tracker.state['ml_pes_model'] = str(model)
            tracker.mark_complete('step3')
            return True
    
    print("\n❌ No ML-PES model found")
    return False


def step4_compute_ir_spectrum(tracker):
    """Step 4: Compute IR spectrum."""
    print_header("STEP 4: Compute IR Spectrum")
    
    # Check prerequisites
    if not tracker.state.get('ml_pes_model'):
        print("\n❌ No ML-PES model found!")
        print("   Complete Step 3 first")
        return False
    
    if not tracker.state.get('dipole_model'):
        print("\n❌ No dipole model found!")
        print("   Complete Step 2 first")
        return False
    
    # Parameters
    print("\nIR Spectrum Parameters:")
    temp = input("  Temperature (K) [300]: ").strip() or "300"
    steps = input("  MD steps [50000]: ").strip() or "50000"
    output_dir = input("  Output directory [ir_spectrum_output]: ").strip() or "ir_spectrum_output"
    
    # Run computation
    print(f"\n🚀 Computing IR spectrum...")
    print(f"   This will take ~3-5 minutes")
    
    cmd = [
        'python3',
        'compute_ir_workflow_FIXED.py',
        '--ml-pes', tracker.state['ml_pes_model'],
        '--dipole-model', tracker.state['dipole_model'],
        '--training-data', tracker.state.get('training_data', ''),
        '--temp', temp,
        '--steps', steps,
        '--output-dir', output_dir
    ]
    
    # Remove empty training-data if not set
    if not tracker.state.get('training_data'):
        cmd = [c for c in cmd if c != '--training-data' and c != '']
    
    try:
        result = subprocess.run(cmd, check=True)
        
        output_path = Path(output_dir)
        if (output_path / 'ir_spectrum.png').exists():
            print(f"\n✅ IR spectrum computed!")
            print(f"\n📊 Results:")
            print(f"   Location: {output_path}/")
            print(f"   - ir_spectrum.png (plot)")
            print(f"   - ir_spectrum.txt (data)")
            print(f"   - md_trajectory.npz (trajectory)")
            print(f"   - dipole_acf.png (autocorrelation)")
            
            tracker.state['ir_output_dir'] = str(output_path)
            tracker.mark_complete('step4')
            return True
        else:
            print("\n❌ IR spectrum not created!")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error computing spectrum: {e}")
        return False
    except FileNotFoundError:
        print("\n❌ compute_ir_workflow_FIXED.py not found!")
        return False


def main_menu():
    """Main menu."""
    tracker = WorkflowTracker()
    
    while True:
        print_header("ML-PES IR SPECTRUM WORKFLOW")
        print_status(tracker)
        
        print("Menu:")
        print("  [1] Combine trajectories & compute dipoles")
        print("  [2] Train ML-dipole model")
        print("  [3] Train ML-PES model")
        print("  [4] Compute IR spectrum")
        print("  [5] View current files")
        print("  [0] Exit")
        
        choice = input("\nSelect [0-5]: ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            step1_combine_and_compute_dipoles(tracker)
        elif choice == '2':
            step2_train_dipole_model(tracker)
        elif choice == '3':
            step3_train_ml_pes(tracker)
        elif choice == '4':
            step4_compute_ir_spectrum(tracker)
        elif choice == '5':
            print_status(tracker)
            input("\nPress Enter to continue...")
        else:
            print("\n❌ Invalid choice")


if __name__ == '__main__':
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
