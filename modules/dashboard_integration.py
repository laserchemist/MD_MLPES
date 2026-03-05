#!/usr/bin/env python3
"""
Dashboard Integration Module

Generates dynamic data for the web dashboard to track computations,
visualizations, and results in real-time.

Functions:
    update_dashboard: Update dashboard with latest computation data
    generate_dashboard_json: Create JSON data file for dashboard
    create_computation_entry: Add new computation to dashboard
    
Author: PSI4-MD Framework
Date: 2025
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import shutil

from .data_formats import load_trajectory


def scan_outputs_directory(output_dir: str) -> Dict:
    """
    Scan outputs directory for completed computations and files.
    
    Args:
        output_dir: Path to outputs directory
        
    Returns:
        Dictionary with computation information
    """
    output_path = Path(output_dir)
    
    data = {
        'molecules': {},
        'computations': {
            'running': [],
            'completed': [],
            'queued': []
        },
        'files': [],
        'statistics': {
            'total_molecules': 0,
            'total_trajectories': 0,
            'total_frames': 0,
            'total_computation_time': 0.0
        },
        'recent_activity': []
    }
    
    # Scan for trajectory files
    trajectory_files = list(output_path.rglob('*.npz')) + \
                      list(output_path.rglob('*.extxyz')) + \
                      list(output_path.rglob('*.h5'))
    
    for traj_file in trajectory_files:
        try:
            # Load trajectory
            traj = load_trajectory(str(traj_file))
            
            # Extract computation info
            molecule_name = traj.metadata.get('molecule', 'unknown')
            method = traj.metadata.get('method', 'unknown')
            basis = traj.metadata.get('basis', 'unknown')
            
            # Create computation entry
            comp_entry = {
                'molecule': molecule_name,
                'method': method,
                'basis': basis,
                'frames': traj.n_frames,
                'energy_range': [float(traj.energies.min()), float(traj.energies.max())],
                'file': str(traj_file.relative_to(output_path)),
                'status': 'completed',
                'timestamp': datetime.fromtimestamp(traj_file.stat().st_mtime).isoformat()
            }
            
            data['computations']['completed'].append(comp_entry)
            
            # Update statistics
            data['statistics']['total_trajectories'] += 1
            data['statistics']['total_frames'] += traj.n_frames
            
            # Add to molecules list
            if molecule_name not in data['molecules']:
                data['molecules'][molecule_name] = {
                    'name': molecule_name,
                    'computations': []
                }
            data['molecules'][molecule_name]['computations'].append(comp_entry)
            
        except Exception as e:
            print(f"Warning: Could not load {traj_file}: {e}")
    
    # Scan for visualization files
    plot_files = list(output_path.rglob('*.png'))
    for plot_file in plot_files:
        file_entry = {
            'name': plot_file.name,
            'path': str(plot_file.relative_to(output_path)),
            'size_kb': plot_file.stat().st_size / 1024,
            'type': 'visualization',
            'timestamp': datetime.fromtimestamp(plot_file.stat().st_mtime).isoformat()
        }
        data['files'].append(file_entry)
    
    # Add trajectory files to files list
    for traj_file in trajectory_files:
        file_entry = {
            'name': traj_file.name,
            'path': str(traj_file.relative_to(output_path)),
            'size_kb': traj_file.stat().st_size / 1024,
            'type': 'trajectory',
            'format': traj_file.suffix[1:].upper(),
            'timestamp': datetime.fromtimestamp(traj_file.stat().st_mtime).isoformat()
        }
        data['files'].append(file_entry)
    
    # Sort files by timestamp (most recent first)
    data['files'].sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Update statistics
    data['statistics']['total_molecules'] = len(data['molecules'])
    
    # Generate recent activity log
    recent_comps = sorted(data['computations']['completed'], 
                         key=lambda x: x['timestamp'], reverse=True)[:5]
    
    for comp in recent_comps:
        activity_entry = {
            'timestamp': comp['timestamp'],
            'message': f"Completed MD for {comp['molecule']} ({comp['frames']} steps, {comp['method']}/{comp['basis']})"
        }
        data['recent_activity'].append(activity_entry)
    
    return data


def generate_dashboard_json(output_dir: str, dashboard_data_file: str = None):
    """
    Generate JSON data file for dashboard.
    
    Args:
        output_dir: Path to outputs directory
        dashboard_data_file: Path to output JSON file (default: outputs/dashboard_data.json)
    """
    # Scan outputs
    data = scan_outputs_directory(output_dir)
    
    # Default output location
    if dashboard_data_file is None:
        dashboard_data_file = Path(output_dir) / 'dashboard_data.json'
    
    # Save JSON
    with open(dashboard_data_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Dashboard data saved to: {dashboard_data_file}")
    print(f"  Total molecules: {data['statistics']['total_molecules']}")
    print(f"  Total trajectories: {data['statistics']['total_trajectories']}")
    print(f"  Total frames: {data['statistics']['total_frames']}")
    print(f"  Files tracked: {len(data['files'])}")
    
    return data


def update_dashboard_html(dashboard_html_path: str, dashboard_data_path: str, output_html_path: str = None):
    """
    Create a copy of the dashboard HTML that loads real data.
    
    Args:
        dashboard_html_path: Path to template dashboard HTML
        dashboard_data_path: Path to dashboard data JSON
        output_html_path: Path for output HTML (default: same directory as data)
    """
    # Read template
    with open(dashboard_html_path, 'r') as f:
        html_content = f.read()
    
    # Add script to load data
    data_loading_script = f"""
    <script>
        // Load dashboard data
        fetch('dashboard_data.json')
            .then(response => response.json())
            .then(data => {{
                updateDashboardWithData(data);
            }})
            .catch(error => {{
                console.error('Error loading dashboard data:', error);
            }});
        
        function updateDashboardWithData(data) {{
            // Update statistics
            document.querySelector('.stat-card:nth-child(1) .value').textContent = data.statistics.total_molecules;
            document.querySelector('.stat-card:nth-child(2) .value').textContent = data.statistics.total_trajectories;
            document.querySelector('.stat-card:nth-child(3) .value').textContent = data.statistics.total_frames.toLocaleString();
            
            // Update recent activity
            const activityLog = document.querySelector('#overview .card:nth-child(2)');
            if (activityLog && data.recent_activity.length > 0) {{
                let activityHTML = '<h3>📊 Recent Activity</h3>';
                data.recent_activity.forEach(activity => {{
                    const timestamp = new Date(activity.timestamp).toLocaleString();
                    activityHTML += `<div class="log-entry">
                        <span class="timestamp">${{timestamp}}</span> - ${{activity.message}}
                    </div>`;
                }});
                activityLog.innerHTML = activityHTML;
            }}
            
            // Update computations table
            const compTable = document.querySelector('#computations table tbody');
            if (compTable && data.computations.completed.length > 0) {{
                compTable.innerHTML = '';
                data.computations.completed.slice(0, 10).forEach(comp => {{
                    const row = document.createElement('tr');
                    const energyMin = (comp.energy_range[0] * 627.509).toFixed(1);
                    const energyMax = (comp.energy_range[1] * 627.509).toFixed(1);
                    row.innerHTML = `
                        <td>${{comp.molecule}}</td>
                        <td>${{comp.method}}/${{comp.basis}}</td>
                        <td>${{comp.frames}}</td>
                        <td>${{energyMin}} to ${{energyMax}}</td>
                        <td><span class="status completed">Completed</span></td>
                    `;
                    compTable.appendChild(row);
                }});
            }}
            
            // Update files table
            const filesTable = document.querySelector('#results table tbody');
            if (filesTable && data.files.length > 0) {{
                filesTable.innerHTML = '';
                data.files.filter(f => f.type === 'trajectory').slice(0, 10).forEach(file => {{
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${{file.name}}</td>
                        <td>${{file.format}}</td>
                        <td>${{file.size_kb.toFixed(2)}} KB</td>
                        <td>-</td>
                        <td><button onclick="window.open('${{file.path}}', '_blank')">📥 Download</button></td>
                    `;
                    filesTable.appendChild(row);
                }});
            }}
        }}
    </script>
    """
    
    # Insert script before closing body tag
    html_content = html_content.replace('</body>', data_loading_script + '\n</body>')
    
    # Default output path
    if output_html_path is None:
        output_html_path = Path(dashboard_data_path).parent / 'dashboard.html'
    
    # Write updated HTML
    with open(output_html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Updated dashboard HTML saved to: {output_html_path}")
    print(f"Open in browser: file://{Path(output_html_path).absolute()}")


def create_live_dashboard(output_dir: str, template_html: str = None):
    """
    Create a live dashboard that shows actual computation data.
    
    Args:
        output_dir: Path to outputs directory
        template_html: Path to dashboard template (default: docs/dashboard.html)
        
    Returns:
        Path to generated dashboard HTML
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Find template
    if template_html is None:
        # Look for template in common locations
        possible_templates = [
            Path(__file__).parent.parent / 'docs' / 'dashboard.html',
            Path('docs/dashboard.html'),
            Path('../docs/dashboard.html')
        ]
        for template in possible_templates:
            if template.exists():
                template_html = str(template)
                break
    
    if template_html is None or not Path(template_html).exists():
        print("Warning: Dashboard template not found. Using basic HTML.")
        return None
    
    # Generate data
    print("Scanning outputs directory...")
    data_file = output_path / 'dashboard_data.json'
    generate_dashboard_json(str(output_path), str(data_file))
    
    # Create live HTML
    print("Creating live dashboard...")
    dashboard_file = output_path / 'dashboard.html'
    update_dashboard_html(template_html, str(data_file), str(dashboard_file))
    
    return dashboard_file


if __name__ == "__main__":
    import sys
    
    print("🌐 Dashboard Integration Module")
    print("=" * 60)
    
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = sys.argv[1]
    else:
        output_dir = '/home/claude/psi4md_framework/outputs'
    
    print(f"\nScanning outputs directory: {output_dir}")
    
    # Create live dashboard
    dashboard_path = create_live_dashboard(output_dir)
    
    if dashboard_path:
        print(f"\n✅ Dashboard created successfully!")
        print(f"\n📂 Files created:")
        print(f"   - {dashboard_path}")
        print(f"   - {dashboard_path.parent / 'dashboard_data.json'}")
        
        print(f"\n🌐 To view dashboard:")
        print(f"   1. Open in browser: file://{dashboard_path.absolute()}")
        print(f"   2. Or run: firefox {dashboard_path}")
        print(f"   3. Or run: google-chrome {dashboard_path}")
    else:
        print("\n❌ Could not create dashboard (template not found)")
        print("   Dashboard template should be at: docs/dashboard.html")
