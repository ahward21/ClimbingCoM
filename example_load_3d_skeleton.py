"""
Example: Load and Visualize 3D Skeleton Data from ClimbingEst

This script demonstrates how to load and use the 3D skeleton data exported
by the ClimbingEst system. The data can be used for:
- 3D visualization in Blender, Three.js, etc.
- Biomechanical analysis
- Creating animated climbing tutorials
- Research and analysis

Requirements:
- numpy
- matplotlib (for visualization)
- json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_skeleton_json(json_path):
    """Load 3D skeleton data from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_skeleton_npz(npz_path):
    """Load 3D skeleton data from compressed NumPy format."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        'keypoints_3d': data['keypoints_3d'],
        'keypoints_2d': data['keypoints_2d'],
        'com_trajectory_3d': data['com_trajectory_3d'],
        'joint_names': data['joint_names'].tolist(),
        'skeleton_connections': data['skeleton_connections'].tolist(),
        'fps': float(data['fps']),
        'image_size': data['image_size'].tolist()
    }

def plot_skeleton_3d(kpts_3d, connections, joint_names=None, title="3D Skeleton"):
    """
    Plot a single frame's 3D skeleton.
    
    Args:
        kpts_3d: [17, 3] array of joint positions
        connections: List of (i, j) tuples for skeleton connections
        joint_names: Optional list of joint names
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2], 
               c='red', s=50, depthshade=True)
    
    # Plot skeleton connections
    for i, j in connections:
        ax.plot([kpts_3d[i, 0], kpts_3d[j, 0]],
                [kpts_3d[i, 1], kpts_3d[j, 1]],
                [kpts_3d[i, 2], kpts_3d[j, 2]],
                c='blue', linewidth=2)
    
    # Add joint labels
    if joint_names:
        for idx, name in enumerate(joint_names):
            ax.text(kpts_3d[idx, 0], kpts_3d[idx, 1], kpts_3d[idx, 2], 
                    name[:3], fontsize=8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_title(title)
    
    # Equal aspect ratio
    max_range = np.array([kpts_3d[:, 0].max() - kpts_3d[:, 0].min(),
                          kpts_3d[:, 1].max() - kpts_3d[:, 1].min(),
                          kpts_3d[:, 2].max() - kpts_3d[:, 2].min()]).max() / 2.0
    mid_x = (kpts_3d[:, 0].max() + kpts_3d[:, 0].min()) * 0.5
    mid_y = (kpts_3d[:, 1].max() + kpts_3d[:, 1].min()) * 0.5
    mid_z = (kpts_3d[:, 2].max() + kpts_3d[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    return fig, ax

def plot_com_trajectory_3d(com_trajectory, title="3D Center of Mass Trajectory"):
    """
    Plot the 3D Center of Mass trajectory.
    
    Args:
        com_trajectory: [N, 3] array of CoM positions
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create color gradient based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(com_trajectory)))
    
    # Plot trajectory line
    for i in range(len(com_trajectory) - 1):
        ax.plot([com_trajectory[i, 0], com_trajectory[i+1, 0]],
                [com_trajectory[i, 1], com_trajectory[i+1, 1]],
                [com_trajectory[i, 2], com_trajectory[i+1, 2]],
                c=colors[i], linewidth=2)
    
    # Mark start and end
    ax.scatter(*com_trajectory[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(*com_trajectory[-1], c='red', s=100, marker='s', label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y (Depth)')
    ax.set_zlabel('Z (Height)')
    ax.set_title(title)
    ax.legend()
    
    return fig, ax

def export_to_blender_script(json_path, output_script_path):
    """
    Generate a Blender Python script to import the skeleton data.
    
    This creates a script that can be run in Blender to animate the skeleton.
    """
    data = load_skeleton_json(json_path)
    
    script = '''"""
Blender Import Script for ClimbingEst 3D Skeleton Data
Generated from: {json_path}

To use:
1. Open Blender
2. Switch to Scripting workspace
3. Paste this script
4. Click "Run Script"
"""

import bpy
import json

# Load skeleton data
skeleton_data = {skeleton_data}

# Create armature
bpy.ops.object.armature_add(enter_editmode=True, location=(0, 0, 0))
armature = bpy.context.object
armature.name = "ClimbingSkeleton"

# Get edit bones
edit_bones = armature.data.edit_bones

# Clear default bone
for bone in edit_bones:
    edit_bones.remove(bone)

# Create bones for each connection
joint_names = skeleton_data['joint_names']
connections = skeleton_data['skeleton_connections']
first_frame = skeleton_data['frames'][0]

# Create bones
for i, (start_idx, end_idx) in enumerate(connections):
    start_name = joint_names[start_idx]
    end_name = joint_names[end_idx]
    
    bone = edit_bones.new(f"{{start_name}}_to_{{end_name}}")
    
    start_pos = first_frame['joints'][start_name]
    end_pos = first_frame['joints'][end_name]
    
    bone.head = (start_pos['x'], start_pos['y'], start_pos['z'])
    bone.tail = (end_pos['x'], end_pos['y'], end_pos['z'])

bpy.ops.object.mode_set(mode='OBJECT')

# Create animation
fps = skeleton_data['metadata']['fps']
bpy.context.scene.render.fps = int(fps)

# Set up animation keys
for frame_data in skeleton_data['frames']:
    frame_id = frame_data['frame_id']
    bpy.context.scene.frame_set(frame_id)
    
    # Insert keyframes for each bone
    bpy.ops.object.mode_set(mode='POSE')
    # ... (keyframe logic would go here)
    bpy.ops.object.mode_set(mode='OBJECT')

print(f"Imported {{len(skeleton_data['frames'])}} frames of skeleton animation")
'''.format(json_path=json_path, skeleton_data=json.dumps(data, indent=2))
    
    with open(output_script_path, 'w') as f:
        f.write(script)
    
    print(f"Blender import script saved to: {output_script_path}")

def analyze_climbing_metrics(data):
    """
    Analyze climbing metrics from skeleton data.
    
    Returns dict with:
    - total_duration: Duration in seconds
    - com_height_change: Vertical distance climbed
    - max_velocity: Maximum vertical velocity
    - avg_velocity: Average vertical velocity
    """
    if isinstance(data, str):
        data = load_skeleton_json(data)
    
    frames = data['frames']
    fps = data['metadata']['fps']
    
    # Extract CoM trajectory
    com_z = [f['center_of_mass']['z'] for f in frames]
    
    # Calculate metrics
    total_duration = len(frames) / fps
    com_height_change = com_z[-1] - com_z[0]
    
    # Velocity (derivative of position)
    velocities = np.diff(com_z) * fps
    max_velocity = np.max(np.abs(velocities))
    avg_velocity = np.mean(velocities)
    
    return {
        'total_duration': total_duration,
        'total_frames': len(frames),
        'fps': fps,
        'com_height_change': com_height_change,
        'max_velocity': max_velocity,
        'avg_velocity': avg_velocity,
        'start_com_z': com_z[0],
        'end_com_z': com_z[-1]
    }

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_load_3d_skeleton.py <skeleton_3d.json>")
        print("\nThis script demonstrates how to load and visualize 3D skeleton data")
        print("exported by the ClimbingEst system.")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    print(f"Loading skeleton data from: {json_path}")
    data = load_skeleton_json(json_path)
    
    print(f"\nMetadata:")
    print(f"  Format: {data['metadata']['format']}")
    print(f"  Frames: {data['metadata']['num_frames']}")
    print(f"  Joints: {data['metadata']['num_joints']}")
    print(f"  FPS: {data['metadata']['fps']}")
    print(f"  Resolution: {data['metadata']['original_resolution']}")
    
    # Analyze metrics
    metrics = analyze_climbing_metrics(data)
    print(f"\nClimbing Metrics:")
    print(f"  Duration: {metrics['total_duration']:.2f} seconds")
    print(f"  CoM Height Change: {metrics['com_height_change']:.4f} (normalized units)")
    print(f"  Max Velocity: {metrics['max_velocity']:.4f} units/s")
    print(f"  Avg Velocity: {metrics['avg_velocity']:.4f} units/s")
    
    # Extract data for visualization
    frames = data['frames']
    joint_names = data['joint_names']
    connections = data['skeleton_connections']
    
    # Get first frame keypoints
    first_frame = frames[0]
    kpts_3d = np.array([[first_frame['joints'][name]['x'],
                         first_frame['joints'][name]['y'],
                         first_frame['joints'][name]['z']] for name in joint_names])
    
    # Get CoM trajectory
    com_trajectory = np.array([[f['center_of_mass']['x'],
                                f['center_of_mass']['y'],
                                f['center_of_mass']['z']] for f in frames])
    
    # Plot skeleton at first frame
    print("\nGenerating 3D visualizations...")
    fig1, ax1 = plot_skeleton_3d(kpts_3d, connections, joint_names, 
                                  title=f"3D Skeleton (Frame 0)")
    
    # Plot CoM trajectory
    fig2, ax2 = plot_com_trajectory_3d(com_trajectory, 
                                        title="3D Center of Mass Trajectory")
    
    plt.show()
    
    print("\nDone! You can also export this data to Blender using export_to_blender_script()")
