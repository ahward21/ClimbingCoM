
import json
import pandas as pd
import numpy as np
import os
import argparse

RAJAGOPAL_MAP = {
    'nose': 'Head',
    'left_shoulder': 'L.Shoulder',
    'right_shoulder': 'R.Shoulder',
    'left_elbow': 'L.Elbow',
    'right_elbow': 'R.Elbow',
    'left_wrist': 'L.Wrist',
    'right_wrist': 'R.Wrist',
    'left_hip': 'L.Hip',
    'right_hip': 'R.Hip',
    'left_knee': 'L.Knee',
    'right_knee': 'R.Knee',
    'left_ankle': 'L.Ankle',
    'right_ankle': 'R.Ankle'
}

def rotate_point(x, y, angle):
    """Rotates a point (x, y) around origin (0, 0) by angle (radians)."""
    s, c = np.sin(angle), np.cos(angle)
    return x * c - y * s, x * s + y * c

def convert_json_to_trc(json_path, trc_output_path=None, target_height=1.70):
    if trc_output_path is None:
        trc_output_path = json_path.replace('.json', '.trc')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data['frames']
    fps = data['metadata'].get('fps', 30.0)
    num_frames = len(frames)
    
    base_joints = [j for j in RAJAGOPAL_MAP.keys() if j in frames[0]['joints']]
    has_com = 'center_of_mass' in frames[0]
    trc_markers = [RAJAGOPAL_MAP[j] for j in base_joints]
    if has_com: trc_markers.append('CoM')
    num_markers = len(trc_markers)

    # 1. Determine Scale and Inversion
    nose_zs = [f['joints']['nose']['z'] for f in frames]
    ankle_zs = [(f['joints']['left_ankle']['z'] + f['joints']['right_ankle']['z'])/2 for f in frames]
    height_units = abs(np.mean(nose_zs) - np.mean(ankle_zs))
    scale = target_height / height_units
    inverted = np.mean(nose_zs) < np.mean(ankle_zs)

    # 2. Alignment (First Frame Hips)
    h0 = frames[0]['joints']
    # Re-calculate mid-hip origin in raw units
    raw_mid_x = (h0['left_hip']['x'] + h0['right_hip']['x']) / 2
    raw_mid_y = (h0['left_hip']['y'] + h0['right_hip']['y']) / 2
    raw_mid_z = (h0['left_hip']['z'] + h0['right_hip']['z']) / 2
    
    # Calculate initial rotation to "square" the person to the camera (Z-axis is Medial-Lateral)
    # Vector from right hip to left hip
    dx = h0['left_hip']['x'] - h0['right_hip']['x']
    dy = h0['left_hip']['y'] - h0['right_hip']['y']
    angle = -np.arctan2(dy, dx) # Rotate so hip line is horizontal (along X in JSON)
    
    print(f"Aligning skeleton. Auto-Rotation: {np.degrees(angle):.1f} degrees.")

    # 3. Write TRC
    with open(trc_output_path, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{os.path.basename(trc_output_path)}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{fps:.2f}\t{fps:.2f}\t{num_frames}\t{num_markers}\tm\t{fps:.2f}\t1\t{num_frames}\n")
        f.write("Frame#\tTime\t" + "\t\t\t".join(trc_markers) + "\t\n")
        
        labels = []
        for i in range(1, num_markers + 1):
            labels.extend([f"X{i}", f"Y{i}", f"Z{i}"])
        f.write("\t\t" + "\t".join(labels) + "\n")
        
        for idx, frame in enumerate(frames):
            row_data = [str(idx + 1), f"{frame['timestamp']:.4f}"]
            
            for j_name in base_joints:
                pos = frame['joints'][j_name]
                
                # Center and Rotate Side/Depth
                cx = (pos['x'] - raw_mid_x)
                cy = (pos['y'] - raw_mid_y)
                rx, ry = rotate_point(cx, cy, angle)
                
                # Map to OpenSim
                os_z = rx * scale # Side
                os_x = ry * scale # Depth
                if inverted:
                    os_y = (raw_mid_z - pos['z']) * scale + 0.95 # Vertical (Hips at 0.95m)
                else:
                    os_y = (pos['z'] - raw_mid_z) * scale + 0.95
                
                row_data.extend([f"{os_x:.6f}", f"{os_y:.6f}", f"{os_z:.6f}"])

            if has_com:
                pos = frame['center_of_mass']
                cx = (pos['x'] - raw_mid_x)
                cy = (pos['y'] - raw_mid_y)
                rx, ry = rotate_point(cx, cy, angle)
                os_z = rx * scale
                os_x = ry * scale
                if inverted:
                    os_y = (raw_mid_z - pos['z']) * scale + 0.95
                else:
                    os_y = (pos['z'] - raw_mid_z) * scale + 0.95
                row_data.extend([f"{os_x:.6f}", f"{os_y:.6f}", f"{os_z:.6f}"])
            
            f.write("\t".join(row_data) + "\n")

    print(f"Exported aligned TRC to {trc_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    convert_json_to_trc(args.input)
