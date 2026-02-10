import argparse
import sys
import os
import time
import json
import logging
import requests
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
from ultralytics import YOLO
import hold_segmentation_utils as hsu

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
# Mass percentages (Dempster model)
# Segment names based on COCO 17 keypoints
# COCO Indices:
# 0: Nose, 1: LEye, 2: REye, 3: LEar, 4: REar
# 5: LShoulder, 6: RShoulder, 7: LElbow, 8: RElbow
# 9: LWrist, 10: RWrist, 11: LHip, 12: RHip
# 13: LKnee, 14: RKnee, 15: LAnkle, 16: RAnkle

DEMPSTER_SEGMENTS = {
    'head': {'indices': [0, 1, 2, 3, 4], 'mass': 0.081, 'name': 'Head'}, # Approx head/neck
    'torso': {'indices': [5, 6, 11, 12], 'mass': 0.497 + 0.02, 'name': 'Torso'}, # Upper + Lower trunk + Neck correction
    'upper_arm_l': {'indices': [5, 7], 'mass': 0.028, 'name': 'L_UpperArm'},
    'upper_arm_r': {'indices': [6, 8], 'mass': 0.028, 'name': 'R_UpperArm'},
    'forearm_l': {'indices': [7, 9], 'mass': 0.016, 'name': 'L_Forearm'}, # Plus hand approx
    'forearm_r': {'indices': [8, 10], 'mass': 0.016, 'name': 'R_Forearm'}, # Plus hand approx
    'hand_l': {'indices': [9], 'mass': 0.006, 'name': 'L_Hand'},
    'hand_r': {'indices': [10], 'mass': 0.006, 'name': 'R_Hand'},
    'thigh_l': {'indices': [11, 13], 'mass': 0.1, 'name': 'L_Thigh'},
    'thigh_r': {'indices': [12, 14], 'mass': 0.1, 'name': 'R_Thigh'},
    'shank_l': {'indices': [13, 15], 'mass': 0.0465, 'name': 'L_Shank'},
    'shank_r': {'indices': [14, 16], 'mass': 0.0465, 'name': 'R_Shank'},
    'foot_l': {'indices': [15], 'mass': 0.0145, 'name': 'L_Foot'}, # using ankle as proxy for foot start
    'foot_r': {'indices': [16], 'mass': 0.0145, 'name': 'R_Foot'},
}
# Note: Masses are approximate and summed to ~1.0. A simplified Dempster model.

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin"
WEIGHTS_FILE = "pretrained_h36m_detectron_coco.bin"

# --- VideoPose3D Model Definition ---
class TemporalModelBase(nn.Module):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels):
        super().__init__()
        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths
        self.causal = causal
        self.dropout = dropout
        self.channels = channels

        layers_conv = []
        layers_bn = []
        
        # expand_conv has kernel=3 in checkpoint, with padding=1 for same output length
        self.expand_conv = nn.Conv1d(num_joints_in*in_features, channels, 3, padding=1)
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        
        next_dilation = 1
        for i in range(len(filter_widths)):
            pad = (filter_widths[i] - 1) * next_dilation // 2
            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i] if not causal else 2*pad + 1, 
                                         dilation=next_dilation, bias=False, 
                                         padding=pad if not causal else 0)) 
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]
            
        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)
        
        self.shrink = nn.Conv1d(channels, num_joints_out*3, 1)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum
            
    def receptive_field(self):
        frames = 0
        for f in self.filter_widths:
            frames += f
        return frames

    def forward(self, x):
        assert len(x.shape) == 4
        # x: [B, T, J, C]
        B, T, J, C = x.shape
        x = x.view(B, T, -1).permute(0, 2, 1) # [B, J*C, T]
        
        x = self.expand_conv(x)
        x = self.expand_bn(x)
        x = F.relu(x)
        
        for i in range(len(self.layers_conv)):
            res = x
            x = self.layers_conv[i](x)
            x = self.layers_bn[i](x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + res # Residual connection
            
        x = self.shrink(x)
        x = x.permute(0, 2, 1).view(B, T, self.num_joints_out, 3)
        return x

class TemporalModel(TemporalModelBase):
    def __init__(self, num_joints_in, in_features, num_joints_out, filter_widths=[3,3,3,3,3], causal=False, dropout=0.25, channels=1024):
        super().__init__(num_joints_in, in_features, num_joints_out, filter_widths, causal, dropout, channels)
        self.dropout = nn.Dropout(dropout)

# --- Utilities ---

def download_weights_if_missing():
    if not os.path.exists(WEIGHTS_FILE):
        logger.info(f"Downloading VideoPose3D weights from {WEIGHTS_URL}...")
        try:
            r = requests.get(WEIGHTS_URL, stream=True)
            total_size = int(r.headers.get('content-length', 0))
            with open(WEIGHTS_FILE, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True, desc=WEIGHTS_FILE) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
            logger.info("Download complete.")
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            sys.exit(1)
    else:
        logger.info("VideoPose3D weights found.")

def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2
    
    # Normalize so that [0, w] is mapped to [-1, 1], same for h
    return X / w * 2 - [1, h/w]

def get_bboxes_yolo(video_path, model_name='yolov8n.pt'):
    logger.info("Running YOLOv8 for person detection...")
    model = YOLO(model_name)
    results = model(video_path, stream=True, classes=[0], verbose=False) # class 0 is person
    
    bboxes = []
    # Collect all boxes first. In complex scenes, we might want to track the *same* person.
    # For now, we assume the person with high confidence or largest area is the climber.
    # To be robust, we'll pick the largest person box in the frame.
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) == 0:
            bboxes.append(None)
        else:
            # Find largest box
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            bboxes.append(boxes[idx])
            
    return bboxes

def pad_bbox(bbox, img_shape, padding_ratio=0.1):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_shape[1], x2 + pad_w)
    y2 = min(img_shape[0], y2 + pad_h)
    
    return [int(x1), int(y1), int(x2), int(y2)]

def run_vitpose(video_path, bboxes, model_name="usyd-community/vitpose-plus-base"):
    logger.info(f"Loading ViTPose model: {model_name}")
    try:
        model = VitPoseForPoseEstimation.from_pretrained(model_name)
        processor = VitPoseImageProcessor.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load ViTPose model: {e}")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoints_2d = []
    keypoints_scores = []
    
    logger.info("Extracting 2D pose...")
    
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
            
        bbox = bboxes[i]
        if bbox is None:
            # If no person detected, use previous or center?
            # Better to just output zeros or interpolate later.
            keypoints_2d.append(np.zeros((17, 2)))
            keypoints_scores.append(np.zeros((17,)))
            continue
            
        # Crop to person
        crop_box = pad_bbox(bbox, frame.shape)
        crop_img = frame[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
        
        # Convert to RGB PIL Image (required by VitPose)
        img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        
        # VitPose requires boxes in [x, y, w, h] format relative to input image
        # Since we cropped, box is the full crop
        crop_h, crop_w = crop_img.shape[:2]
        boxes = [[[0, 0, crop_w, crop_h]]]  # batch of 1 image, 1 person
        
        # Inference
        inputs = processor(pil_img, boxes=boxes, return_tensors="pt").to(device)
        # dataset_index=0 for COCO format (ViTPose++ uses MoE)
        inputs['dataset_index'] = torch.tensor([0]).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Getting results
        # boxes need to be passed for post_process as well
        results = processor.post_process_pose_estimation(outputs, boxes=boxes)[0]
        
        # We assume 1 person since we cropped
        kpts = results[0]['keypoints'].cpu().numpy() # [17, 2]
        scores = results[0]['scores'].cpu().numpy() # [17]
        
        # Translate back to full image coords
        kpts[:, 0] += crop_box[0]
        kpts[:, 1] += crop_box[1]
        
        keypoints_2d.append(kpts)
        keypoints_scores.append(scores)
        
    cap.release()
    return np.array(keypoints_2d), np.array(keypoints_scores)

def lift_to_3d(kpts_2d, image_width, image_height):
    logger.info("Lifting to 3D...")
    
    # Check weights
    download_weights_if_missing()
    
    # Normalize 
    # [N, 17, 2]
    # VideoPose3D expects input normalized to [-1, 1]
    # AND it typically expects normalized by width? 
    # The standard training data (H3.6M) was normalized such that the image is [-1, 1] in width?
    # Let's stick to the official repository normalization logic:
    # 2 * (x - w/2) / w  = 2x/w - 1
    
    # Wait, the official code uses: (x - w/2) / w * 2 ... yes.
    kpts_norm = np.copy(kpts_2d)
    kpts_norm[..., 0] = kpts_norm[..., 0] / image_width * 2 - 1
    kpts_norm[..., 1] = kpts_norm[..., 1] / image_width * 2 - image_height / image_width # Preserves aspect ratio 
    
    # Model Setup
    # 17 joints, 2 coords -> 17 joints, 3 coords
    # Checkpoint has 8 layers: 3, 1, 3, 1, 3, 1, 3, 1
    filter_widths = [3, 1, 3, 1, 3, 1, 3, 1]
    model = TemporalModel(17, 2, 17, filter_widths=filter_widths, causal=False, dropout=0.25, channels=1024)
    
    # Load weights (strict=False to handle minor mismatches)
    checkpoint = torch.load(WEIGHTS_FILE, map_location='cpu')
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Prepare input [B, T, J, C]
    # We treat the whole video as one batch of 1 temporal sequence? 
    # Or chunks? The model uses dilated convolutions, so it can handle arbitrary length (if memory allows)
    # But usually it's trained on specific receptive fields (e.g. 243).
    # We can pass the full sequence if it fits in VRAM, or sliding window.
    # For simplicity and stability, let's pass the whole sequence if < few thousand frames.
    
    inputs_2d = torch.from_numpy(kpts_norm).float().unsqueeze(0) # [1, N, 17, 2]
    inputs_2d = inputs_2d.to(device)
    
    with torch.no_grad():
        # inference
        predicted_3d = model(inputs_2d) # [1, N, 17, 3]
        
    kpts_3d = predicted_3d.squeeze(0).cpu().numpy()
    return kpts_3d

def calculate_com(kpts_3d):
    # kpts_3d: [N, 17, 3]
    # returns: [N, 3]
    
    num_frames = kpts_3d.shape[0]
    com_traj = np.zeros((num_frames, 3))
    
    for i in range(num_frames):
        # Calculate segments
        frame_com = np.zeros(3)
        current_kpts = kpts_3d[i]
        
        for name, data in DEMPSTER_SEGMENTS.items():
            indices = data['indices']
            mass = data['mass']
            
            # Segment center is mean of its keypoints
            # E.g. Thigh center = (Hip + Knee) / 2
            # Head = mean of all head points
            seg_points = current_kpts[indices]
            seg_center = np.mean(seg_points, axis=0)
            
            frame_com += seg_center * mass
            
        com_traj[i] = frame_com
        
    return com_traj

# --- 3D Skeleton Data Export ---
# COCO 17 Keypoint Names for export
JOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Skeleton connections for 3D rendering (COCO format)
SKELETON_CONNECTIONS = [
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Upper body
    (5, 6),   # Shoulders
    (5, 7), (7, 9),    # Left arm
    (6, 8), (8, 10),   # Right arm
    (5, 11), (6, 12),  # Torso sides
    # Lower body
    (11, 12),  # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)   # Right leg
]

def export_skeleton_3d(kpts_3d, kpts_2d, com_trajectory_3d, confidence_scores, output_base_path, fps=30.0, image_width=1920, image_height=1080):
    """
    Export raw 3D skeletal data in multiple formats for 3D rendering and analysis.
    
    Args:
        kpts_3d: [N, 17, 3] array of 3D keypoints from VideoPose3D
        kpts_2d: [N, 17, 2] array of 2D keypoints
        com_trajectory_3d: [N, 3] array of 3D Center of Mass positions
        confidence_scores: [N, 17] array of confidence scores
        output_base_path: Base path for output files (without extension)
        fps: Video frame rate
        image_width: Original video width
        image_height: Original video height
    
    Returns:
        dict with paths to exported files
    """
    num_frames = kpts_3d.shape[0]
    num_joints = kpts_3d.shape[1]
    
    exported_files = {}
    
    # --- 1. Export as JSON (ideal for web 3D viewers, Three.js, etc.) ---
    json_path = output_base_path + '_skeleton_3d.json'
    
    skeleton_data = {
        'metadata': {
            'format': 'ClimbingEst 3D Skeleton Export',
            'version': '1.0',
            'source': 'VideoPose3D',
            'num_frames': num_frames,
            'num_joints': num_joints,
            'fps': fps,
            'original_resolution': {
                'width': image_width,
                'height': image_height
            },
            'coordinate_system': {
                'description': 'VideoPose3D normalized coordinates',
                'x': 'horizontal (right positive)',
                'y': 'depth (forward positive)', 
                'z': 'vertical (up positive)',
                'units': 'normalized to hip distance'
            }
        },
        'joint_names': JOINT_NAMES,
        'skeleton_connections': SKELETON_CONNECTIONS,
        'frames': []
    }
    
    for frame_idx in range(num_frames):
        frame_data = {
            'frame_id': frame_idx,
            'timestamp': frame_idx / fps,
            'joints': {},
            'joints_2d': {},
            'center_of_mass': {
                'x': float(com_trajectory_3d[frame_idx, 0]),
                'y': float(com_trajectory_3d[frame_idx, 1]),
                'z': float(com_trajectory_3d[frame_idx, 2])
            }
        }
        
        for joint_idx, joint_name in enumerate(JOINT_NAMES):
            frame_data['joints'][joint_name] = {
                'x': float(kpts_3d[frame_idx, joint_idx, 0]),
                'y': float(kpts_3d[frame_idx, joint_idx, 1]),
                'z': float(kpts_3d[frame_idx, joint_idx, 2])
            }
            frame_data['joints_2d'][joint_name] = {
                'x': float(kpts_2d[frame_idx, joint_idx, 0]),
                'y': float(kpts_2d[frame_idx, joint_idx, 1]),
                'conf': float(confidence_scores[frame_idx, joint_idx]) if confidence_scores is not None else 0.0
            }
        
        skeleton_data['frames'].append(frame_data)
    
    with open(json_path, 'w') as f:
        json.dump(skeleton_data, f, indent=2)
    
    exported_files['json'] = json_path
    logger.info(f"Exported 3D skeleton JSON: {json_path}")
    
    # --- 2. Export as detailed CSV (all joints per frame) ---
    csv_detailed_path = output_base_path + '_skeleton_3d_detailed.csv'
    
    # Create header
    header_parts = ['frame', 'timestamp']
    for joint_name in JOINT_NAMES:
        header_parts.extend([f'{joint_name}_x', f'{joint_name}_y', f'{joint_name}_z'])
    header_parts.extend(['com_x', 'com_y', 'com_z'])
    
    with open(csv_detailed_path, 'w') as f:
        f.write(','.join(header_parts) + '\n')
        
        for frame_idx in range(num_frames):
            row = [str(frame_idx), f'{frame_idx / fps:.4f}']
            
            for joint_idx in range(num_joints):
                row.extend([
                    f'{kpts_3d[frame_idx, joint_idx, 0]:.6f}',
                    f'{kpts_3d[frame_idx, joint_idx, 1]:.6f}',
                    f'{kpts_3d[frame_idx, joint_idx, 2]:.6f}'
                ])
            
            row.extend([
                f'{com_trajectory_3d[frame_idx, 0]:.6f}',
                f'{com_trajectory_3d[frame_idx, 1]:.6f}',
                f'{com_trajectory_3d[frame_idx, 2]:.6f}'
            ])
            
            f.write(','.join(row) + '\n')
    
    exported_files['csv_detailed'] = csv_detailed_path
    logger.info(f"Exported 3D skeleton detailed CSV: {csv_detailed_path}")
    
    # --- 3. Export as compact NumPy arrays (for analysis) ---
    npz_path = output_base_path + '_skeleton_3d.npz'
    
    np.savez_compressed(
        npz_path,
        keypoints_3d=kpts_3d,
        keypoints_2d=kpts_2d,
        confidence_scores=confidence_scores,
        com_trajectory_3d=com_trajectory_3d,
        joint_names=JOINT_NAMES,
        skeleton_connections=np.array(SKELETON_CONNECTIONS),
        fps=fps,
        image_size=np.array([image_width, image_height])
    )
    
    exported_files['npz'] = npz_path
    logger.info(f"Exported 3D skeleton NPZ: {npz_path}")

    # --- 4. Export Confidence Levels CSV (Body Parts Only) ---
    if confidence_scores is not None:
        csv_conf_path = output_base_path + '_confidence_levels.csv'
        
        # Define relevant body parts (exclude face: nose, eyes, ears)
        # Indices 5-16 in COCO format correspond to shoulders, elbows, wrists, hips, knees, ankles
        BODY_INDICES = list(range(5, 17))
        BODY_NAMES = [JOINT_NAMES[i] for i in BODY_INDICES]
        
        # Header: frame, timestamp, [body_parts], average_body_confidence
        conf_header = ['frame', 'timestamp'] + BODY_NAMES + ['average_body_confidence']
        
        with open(csv_conf_path, 'w') as f:
            f.write(','.join(conf_header) + '\n')
            
            for frame_idx in range(num_frames):
                row = [str(frame_idx), f'{frame_idx / fps:.4f}']
                
                # Get scores for relevant body parts
                all_scores = confidence_scores[frame_idx]
                body_scores = all_scores[BODY_INDICES]
                
                # Add individual scores
                row.extend([f'{s:.4f}' for s in body_scores])
                
                # Add average for body parts only
                row.append(f'{np.mean(body_scores):.4f}')
                
                f.write(','.join(row) + '\n')
                
        exported_files['csv_confidence'] = csv_conf_path
        logger.info(f"Exported Confidence CSV: {csv_conf_path}")
    
    return exported_files

# Limb pairs for validation (keypoint indices)
LIMB_PAIRS = [
    (5, 7),   # L Shoulder -> L Elbow
    (7, 9),   # L Elbow -> L Wrist
    (6, 8),   # R Shoulder -> R Elbow
    (8, 10),  # R Elbow -> R Wrist
    (11, 13), # L Hip -> L Knee
    (13, 15), # L Knee -> L Ankle
    (12, 14), # R Hip -> R Knee
    (14, 16), # R Knee -> R Ankle
    (5, 11),  # L Shoulder -> L Hip (torso)
    (6, 12),  # R Shoulder -> R Hip (torso)
]

def stabilize_skeleton_keypoints(keypoints_2d, threshold_multiplier=2.5):
    """
    Stabilize keypoints by detecting and fixing physically impossible poses.
    
    Detects when limbs suddenly stretch to impossible lengths (like the foot
    going far away) and uses the previous good frame's keypoints instead.
    
    Args:
        keypoints_2d: [N, 17, 2] array of 2D keypoints
        threshold_multiplier: How many times the median length is considered "impossible"
    
    Returns:
        Stabilized keypoints array
    """
    if len(keypoints_2d) < 3:
        return keypoints_2d
    
    stabilized = keypoints_2d.copy()
    
    # Calculate all limb lengths for all frames
    limb_lengths = np.zeros((len(keypoints_2d), len(LIMB_PAIRS)))
    
    for frame_idx in range(len(keypoints_2d)):
        kpts = keypoints_2d[frame_idx]
        for limb_idx, (p1, p2) in enumerate(LIMB_PAIRS):
            pt1, pt2 = kpts[p1], kpts[p2]
            # Only calculate if both points are valid
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                length = np.linalg.norm(pt2 - pt1)
                limb_lengths[frame_idx, limb_idx] = length
            else:
                limb_lengths[frame_idx, limb_idx] = 0
    
    # Calculate median length for each limb (ignoring zeros)
    median_lengths = np.zeros(len(LIMB_PAIRS))
    for limb_idx in range(len(LIMB_PAIRS)):
        valid_lengths = limb_lengths[:, limb_idx]
        valid_lengths = valid_lengths[valid_lengths > 0]
        if len(valid_lengths) > 0:
            median_lengths[limb_idx] = np.median(valid_lengths)
    
    # Detect and fix impossible frames
    last_good_kpts = None
    frames_fixed = 0
    
    for frame_idx in range(len(keypoints_2d)):
        kpts = keypoints_2d[frame_idx]
        is_impossible = False
        
        # Check each limb
        for limb_idx, (p1, p2) in enumerate(LIMB_PAIRS):
            if median_lengths[limb_idx] == 0:
                continue
                
            current_length = limb_lengths[frame_idx, limb_idx]
            if current_length == 0:
                continue
            
            # If limb is more than threshold_multiplier times the median, it's impossible
            if current_length > median_lengths[limb_idx] * threshold_multiplier:
                is_impossible = True
                break
        
        if is_impossible and last_good_kpts is not None:
            # Use the last good frame's keypoints
            stabilized[frame_idx] = last_good_kpts
            frames_fixed += 1
        else:
            # This frame is good, save it
            if np.sum(kpts) > 0:  # Only save if not all zeros
                last_good_kpts = kpts.copy()
    
    if frames_fixed > 0:
        logger.info(f"Skeleton stabilization: Fixed {frames_fixed} impossible frames")
    
    return stabilized

def extract_hold_positions(video_path, sample_frames=10):
    """
    Detect climbing holds using ML-based segmentation (Detectron2).
    Also detects LED-lit holds to mark active route.
    
    Args:
        video_path: Path to video file
        sample_frames: Frames to sample for LED detection
    
    Returns:
        hold_positions: List of (x, y, radius, is_active) tuples
        background_holds_img: Image with detected holds drawn
    """
    # Import ML module (lazy import to avoid loading Detectron2 if not needed)
    try:
        import ml_hold_segmentation as ml_seg
    except ImportError as e:
        logger.error(f"ML segmentation module not available: {e}")
        return [], None
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- ML Hold Segmentation ---
    logger.info("Starting ML-based hold segmentation...")
    start_time = time.time()
    
    # Check if model is available
    if not ml_seg.is_model_available():
        logger.warning("ML model weights not found. Please download from Kaggle.")
        logger.warning(ml_seg.get_model_download_instructions())
        # Return empty - user needs to download model
        cap.release()
        return [], np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get first frame for detection
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, detection_frame = cap.read()
    
    if not ret:
        logger.error("Could not read video for hold detection")
        cap.release()
        return [], np.zeros((height, width, 3), dtype=np.uint8)
    
    # Run ML detection
    try:
        ml_holds = ml_seg.detect_holds_ml(detection_frame)
    except Exception as e:
        logger.error(f"ML detection failed: {e}")
        cap.release()
        return [], np.zeros((height, width, 3), dtype=np.uint8)
    
    # Convert ML results to our format: (x, y, radius, is_active)
    hold_positions = []
    for hold in ml_holds:
        hold_positions.append((hold['x'], hold['y'], hold['radius'], False))
    
    detection_time = time.time() - start_time
    logger.info(f"ML segmentation complete in {detection_time:.2f}s. Found {len(hold_positions)} holds.")
    
    cap.release()
    
    # Note: LED detection removed - used holds detection (based on pose) handles this now
    # The is_active flag will be set by detect_used_holds() in process_video()
    
    # Create background holds image
    background_holds_img = np.zeros((height, width, 3), dtype=np.uint8)
    draw_holds(background_holds_img, hold_positions)
    
    return hold_positions, background_holds_img

def draw_holds(img, hold_positions, active_color=(0, 255, 255), inactive_color=(80, 80, 80), uniform_radius=25):
    """
    Draw hold positions on an image.
    
    Args:
        img: Image to draw on
        hold_positions: List of (x, y, radius, is_active) tuples
        active_color: Color for used/active holds (cyan outline)
        inactive_color: Color for inactive holds (grey outline)
        uniform_radius: If set, use this radius for all holds (consistent sizing)
    """
    for x, y, radius, is_active in hold_positions:
        # Use uniform radius if specified, otherwise use detected radius
        draw_radius = uniform_radius if uniform_radius else radius
        
        if is_active:
            # Used holds: Cyan outline (thicker) with small center dot
            cv2.circle(img, (x, y), draw_radius, active_color, 3)  # 3px cyan outline
            cv2.circle(img, (x, y), 3, active_color, -1)  # Small center dot
        else:
            # Inactive holds: Grey outline with small center dot
            cv2.circle(img, (x, y), draw_radius, inactive_color, 2)  # 2px grey outline
            cv2.circle(img, (x, y), 2, inactive_color, -1)  # Tiny center dot

def detect_used_holds(hold_positions, keypoints_2d, fps, min_contact_frames=5, proximity_threshold=None):
    """
    Detect which holds were actually used by the climber based on hand/foot proximity.
    
    Args:
        hold_positions: List of (x, y, radius, is_active) tuples
        keypoints_2d: Array of shape (num_frames, 17, 2) with COCO keypoints
        fps: Video frames per second
        min_contact_frames: Minimum frames hand/foot must be near hold to count as "used"
        proximity_threshold: Distance threshold (if None, uses hold radius + buffer)
    
    Returns:
        Updated hold_positions with is_active=True for used holds
    """
    if len(hold_positions) == 0 or keypoints_2d is None:
        return hold_positions
    
    # COCO keypoint indices for hands and feet
    # 9: left_wrist, 10: right_wrist, 15: left_ankle, 16: right_ankle
    HAND_FOOT_INDICES = [9, 10, 15, 16]
    
    num_frames = len(keypoints_2d)
    num_holds = len(hold_positions)
    
    # Track contact frames for each hold
    hold_contact_counts = [0] * num_holds
    
    for frame_idx in range(num_frames):
        kpts = keypoints_2d[frame_idx]
        
        for kpt_idx in HAND_FOOT_INDICES:
            if kpts[kpt_idx][0] <= 0 or kpts[kpt_idx][1] <= 0:
                continue  # Skip invalid keypoints
            
            px, py = kpts[kpt_idx]
            
            # Check proximity to each hold
            for hold_idx, (hx, hy, hr, _) in enumerate(hold_positions):
                # Distance from keypoint to hold center
                dist = np.sqrt((px - hx)**2 + (py - hy)**2)
                
                # Use hold radius + buffer as threshold
                threshold = hr + 20 if proximity_threshold is None else proximity_threshold
                
                if dist < threshold:
                    hold_contact_counts[hold_idx] += 1
    
    # Mark holds as "used" if they had enough contact frames
    used_count = 0
    updated_positions = []
    
    for i, (hx, hy, hr, _) in enumerate(hold_positions):
        # A hold is "used" if touched for at least min_contact_frames
        is_used = hold_contact_counts[i] >= min_contact_frames
        if is_used:
            used_count += 1
        updated_positions.append((hx, hy, hr, is_used))
    
    logger.info(f"Detected {used_count} holds used by climber (out of {num_holds})")
    return updated_positions

def draw_skeleton(img, kpts, skeleton_color=(0, 255, 0), keypoint_color=(0, 0, 255), 
                  keypoint_size=6, line_thickness=2, show_keypoints=True, hide_face=False):
    """Draw skeleton with configurable appearance.
    
    Args:
        hide_face: If True, don't draw face keypoints (0-4) for extra anonymization
    """
    # kpts: [17, 2]
    # COCO connections - face connections are first 4
    face_connections = [(0,1), (0,2), (1,3), (2,4)]
    body_connections = [
        (5,7), (7,9), # Left Arm
        (6,8), (8,10), # Right Arm
        (5,6), (5,11), (6,12), # Torso
        (11,12), # Hip connection
        (11,13), (13,15), # Left Leg
        (12,14), (14,16)  # Right Leg
    ]
    
    # Choose which connections to draw
    if hide_face:
        skeleton = body_connections
    else:
        skeleton = face_connections + body_connections
    
    # Face keypoint indices (to skip when hide_face=True)
    face_indices = {0, 1, 2, 3, 4}
    
    # Draw skeleton lines first (so keypoints are on top)
    for p1, p2 in skeleton:
        pt1 = (int(kpts[p1][0]), int(kpts[p1][1]))
        pt2 = (int(kpts[p2][0]), int(kpts[p2][1]))
        # Only draw if points are valid (non-zero)
        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
            cv2.line(img, pt1, pt2, skeleton_color, line_thickness)
    
    # Draw keypoints on top
    if show_keypoints:
        # Color mapping for different body parts
        keypoint_colors = {
            0: (255, 200, 200),  # Nose - light pink
            1: (255, 150, 150), 2: (255, 150, 150),  # Eyes
            3: (255, 100, 100), 4: (255, 100, 100),  # Ears
            5: (100, 255, 100), 6: (100, 255, 100),  # Shoulders - green
            7: (100, 200, 100), 8: (100, 200, 100),  # Elbows
            9: (100, 150, 100), 10: (100, 150, 100), # Wrists
            11: (100, 100, 255), 12: (100, 100, 255), # Hips - blue
            13: (150, 100, 255), 14: (150, 100, 255), # Knees
            15: (200, 100, 255), 16: (200, 100, 255)  # Ankles
        }
        for idx, pt in enumerate(kpts):
            # Skip face keypoints if hide_face is enabled
            if hide_face and idx in face_indices:
                continue
            if pt[0] > 0 and pt[1] > 0:
                color = keypoint_colors.get(idx, keypoint_color)
                cv2.circle(img, (int(pt[0]), int(pt[1])), keypoint_size, color, -1)
                cv2.circle(img, (int(pt[0]), int(pt[1])), keypoint_size, (0, 0, 0), 1)  # Black border

def draw_com_trail(img, com_history, trail_length=30, com_size=12, com_color=(0, 255, 255),
                   persistent_trail=False, show_speed_color=False, velocity_history=None):
    """
    Draw CoM point with trailing history path.
    
    Args:
        persistent_trail: If True, never fade the trail (show all history)
        show_speed_color: If True, color based on vertical speed (green=up, red=down)
        velocity_history: List of vertical velocities for speed coloring
    """
    if len(com_history) == 0:
        return
    
    # Determine which points to show
    if persistent_trail:
        trail_points = com_history  # Show all
    else:
        trail_points = com_history[-trail_length:]
    
    # Draw trail points
    for idx, (cx, cy) in enumerate(trail_points[:-1]):
        # Calculate alpha based on position in trail
        if persistent_trail:
            alpha = 0.6  # Constant alpha for persistent trail
            trail_size = max(2, int(com_size * 0.25))
        else:
            alpha = (idx + 1) / len(trail_points)
            trail_size = max(2, int(com_size * 0.3 * alpha))
        
        # Determine color
        if show_speed_color and velocity_history and idx < len(velocity_history):
            vel = velocity_history[idx]
            # Positive velocity = moving up (green), negative = down (red)
            # Note: In image coords, Y increases downward, so negative vel = moving up
            if vel < -2:  # Moving up fast
                point_color = (0, 255, 0)  # Green
            elif vel > 2:  # Moving down fast
                point_color = (0, 0, 255)  # Red
            else:  # Slow/stationary
                point_color = (0, 255, 255)  # Yellow
        else:
            point_color = (
                int(com_color[0] * alpha * 0.7),
                int(com_color[1] * alpha * 0.7),
                int(com_color[2] * alpha * 0.7)
            )
        
        cv2.circle(img, (int(cx), int(cy)), trail_size, point_color, -1)
    
    # Connect trail with lines
    if len(trail_points) > 1:
        for idx in range(len(trail_points) - 1):
            pt1 = (int(trail_points[idx][0]), int(trail_points[idx][1]))
            pt2 = (int(trail_points[idx + 1][0]), int(trail_points[idx + 1][1]))
            
            if show_speed_color and velocity_history and idx < len(velocity_history):
                vel = velocity_history[idx]
                if vel < -2:
                    line_color = (0, 200, 0)
                elif vel > 2:
                    line_color = (0, 0, 200)
                else:
                    line_color = (0, 200, 200)
            else:
                if persistent_trail:
                    alpha = 0.5
                else:
                    alpha = (idx + 1) / len(trail_points)
                line_color = (
                    int(com_color[0] * alpha),
                    int(com_color[1] * alpha),
                    int(com_color[2] * alpha)
                )
            cv2.line(img, pt1, pt2, line_color, 2)
    
    # Draw current CoM point (largest, brightest)
    current = trail_points[-1]
    current_color = com_color
    if show_speed_color and velocity_history and len(velocity_history) > 0:
        vel = velocity_history[-1]
        if vel < -2:
            current_color = (0, 255, 0)
        elif vel > 2:
            current_color = (0, 0, 255)
    
    cv2.circle(img, (int(current[0]), int(current[1])), com_size, current_color, -1)
    cv2.circle(img, (int(current[0]), int(current[1])), com_size, (0, 0, 0), 2)
    cv2.putText(img, "CoM", (int(current[0]) + com_size + 5, int(current[1])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, current_color, 2)

def process_video(video_path, output_path='output_video.mp4', progress=None,
                  trail_length=30, keypoint_size=6, com_size=12, show_keypoints=True,
                  smooth_com=True, persistent_trail=False, show_speed_color=False,
                  stick_figure_mode=False, stabilize_skeleton=True, show_holds=False,
                  hide_skeleton=False, bg_color=(0, 0, 0)):
    """
    Process video with configurable visualization options.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        progress: Progress callback function(value, desc)
        trail_length: Number of frames to show in CoM trail
        keypoint_size: Size of keypoint markers
        com_size: Size of CoM marker
        show_keypoints: Whether to show individual keypoints
        smooth_com: Apply temporal smoothing to CoM trajectory
        persistent_trail: Keep entire trail visible (never fade)
        show_speed_color: Color CoM by vertical speed (green=up, red=down)
        stick_figure_mode: If True, render skeleton on plain background (anonymized)
        stabilize_skeleton: If True, filter out impossible poses (stretched limbs)
        show_holds: DEPRECATED - Hold detection has been removed
        hide_skeleton: If True, do not draw the skeleton (useful for CoM only view)
        bg_color: Background color for stick figure mode (default black)
    """
    from scipy.signal import savgol_filter
    
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. This will be very slow.")
    
    if not video_path:
        logger.info("No video provided.")
        return None, None

    # Note: Hold detection has been removed as it didn't work well enough

    # 1. Detect
    if progress: progress(0.1, desc="Detecting Climber (YOLOv8)")
    bbox_list = get_bboxes_yolo(video_path)
    
    # 2. Keypoints
    if progress: progress(0.3, desc="Estimating Pose (ViTPose)")
    final_kpts_2d, confidence_scores = run_vitpose(video_path, bbox_list)
    if final_kpts_2d is None:
        logger.error("Failed to run pose estimation.")
        return None, None
    
    # Calculate model accuracy (average confidence score of BODY parts only)
    # Indices 5-16: Shoulders, Elbows, Wrists, Hips, Knees, Ankles
    if len(confidence_scores) > 0:
        # Slice only body parts [all_frames, 5:17]
        body_scores = confidence_scores[:, 5:17]
        avg_confidence = np.mean(body_scores)
    else:
        avg_confidence = 0.0
        
    model_accuracy_percent = avg_confidence * 100
    logger.info(f"Model Accuracy (Body Confidence): {model_accuracy_percent:.2f}%")
    
    # Stabilization: Fill missing poses with last known good pose
    last_good_pose = None
    for i in range(len(final_kpts_2d)):
        if np.sum(final_kpts_2d[i]) == 0:  # All zeros = lost pose
            if last_good_pose is not None:
                final_kpts_2d[i] = last_good_pose
                logger.debug(f"Frame {i}: Using fallback pose")
        else:
            last_good_pose = final_kpts_2d[i].copy()
    
    # Apply skeleton stabilization to filter impossible poses (stretched limbs)
    if stabilize_skeleton:
        if progress: progress(0.45, desc="Stabilizing Skeleton")
        final_kpts_2d = stabilize_skeleton_keypoints(final_kpts_2d)
        
    # Get video dims
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Note: Hold detection step removed
    
    # 3. Lift
    if progress: progress(0.6, desc="Lifting to 3D (VideoPose3D)")
    final_kpts_3d = lift_to_3d(final_kpts_2d, width, height)
    
    # 4. CoM
    com_trajectory_3d = calculate_com(final_kpts_3d)
    
    # 4b. Pre-calculate 2D CoM for all frames
    logger.info("Calculating 2D CoM trajectory...")
    com_2d_trajectory = []
    for i in range(len(final_kpts_2d)):
        kpts_2d = final_kpts_2d[i]
        com_x, com_y = 0.0, 0.0
        for name, data in DEMPSTER_SEGMENTS.items():
            indices = data['indices']
            mass = data['mass']
            seg_points = kpts_2d[indices]
            valid_points = seg_points[np.any(seg_points > 0, axis=1)]
            if len(valid_points) > 0:
                seg_center = np.mean(valid_points, axis=0)
                com_x += seg_center[0] * mass
                com_y += seg_center[1] * mass
        com_2d_trajectory.append((com_x, com_y))
    
    com_2d_trajectory = np.array(com_2d_trajectory)
    
    # 4c. Apply smoothing if enabled
    if smooth_com and len(com_2d_trajectory) > 15:
        window_length = min(31, len(com_2d_trajectory) // 2 * 2 - 1)  # Must be odd
        if window_length >= 5:
            com_2d_trajectory[:, 0] = savgol_filter(com_2d_trajectory[:, 0], window_length, 3)
            com_2d_trajectory[:, 1] = savgol_filter(com_2d_trajectory[:, 1], window_length, 3)
            logger.info(f"Applied Savitzky-Golay smoothing (window={window_length})")
    
    # 4d. Calculate vertical velocities for speed coloring
    velocity_y = np.zeros(len(com_2d_trajectory))
    if len(com_2d_trajectory) > 1:
        velocity_y[1:] = np.diff(com_2d_trajectory[:, 1])
    
    # 5. Export & Render
    logger.info("rendering output...")
    if progress: progress(0.8, desc="Rendering Output")
    
    # Save CoM trajectory CSV (basic)
    csv_path = output_path.replace('.mp4', '.csv')
    np.savetxt(csv_path, com_trajectory_3d, delimiter=",", header="x,y,z", comments="")
    
    # Export full 3D skeleton data (JSON, detailed CSV, NPZ, Confidence CSV)
    output_base = output_path.replace('.mp4', '')
    skeleton_files = export_skeleton_3d(
        kpts_3d=final_kpts_3d,
        kpts_2d=final_kpts_2d,
        com_trajectory_3d=com_trajectory_3d,
        confidence_scores=confidence_scores,
        output_base_path=output_base,
        fps=fps,
        image_width=width,
        image_height=height
    )
    logger.info(f"Exported 3D skeleton data: {list(skeleton_files.keys())}")
    
    # Render Video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # CoM history for trail
    com_history = []
    velocity_history = []
    
    for i in tqdm(range(len(final_kpts_2d))):
        ret, frame = cap.read()
        if not ret: break
        
        # In stick figure mode, create a blank canvas instead of video
        if stick_figure_mode:
            frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
            # Use thicker lines for better visibility on plain background
            line_thickness = 3
        else:
            line_thickness = 2
        
        # Draw 2D Skeleton with enhanced visuals
        if not hide_skeleton:
            draw_skeleton(frame, final_kpts_2d[i], keypoint_size=keypoint_size, 
                          show_keypoints=show_keypoints, line_thickness=line_thickness)
        
        # Get smoothed CoM position
        com_x, com_y = com_2d_trajectory[i]
        
        # Add to history and draw trail
        com_history.append((com_x, com_y))
        velocity_history.append(velocity_y[i])
        
        draw_com_trail(frame, com_history, trail_length=trail_length, com_size=com_size,
                       persistent_trail=persistent_trail, show_speed_color=show_speed_color,
                       velocity_history=velocity_history)
        
        out.write(frame)
        
    cap.release()
    out.release()
    logger.info(f"Done! Results saved to {output_path} and {csv_path}")
    logger.info(f"3D skeleton data exported: {skeleton_files}")
    metrics = {
        'model_accuracy_percent': float(model_accuracy_percent),
        'average_confidence': float(avg_confidence)
    }
    return output_path, csv_path, skeleton_files, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Path to output video')
    args = parser.parse_args()
    
    video_path = args.video
    if not video_path:
        print("Please provide a video path.")
        return

    process_video(video_path, args.output)

if __name__ == "__main__":
    main()
