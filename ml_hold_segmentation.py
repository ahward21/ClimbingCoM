"""
ML-based Hold Segmentation using Detectron2.

This module provides hold detection using a pre-trained Mask R-CNN model
from the xiaoxiae/Indoor-Climbing-Hold-and-Route-Segmentation project.

Model weights should be downloaded from:
https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation
"""

import os
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Model configuration
MODEL_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "model_final_holds.pth")
MODEL_CONFIG_URL = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

# Global predictor (loaded once)
_predictor = None
_cfg = None

def is_model_available():
    """Check if model weights are downloaded."""
    return os.path.exists(MODEL_WEIGHTS_PATH)

def get_model_download_instructions():
    """Get instructions for downloading the model."""
    return f"""
    Model weights not found at: {MODEL_WEIGHTS_PATH}
    
    Please download the model weights from Kaggle:
    1. Go to https://www.kaggle.com/datasets/tomasslama/indoor-climbing-gym-hold-segmentation
    2. Download the dataset
    3. Find the model weights file (e.g., model_final.pth)
    4. Rename it to 'model_final_holds.pth' and place it in: {os.path.dirname(__file__)}
    """

def load_predictor():
    """Load the Detectron2 predictor (lazy loading)."""
    global _predictor, _cfg
    
    if _predictor is not None:
        return _predictor
    
    if not is_model_available():
        logger.error(get_model_download_instructions())
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS_PATH}")
    
    try:
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from detectron2 import model_zoo
        import yaml
        
        logger.info("Loading Detectron2 hold segmentation model...")
        
        _cfg = get_cfg()
        
        # Load base config
        _cfg.merge_from_file(model_zoo.get_config_file(MODEL_CONFIG_URL))
        
        # Load custom experiment config if available
        experiment_config_path = os.path.join(os.path.dirname(__file__), "experiment_config.yml")
        if os.path.exists(experiment_config_path):
            logger.info(f"Loading experiment config from {experiment_config_path}")
            with open(experiment_config_path, 'r') as f:
                exp_cfg = yaml.safe_load(f)
            
            # Apply critical settings from experiment config
            # Anchor Generator settings (this was causing the mismatch!)
            if 'MODEL' in exp_cfg:
                model_cfg = exp_cfg['MODEL']
                
                if 'ANCHOR_GENERATOR' in model_cfg:
                    ag = model_cfg['ANCHOR_GENERATOR']
                    if 'SIZES' in ag:
                        _cfg.MODEL.ANCHOR_GENERATOR.SIZES = ag['SIZES']
                    if 'ASPECT_RATIOS' in ag:
                        _cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = ag['ASPECT_RATIOS']
                
                if 'ROI_HEADS' in model_cfg:
                    rh = model_cfg['ROI_HEADS']
                    if 'NUM_CLASSES' in rh:
                        _cfg.MODEL.ROI_HEADS.NUM_CLASSES = rh['NUM_CLASSES']
                    if 'IN_FEATURES' in rh:
                        _cfg.MODEL.ROI_HEADS.IN_FEATURES = rh['IN_FEATURES']
                    if 'SCORE_THRESH_TEST' in rh:
                        _cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = rh['SCORE_THRESH_TEST']
                
                if 'RPN' in model_cfg:
                    rpn = model_cfg['RPN']
                    if 'IN_FEATURES' in rpn:
                        _cfg.MODEL.RPN.IN_FEATURES = rpn['IN_FEATURES']
        else:
            # Fallback to hardcoded values from the known config
            logger.warning("experiment_config.yml not found, using hardcoded settings")
            _cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            _cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64]]
            _cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3"]
            _cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
        
        # Set model weights path
        _cfg.MODEL.WEIGHTS = MODEL_WEIGHTS_PATH
        
        # Detection threshold - lower for more detections during testing
        _cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        
        # Use GPU if available
        _cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        _predictor = DefaultPredictor(_cfg)
        logger.info(f"Model loaded successfully on {_cfg.MODEL.DEVICE}")
        
        return _predictor
        
    except Exception as e:
        logger.error(f"Failed to load Detectron2 model: {e}")
        import traceback
        traceback.print_exc()
        raise

def detect_holds_ml(image):
    """
    Detect climbing holds using ML-based segmentation.
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        List of (x, y, radius, contour) tuples for each detected hold
    """
    predictor = load_predictor()
    
    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    holds = []
    
    # Process each detected instance
    for i in range(len(instances)):
        # Get the mask
        mask = instances.pred_masks[i].numpy().astype(np.uint8)
        
        # Get class (0=hold, 1=volume) - we want both
        # pred_class = instances.pred_classes[i].item()
        
        # Find contours from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            continue
            
        # Get the largest contour (in case of multiple)
        contour = max(contours, key=cv2.contourArea)
        
        # Get minimum enclosing circle for (x, y, radius)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        # Filter out very small detections (noise)
        if radius < 5:
            continue
            
        holds.append({
            'x': int(x),
            'y': int(y),
            'radius': int(radius),
            'contour': contour,
            'mask': mask,
            'score': instances.scores[i].item() if hasattr(instances, 'scores') else 1.0
        })
    
    logger.info(f"ML detection found {len(holds)} holds")
    return holds

def detect_holds_ml_simple(image):
    """
    Simplified interface: returns list of (x, y, radius) tuples.
    
    Args:
        image: BGR image (numpy array)
        
    Returns:
        List of (x, y, radius) tuples
    """
    holds = detect_holds_ml(image)
    return [(h['x'], h['y'], h['radius']) for h in holds]

def visualize_ml_detections(image, holds, output_path=None):
    """
    Visualize detected holds on an image.
    
    Args:
        image: BGR image
        holds: List from detect_holds_ml()
        output_path: Optional path to save visualization
        
    Returns:
        Annotated image
    """
    result = image.copy()
    
    for hold in holds:
        # Draw mask outline
        if 'contour' in hold:
            cv2.drawContours(result, [hold['contour']], -1, (0, 255, 0), 2)
        
        # Draw center circle
        cv2.circle(result, (hold['x'], hold['y']), 5, (0, 0, 255), -1)
        
        # Draw enclosing circle
        cv2.circle(result, (hold['x'], hold['y']), hold['radius'], (255, 0, 0), 2)
    
    if output_path:
        cv2.imwrite(output_path, result)
        
    return result

# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ml_hold_segmentation.py <image_path>")
        sys.exit(1)
    
    if not is_model_available():
        print(get_model_download_instructions())
        sys.exit(1)
        
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)
    
    holds = detect_holds_ml(image)
    print(f"Detected {len(holds)} holds")
    
    output_path = "ml_detection_result.jpg"
    visualize_ml_detections(image, holds, output_path)
    print(f"Saved visualization to {output_path}")
