import torch
import torch.nn as nn
import sys
import os

# Import the class from the main script to test EXACTLY what is being used
from climbing_pose_analysis import TemporalModel

WEIGHTS_FILE = "pretrained_h36m_detectron_coco.bin"

def test_load():
    if not os.path.exists(WEIGHTS_FILE):
        print("Weights file missing!")
        return

    print("Attempting to create model and load weights...")
    try:
        # Configuration matches our latest edit: 17 joints, 8 layers of width 3, 1, 3, 1...
        model = TemporalModel(17, 2, 17, filter_widths=[3,1,3,1,3,1,3,1], causal=False, dropout=0.25, channels=1024)
        
        checkpoint = torch.load(WEIGHTS_FILE, map_location='cpu')
        model.load_state_dict(checkpoint['model_pos'])
        
        print("SUCCESS! Model loaded without errors.")
    except Exception as e:
        print(f"FAILED: {e}")
        # Print expected vs actual if possible
        # (Torch usually does this in the error message)

if __name__ == "__main__":
    test_load()
