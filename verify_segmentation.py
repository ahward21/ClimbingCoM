import cv2
import logging
import numpy as np
import hold_segmentation_utils as hsu
from climbing_pose_analysis import extract_hold_positions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VIDEO_PATH = "f:/ClimbingEst/dummy_climb.mp4"
OUTPUT_IMAGE = "f:/ClimbingEst/verified_holds.jpg"

def verify_holds():
    print(f"Processing {VIDEO_PATH}...")
    
    try:
        hold_positions, hold_img = extract_hold_positions(VIDEO_PATH, sample_frames=10)
        
        print(f"Detected {len(hold_positions)} holds.")
        active_holds = [h for h in hold_positions if h[3]]
        print(f"Active holds detected: {len(active_holds)}")
        
        # Save output
        cv2.imwrite(OUTPUT_IMAGE, hold_img)
        print(f"Saved visualization to {OUTPUT_IMAGE}")
        
    except Exception as e:
        print(f"Error during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_holds()
