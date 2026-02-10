import torch
import os

def inspect():
    if not os.path.exists("pretrained_h36m_detectron_coco.bin"):
        print("Weights file not found.")
        return
        
    chk = torch.load("pretrained_h36m_detectron_coco.bin", map_location='cpu')
    print("Keys in checkpoint:", chk.keys())
    if 'model_pos' in chk:
        state_dict = chk['model_pos']
        print("Model state dict keys:")
        print("All keys:")
        for key, val in state_dict.items():
            print(f"{key}: {val.shape}")
            
if __name__ == "__main__":
    inspect()
