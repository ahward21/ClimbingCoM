import cv2
import numpy as np

def create_dummy_video(filename='dummy_climb.mp4', width=640, height=480, frames=60):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    # Create a moving "person" (rectangle)
    # Start at bottom
    x, y = width // 2, height - 100
    
    for i in range(frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw "person"
        # Move up and slightly right
        y -= 2
        x += int(np.sin(i / 10) * 5)
        
        cv2.rectangle(img, (x-20, y-50), (x+20, y+50), (255, 255, 255), -1)
        
        out.write(img)
        
    out.release()
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_video()
