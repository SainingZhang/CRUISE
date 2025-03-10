import cv2
import numpy as np
import os
from tqdm import tqdm

def create_side_by_side_video(img_dir, output_path, fps=24, start_idx=0, end_idx=0):
    # Get the first images to determine dimensions
    img0 = cv2.imread(os.path.join(img_dir, f'{start_idx:06d}_0_bbox.png'))
    img1 = cv2.imread(os.path.join(img_dir, f'{start_idx:06d}_1_bbox.png'))
    
    if img0 is None or img1 is None:
        raise Exception("Could not read first images. Please check the path and file names.")
    
    # Get dimensions
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    # Create video writer
    # Combined width will be sum of individual widths, height will be max of heights
    combined_width = w0 + w1
    combined_height = max(h0, h1)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height))
    
    # Process each frame
    for idx in tqdm(range(len([f for f in os.listdir(img_dir) if f.endswith('_0_bbox.png')]))):
        # Read both images
        img0_path = os.path.join(img_dir, f'{idx:06d}_0_bbox.png')
        img1_path = os.path.join(img_dir, f'{idx:06d}_1_bbox.png')
        
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        
        if img0 is None or img1 is None:
            print(f"Warning: Could not read images for index {idx}")
            continue
        
        # Resize images to match the height if necessary
        if h0 != combined_height:
            img0 = cv2.resize(img0, (int(w0 * combined_height / h0), combined_height))
        if h1 != combined_height:
            img1 = cv2.resize(img1, (int(w1 * combined_height / h1), combined_height))
        
        # Combine images side by side
        combined_img = np.hstack((img0, img1))
        
        # Write frame
        out.write(combined_img)
    
    # Release video writer
    out.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    # Configuration
    img_directory = "/mnt/xuhr/street-gs/output/dair_seq_0007/exp_1/train/ours_100000/using_modified_bbox/bbox"  # Replace with your image directory
    output_video = "0007_video.mp4"
    fps = 24
    start_frame = 0
    end_frame = 153
    
    create_side_by_side_video(
        img_dir=img_directory,
        output_path=output_video,
        fps=fps,
        start_idx=start_frame,
        end_idx=end_frame
    )