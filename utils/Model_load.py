
import numpy as np
from tqdm import tqdm
import re
import os
import cv2
import scipy
from object_tracking.track.tracking import BaseTracker

def extract_number(file_name):
    """Extract numbers from a given file name"""
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0

class Model_load():
    def __init__(self,cutie_checkpoint="weights/cutie-base-mega.pth", device='cuda:0'):
        self.cutie = BaseTracker(cutie_checkpoint, device)
        
        
    def tracking_person(self, video_state, mask:np.ndarray, person_state, boundary_dilation=6):
        
        
        frame_files = sorted(
        [os.path.join(video_state["origin_images_path"], f) for f in os.listdir(video_state["origin_images_path"]) if f.endswith('.png') or f.endswith('.jpg')],
        key=lambda x: extract_number(os.path.basename(x))
        )
        
        files = frame_files[person_state[0]:person_state[1]]
        
        for i, frame_file in enumerate(tqdm(files, desc="Tracking image")):
            
            i = i + person_state[0]
            frame = cv2.imread(frame_file)
                        
            if frame is None:
                print(f"Failed to read frame {frame_file}")
                continue
            
            if i == person_state[0]:
                mask, painted_image = self.cutie.track(frame, mask)
            else:
                mask, painted_image = self.cutie.track(frame)   

            true_ratio = np.mean(mask == True)
            if not np.any(mask) or true_ratio >= 0.80:
                 break
            
            if boundary_dilation > 0:
                mask = scipy.ndimage.binary_dilation(mask, iterations=boundary_dilation).astype(np.uint8)
            
            binary_mask = mask.astype(np.uint8) * 255
            rgb_mask = cv2.merge([binary_mask, binary_mask, binary_mask])
            mask_path = os.path.join(video_state["mask_images_path"], f"{i}.jpg")
            painted_image_path = os.path.join(video_state["tracking_images_path"], f"{i}.jpg")
            
            cv2.imwrite(mask_path, rgb_mask)
            cv2.imwrite(painted_image_path, painted_image)