
import numpy as np
from object_tracking.segment.Segmenter import BaseSegmenter
from PIL import Image, ImageDraw, ImageOps
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from object_tracking.tools.painter import mask_painter, point_painter

mask_color = 3
mask_alpha = 0.7
contour_color = 1
contour_width = 5
point_color_ne = 8
point_color_ps = 50
point_alpha = 0.9
point_radius = 15
contour_color = 2
contour_width = 5


class SamControler():
    def __init__(self, SAM_checkpoint='./weights/sam_vit_h_4b8939.pth', device='cuda:0'):
        self.sam_controler = BaseSegmenter(SAM_checkpoint, device)
        
    
    def create_mask(self, image: np.ndarray, points:np.ndarray, labels: np.ndarray, multimask=True,mask_color=3):
        '''
        it is used in first frame in video
        return: mask, logit, painted image(mask+point)
        '''
        self.sam_controler.set_image(image)
        #origal_image = self.sam_controler.orignal_image
        neg_flag = labels[-1]
        if neg_flag==1:
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            prompts = {
                'point_coords': points,
                'point_labels': labels,
                'mask_input': logit[None, :, :]
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'both', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
        else:
           #find positive
            prompts = {
                'point_coords': points,
                'point_labels': labels,
            }
            masks, scores, logits = self.sam_controler.predict(prompts, 'point', multimask)
            mask, logit = masks[np.argmax(scores)], logits[np.argmax(scores), :, :]
            
        
        assert len(points)==len(labels)
        
        painted_image = mask_painter(image, mask.astype('uint8'), mask_color, mask_alpha, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels>0)],axis = 1), point_color_ne, point_alpha, point_radius, contour_color, contour_width)
        painted_image = point_painter(painted_image, np.squeeze(points[np.argwhere(labels<1)],axis = 1), point_color_ps, point_alpha, point_radius, contour_color, contour_width)
        painted_image = Image.fromarray(painted_image)
        
        return mask, painted_image