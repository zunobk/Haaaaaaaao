import time
import torch
import cv2
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import Union
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import matplotlib.pyplot as plt
import PIL
from torch.quantization import quantize_dynamic, get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

class BaseSegmenter:
    def __init__(self, SAM_checkpoint, device='cuda:0'):
        print(f"Loading Sengmentation Model to {device}")
        
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model = sam_model_registry["vit_h"](checkpoint=SAM_checkpoint)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)
        self.embedded = False
        
    def quantize_model(self):
        # 모델을 CPU로 이동 (현재 PyTorch에서는 CUDA에서 양자화를 직접 지원하지 않습니다)
        self.model.cpu()
        
        # 기본 양자화 설정 사용
        self.model.qconfig = get_default_qconfig('fbgemm')
        print(self.model.qconfig)

        # 모델 준비 (calibration)
        prepared_model = prepare_fx(self.model, {'': self.model.qconfig})
        
        # 모델 변환 (quantization)
        quantized_model = convert_fx(prepared_model)
        
        # 모델을 다시 디바이스로 이동
        quantized_model.to(self.device)
        
        # 양자화된 모델을 predictor에 할당
        self.model = quantized_model
        self.predictor = SamPredictor(self.model)
        
    @torch.no_grad()
    def reset_image(self):
        # reset image embeding
        self.predictor.reset_image()
        self.embedded = False
    

    @torch.no_grad()
    def set_image(self, image: np.ndarray):
        # PIL.open(image_path) 3channel: RGB
        # image embedding: avoid encode the same image multiple times
        self.orignal_image = image
        if self.embedded:
            print('repeat embedding, please reset_image.')
            return
        self.predictor.set_image(image)
        self.embedded = True
        return

    def predict(self, prompts, mode, multimask=True):
        assert self.embedded, 'prediction is called before set_image (feature embedding).'
        assert mode in ['point', 'mask', 'both'], 'mode must be point, mask, or both'
        
        if mode == 'point':
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                multimask_output=multimask)
        elif mode == 'mask':
            masks, scores, logits = self.predictor.predict(mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        elif mode == 'both':   # both
            masks, scores, logits = self.predictor.predict(point_coords=prompts['point_coords'], 
                                point_labels=prompts['point_labels'], 
                                mask_input=prompts['mask_input'], 
                                multimask_output=multimask)
        else:
            raise("Not implement now!")
        # masks (n, h, w), scores (n,), logits (n, 256, 256)
        return masks, scores, logits