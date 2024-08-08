import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.quantization import quantize_dynamic, get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

from object_tracking.track.config import CONFIG
from object_tracking.track.model.cutie import CUTIE
from object_tracking.track.inference.inference_core import InferenceCore
from object_tracking.track.utils.mask_mapper import MaskMapper
from object_tracking.tools.painter import mask_painter

class BaseTracker:
    def __init__(self, cutie_checkpoint, device) -> None:
        config = OmegaConf.create(CONFIG)
        
        network = CUTIE(config).to(device).eval()
        model_weights = torch.load(cutie_checkpoint, map_location=device)
        network.load_weights(model_weights)

        self.tracker = InferenceCore(network, config)
        self.device = device
        
        self.mapper = MaskMapper()
        self.initialised = False
        
    def quantize_model(self):
        # 모델을 CPU로 이동 (양자화는 CPU에서 수행됨)
        self.network.cpu()
        
        # 기본 양자화 설정 사용
        self.network.qconfig = get_default_qconfig('fbgemm')
        print(self.network.qconfig)

        # 모델 준비 (calibration)
        prepared_model = prepare_fx(self.network, {'': self.network.qconfig})
        
        # 모델 변환 (quantization)
        quantized_model = convert_fx(prepared_model)
        
        # 모델을 다시 디바이스로 이동
        quantized_model.to(self.device)
        
        # 양자화된 모델을 tracker에 할당
        self.network = quantized_model
        self.tracker = InferenceCore(self.network, OmegaConf.create(CONFIG))
    
    @torch.no_grad()
    def clear_memory(self):
        self.tracker.clear_memory()
        self.mapper.clear_labels()

    @torch.no_grad()
    def image_to_torch(self, frame: np.ndarray, device: str = 'cuda'):
            # frame: H*W*3 numpy array
            frame = frame.transpose(2, 0, 1)
            frame = torch.from_numpy(frame).float().to(device, non_blocking=True) / 255
            return frame
        
    @torch.no_grad()
    def track(self, frame, first_frame_annotation=None):
        # 첫 번째 프레임에서 마스크 초기화
        if first_frame_annotation is not None:
            mask, labels = self.mapper.convert_mask(first_frame_annotation)
            mask = torch.tensor(mask, device=self.device)
        else:
            mask = None
            labels = None

        # 입력 준비
        frame_tensor = self.image_to_torch(frame, self.device)
        
        # 트래킹 단계
        probs = self.tracker.step(frame_tensor, mask, labels)
        
        # 필요 없는 텐서 제거
        del mask, labels

        # 확률 맵에서 마스크로 변환
        out_mask = torch.argmax(probs, dim=0).detach().cpu().numpy().astype(np.uint8)
        del probs  # 필요 없는 텐서 제거

        # 최종 마스크 초기화 및 복원
        final_mask = np.zeros_like(out_mask)
        for k, v in self.mapper.remappings.items():
            final_mask[out_mask == v] = k

        # 물체 수 계산 및 페인팅 이미지 초기화
        num_objs = final_mask.max()
        painted_image = frame
        for obj in range(1, num_objs + 1):
            if np.max(final_mask == obj) == 0:
                continue
            painted_image = mask_painter(painted_image, (final_mask == obj).astype('uint8'), mask_color=obj + 1)

        # 메모리 클리어링 (필요할 때만 사용)
        torch.cuda.empty_cache()

        return final_mask, painted_image
    