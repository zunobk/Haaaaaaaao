from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
import torch
import os
import cv2
import numpy as np
from detectron2.structures import Instances, Boxes
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm  # tqdm 임포트


class Detector:
    def __init__(self):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 임계값 설정
        self.cfg.MODEL.DEVICE = "cuda"

        self.predictor = DefaultPredictor(self.cfg)
 
    def filter_person_instances2(self, predictions, face_coords=None):
        # Assuming the class "person" has the class id 0
        person_class_id = 0
        instances = predictions["instances"]
        person_indices = [i for i, pred_class in enumerate(instances.pred_classes) if pred_class == person_class_id]

        # Create a new Instances object containing only person instances
        person_instances = instances[person_indices]

        # If face_coords is provided, filter instances based on the face coordinates
        if face_coords:
            face_x, face_y = face_coords
            valid_indices = []
            for i in range(len(person_instances)):
                box = person_instances.pred_boxes[i].tensor.cpu().numpy()[0]  # Convert to CPU tensor before numpy
                if box[0] <= face_x <= box[2] and box[1] <= face_y <= box[3]:
                    valid_indices.append(i)
            person_instances = person_instances[valid_indices]

        predictions["instances"] = person_instances
        return predictions

    def mask_image(self,video_state, person_state):  # 원하는 좌표에 세그멘테이션됨.
        imagePath = video_state["origin_images_path"]+f'\\{person_state["frame_number"]}.jpg'
        image = cv2.imread(imagePath)
        try:
            predictions = self.predictor(image)
            predictions = self.filter_person_instances2(predictions, person_state["face_points"])

            # Dilate masks and create a full mask
            for i in range(len(predictions["instances"].pred_masks)):
                mask = predictions["instances"].pred_masks[i].cpu().numpy()
                kernel = np.ones((10, 10), np.uint8)
                dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
                predictions["instances"].pred_masks[i] = torch.from_numpy(dilated_mask).cuda()

            # 마스크 이미지 생성
            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for mask in predictions["instances"].pred_masks:
                full_mask = np.logical_or(full_mask, mask.cpu().numpy())

            # 흑백 마스크 이미지 생성 (배경은 검정색, 객체는 하얀색)
            mask_image = np.zeros_like(image)
            mask_image[full_mask] = [255, 255, 255]

            # 이미지 저장
            os.makedirs(video_state["video_frame_path"], exist_ok=True)
            mask_image_path = os.path.join(video_state["video_frame_path"], f"masked_{person_state['frame_number']}.jpg")
            cv2.imwrite(mask_image_path, mask_image)
        except AssertionError:
                print(f"얼굴인식 실패{person_state['ID']}")
                pass
        
        
    def segmentation_image(self, video_state, people_list):
            
            
        for i in range(len(people_list)):
            imagePath = video_state["origin_images_path"]+rf"/{people_list[i]['frame']}.jpg"
            image = cv2.imread(imagePath)

            predictions = self.predictor(image)
            predictions = self.filter_person_instances2(predictions, people_list[i]["points"])

            viz = Visualizer(image[:, :, ::-1], metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                            instance_mode=ColorMode.IMAGE)

            try:
                output = viz.draw_instance_predictions(predictions["instances"].to("cpu"), labels=[people_list[i]["ID"]])
                result_image = output.get_image()[:, :, ::-1]


                mask_image_path = os.path.join(video_state["seg_images_path"], rf"{people_list[i]['ID']}_Seg.jpg")
                cv2.imwrite(mask_image_path, result_image)
            except AssertionError:
                print(f"얼굴인식 실패{people_list[i]['ID']}")
                continue