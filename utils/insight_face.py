import os
import cv2
import re
import numpy as np
from tqdm import tqdm
from multiprocessing import Value, Lock, Manager
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='insightface.utils.transform')

def extract_number(file_name):
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0

def update_moving_average(existing_encoding, new_encoding, alpha=0.1):
    return alpha * new_encoding + (1 - alpha) * existing_encoding

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

def save_coordinates(face_id, frame_idx, x, y, output_dir):
    face_dir = os.path.join(output_dir, f"ID_{face_id}")
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    coordinates_file = os.path.join(face_dir, 'coordinates.txt')
    with open(coordinates_file, 'a') as f:
        f.write(f"{frame_idx}, {x}, {y}\n")
        
def save_face_images(frame, face_bbox, face_id, frame_idx, output_dir):
    x1, y1, x2, y2 = face_bbox
    cropped_face = frame[y1:y2, x1:x2]
    face_dir = os.path.join(output_dir, f"ID_{face_id}")
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    face_img_path = os.path.join(face_dir, f"{frame_idx}.jpg")
    cv2.imwrite(face_img_path, cropped_face)
    return face_img_path



class InsightFace:
    def __init__(self):
        self.app = FaceAnalysis(providers=['CUDAExecutionProvider'])  # GPU가 사용 가능하면 'CUDAExecutionProvider' 사용
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    
    def process_frames(self, video_state, tolerance=0.4):
        #파일 정렬
        frame_files = sorted(
            [os.path.join(video_state["origin_images_path"], f) for f in os.listdir(video_state["origin_images_path"]) if f.endswith('.png') or f.endswith('.jpg')],
            key=lambda x: extract_number(os.path.basename(x))
            )
        # os.makedirs('faces', exist_ok=True) # 얼굴 담는 부분?
        with Manager() as manager:
            known_face_encodings = []
            known_face_ids = []
            face_id_counter = Value('i', 0)
            face_id_counter_lock = Lock()
            id_frame_ranges = manager.dict()
            id_frame_ranges_lock = Lock()
            face_encodings_per_id = manager.dict()
            best_face_per_id = manager.dict()
            last_seen_frame = manager.dict()  # 얼굴이 마지막으로 감지된 프레임
            for i, frame_file in enumerate(tqdm(frame_files, desc="Recognition Face")):
                frame = cv2.imread(frame_file)
                
                if frame is None:
                    print(f"Failed to read frame {frame_file}")
                    continue
                
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    faces = self.app.get(rgb_frame)
                    face_encodings = [face.normed_embedding for face in faces]
                    current_face_ids = []
                    for face, face_encoding in zip(faces, face_encodings):
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        matches = []
                        face_similarities = []
                        for encoding in known_face_encodings:
                            if isinstance(encoding, np.ndarray) and isinstance(face_encoding, np.ndarray) and encoding.ndim == 1 and face_encoding.ndim == 1:
                                similarity = cosine_similarity(encoding, face_encoding)
                                face_similarities.append(similarity)
                                matches.append(similarity > tolerance)
                            else:
                                print(f"벡터의 차원이 맞지 않습니다. Encoding: {encoding}, Face Encoding: {face_encoding}")
                                continue
                        face_id = None
                        if matches and any(matches):
                            best_match_index = np.argmax(face_similarities)
                            if matches[best_match_index]:
                                face_id = known_face_ids[best_match_index]
                                known_face_encodings[best_match_index] = update_moving_average(
                                    known_face_encodings[best_match_index], face_encoding)
                        else:
                            recent_face_encodings = []
                            recent_face_ids = []
                            for id, encodings in face_encodings_per_id.items():
                                recent_face_encodings.extend(encodings)
                                recent_face_ids.extend([id] * len(encodings))
                            if recent_face_encodings:
                                recent_similarities = []
                                for encoding in recent_face_encodings:
                                    if isinstance(encoding, np.ndarray) and isinstance(face_encoding, np.ndarray) and encoding.ndim == 1 and face_encoding.ndim == 1:
                                        similarity = cosine_similarity(encoding, face_encoding)
                                        recent_similarities.append(similarity)
                                    else:
                                        print(f"벡터의 차원이 맞지 않습니다. Recent Encoding: {encoding}, Face Encoding: {face_encoding}")
                                        continue
                                if any([sim > tolerance for sim in recent_similarities]):
                                    best_recent_match_index = np.argmax(recent_similarities)
                                    if recent_similarities[best_recent_match_index] > tolerance:
                                        face_id = recent_face_ids[best_recent_match_index]
                            if face_id is None:
                                with face_id_counter_lock:
                                    face_id_counter.value += 1
                                    face_id = face_id_counter.value
                                    known_face_encodings.append(face_encoding)
                                    known_face_ids.append(face_id)
                                    face_encodings_per_id[face_id] = [face_encoding]
                            else:
                                face_encodings_per_id[face_id].append(face_encoding)
                        current_face_ids.append(face_id)
                        with id_frame_ranges_lock:
                            if face_id in id_frame_ranges:
                                id_frame_ranges[face_id].append(i)
                            else:
                                id_frame_ranges[face_id] = [i]
                        # 얼굴의 바운딩 박스 중앙 좌표 계산
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        if face_id not in last_seen_frame or i - last_seen_frame[face_id] > 1:
                            save_coordinates(face_id, i, center_x, center_y, output_dir=video_state["face_images_path"])
                        last_seen_frame[face_id] = i
                        face_img_path = save_face_images(frame, (x1, y1, x2, y2), face_id, i, output_dir=video_state["face_images_path"])
                        # Update best face per ID
                        with id_frame_ranges_lock:
                            if face_id not in best_face_per_id or np.linalg.norm(face_encoding) > np.linalg.norm(face_encodings_per_id[face_id][0]):
                                best_face_per_id[face_id] = face_img_path

                except Exception as e:
                    continue



            
            os.makedirs(video_state["bestface_img_path"], exist_ok=True)
            for face_id, best_face_path in best_face_per_id.items():
                img = cv2.imread(best_face_path)
                if img is not None:
                    cv2.imwrite(os.path.join(video_state["bestface_img_path"], f"ID_{face_id}.jpg"), img)
            print("가장 잘 인식된 얼굴 이미지를 best_faces 폴더에 저장했습니다.")