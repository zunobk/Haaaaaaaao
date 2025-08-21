import re
import os
import cv2
import numpy as np
from PIL import Image
import torch
import ast
from moviepy.editor import VideoFileClip
import gc



def extract_number(file_name):
    """Extract numbers from a given file name"""
    match = re.search(r'\d+', file_name)
    return int(match.group()) if match else 0


def make_file(video_name):
    base_path = f'.\\static\\videos\\{video_name}'
    origin_images_path = os.path.join(base_path, "origin_images")
    mask_images_path = os.path.join(base_path, "mask_images")
    tracking_images_path = os.path.join(base_path, "tracking_images")
    face_images_path = os.path.join(base_path, "face_images")
    seg_images_path = os.path.join(base_path, "seg_images")

    detect_mask_images_path = os.path.join(base_path, "Detect_mask_images")

    # 각 디렉토리 생성
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(origin_images_path, exist_ok=True)
    os.makedirs(mask_images_path, exist_ok=True)
    os.makedirs(tracking_images_path, exist_ok=True)
    os.makedirs(seg_images_path, exist_ok=True)
    os.makedirs(base_path + "/inpainting_images", exist_ok=True)
    os.makedirs(detect_mask_images_path, exist_ok=True)
    os.makedirs(base_path + "/inpainting_videos", exist_ok=True)
    os.makedirs(base_path + "/segmentation_videos", exist_ok=True)
    os.makedirs(base_path + "/web_inpainting_videos", exist_ok=True)
    os.makedirs(base_path + "/web_segmentation_videos", exist_ok=True)
    
    video_state = {
        "video_name": video_name,
        "video_frame_path": base_path,
        "inpainting_videos" : base_path + "/inpainting_videos",
        "web_inpainting_videos" : base_path + "/web_inpainting_videos",
        "segmentation_videos" : base_path + "/segmentation_videos",
        "web_segmentation_videos" : base_path + "/web_segmentation_videos",
        "inpainting_images_path": base_path + "/inpainting_images",
        "origin_images_path": origin_images_path,
        "mask_images_path": mask_images_path,
        "tracking_images_path": tracking_images_path,
        "face_images_path": face_images_path,
        "bestface_img_path": f"{base_path}\\best_faces",
        "seg_images_path": seg_images_path,
        "detect_mask_output_path": detect_mask_images_path,

        "fps": "",
        "total_frame": "",
        "video_length ": "",
        "frame_size": "",
    }
    return video_state
def get_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length = total_frames / fps
    
    video_state = make_file(os.path.basename(video_path))
    video_state["fps"] = fps
    video_state["total_frame"] = total_frames
    video_state["video_length"] = video_length
    video_state["frame_size"] = (frame_width, frame_height)
    print(video_state["frame_size"])
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            black_image = np.zeros((frame_height, frame_width), np.uint8)
            output_filename = os.path.join(video_state["origin_images_path"], f"{idx}.jpg")
            inpaint_filename = os.path.join(video_state["inpainting_images_path"], f"{idx}.jpg")
            tracking_filename = os.path.join(video_state["tracking_images_path"], f"{idx}.jpg")
            maskimg_filename = os.path.join(video_state["mask_images_path"], f"{idx}.jpg")
            cv2.imwrite(output_filename, frame)
            cv2.imwrite(inpaint_filename, frame)
            cv2.imwrite(tracking_filename, frame)
            cv2.imwrite(maskimg_filename, black_image)
            idx += 1
        else:
            break
        
    return video_state


def re_get_video(video_state, video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(video_state["frame_size"])
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            black_image = np.zeros((frame_height, frame_width), np.uint8)
            output_filename = os.path.join(video_state["origin_images_path"], f"{idx}.jpg")
            inpaint_filename = os.path.join(video_state["inpainting_images_path"], f"{idx}.jpg")
            tracking_filename = os.path.join(video_state["tracking_images_path"], f"{idx}.jpg")
            maskimg_filename = os.path.join(video_state["mask_images_path"], f"{idx}.jpg")
            cv2.imwrite(output_filename, frame)
            cv2.imwrite(inpaint_filename, frame)
            cv2.imwrite(tracking_filename, frame)
            cv2.imwrite(maskimg_filename, black_image)
            idx += 1
        else:
            break
        
    


def select_face(video_state, face_id):
    # 파일 경로 생성
    image_path = os.path.join(video_state["face_images_path"], rf"ID_{face_id}/coordinates.txt")
    
    person_state_list = []
    
    # 파일 열기
    with open(image_path, 'r') as file:
        # 모든 줄을 읽어서 처리
        lines = file.readlines()
    
    # 각 줄에 대해 처리
    for line in lines:
        frame, x, y = map(int, line.strip().split(', '))
        person_state = {
            "face_id": face_id,
            "frame_number": frame,
            "face_points": (x, y)
        }
        person_state_list.append(person_state)
    
    return person_state_list

def change_mask(video_state, person_state):
    #frist_frame_image = cv2.imread(video_state['origin_images_path'] + "/frame_0.jpg")
    #frist_frame_image = cv2.imread("frame_0.jpg")
    file_path = f"{video_state['video_frame_path']}\\masked_{person_state['frame_number']}.jpg"
    #file_path = video_state["video_frame_path"]+"\\masked_image.jpg"
    image = Image.open(file_path).convert('L')
    image_np = np.array(image)
    binary_mask = image_np > 0
    return binary_mask


def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
def generate_video_from_frames(video_state, user_video_name):
        
    painted_frame_files = sorted(
    [os.path.join(video_state["tracking_images_path"], f) for f in os.listdir(video_state["tracking_images_path"]) if f.endswith('.png') or f.endswith('.jpg')],
    key=lambda x: extract_number(os.path.basename(x))
    )
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_state["segmentation_videos"]+f"/{user_video_name}.mp4", fourcc, video_state["fps"], video_state["frame_size"])
    
    for painted_frame in painted_frame_files:   #[A,B]
        image_path = os.path.join(painted_frame)
        frame = cv2.imread(image_path)
        out.write(frame)
        
    out.release()

    convert_video_with_moviepy(video_state["segmentation_videos"]+f"/{user_video_name}.mp4", video_state["web_segmentation_videos"]+f"/{user_video_name}.mp4")

def convert_video_with_moviepy(input_video_path, output_video_path):
    clip = VideoFileClip(input_video_path)
    clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')
    #원본 삭제


    


def read_first_line_from_coordinates(video_state):
    # base_dir은 아이디 폴더들이 있는 기본 디렉토리입니다.
    people_list = []
    
    for root, dirs, files in os.walk(video_state["face_images_path"]):
        for dir_name in dirs:
            coord_file_path = os.path.join(root, dir_name, "coordinates.txt")
            
            if os.path.exists(coord_file_path):
                with open(coord_file_path, 'r') as file:
                    first_line = file.readline().strip()  # 첫 번째 줄 읽기
                    if first_line:
                        try:
                            # 첫 번째 줄에서 frame, x, y 값을 분리
                            frame, x, y = map(int, first_line.split(', '))
                            people_state = {
                                "ID" : dir_name,
                                "frame" : frame,
                                "points" : (x, y)
                            }
                            people_list.append(people_state)
                        except ValueError:
                            print(f"Error parsing line in {dir_name}: {first_line}")
            else:
                print(f"No coordinates.txt found in {dir_name}")
    
    return people_list

def search_video_bestfaces(video_name, video_faces):
    face_info = None  # 초기화

    for i in range(len(video_faces)):
        if video_faces[i]['video_name'] == video_name:
            face_info = video_faces[i]["best_faces"]
            break  # 찾으면 바로 종료

    if face_info is None:
        print(f"No faces found for video: {video_name}")

    return face_info
