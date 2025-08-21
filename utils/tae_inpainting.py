import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import os
import cv2
from tqdm import tqdm
from natsort import natsorted
from utils.tools import convert_video_with_moviepy
import numpy as np

from Inpainting_Model.dataset import CustomTransforms
from Inpainting_Model.model import UNetInpainting


def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image


def inpaint_image(image_path, mask_path, out_path, generator, device, transform, patch_size=256):
    input_image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    input_image_rgb = cv2.cvtColor(input_image_bgr, cv2.COLOR_BGR2RGB)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    original_size = (input_image_rgb.shape[1], input_image_rgb.shape[0])

    # 패치로 이미지 분할
    patches = []
    masks = []
    for y in range(0, original_size[1], patch_size):
        for x in range(0, original_size[0], patch_size):
            patch = input_image_rgb[y:y + patch_size, x:x + patch_size]
            mask_patch = mask_image[y:y + patch_size, x:x + patch_size]
            if patch.shape[:2] != (patch_size, patch_size):
                pad_y = patch_size - patch.shape[0]
                pad_x = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_y), (0, pad_x), (0, 0)), mode='constant', constant_values=0)
                mask_patch = np.pad(mask_patch, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            patches.append(patch)
            masks.append(mask_patch)

    # 패치 처리
    inpainted_patches = []
    for patch, mask_patch in zip(patches, masks):
        input_image, mask = transform(patch, mask_patch)
        mask = (mask > 0).float().expand(3, -1, -1)
        masked_image = input_image * (1 - mask)
        input_tensor = torch.cat((masked_image, mask[0:1]), dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = generator(input_tensor)

        output_image = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        mask = mask.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        input_image_rgb = input_image.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        combined_output = input_image_rgb * (1 - mask) + output_image * mask
        combined_output = denormalize(combined_output)
        inpainted_patches.append(combined_output)

    # 패치를 다시 결합하여 원본 크기의 이미지로 복원
    combined_output_image = np.zeros((original_size[1], original_size[0], 3), dtype=np.float32)
    idx = 0
    for y in range(0, original_size[1], patch_size):
        for x in range(0, original_size[0], patch_size):
            patch = inpainted_patches[idx]
            h, w = min(patch_size, original_size[1] - y), min(patch_size, original_size[0] - x)
            combined_output_image[y:y + h, x:x + w] = patch[:h, :w]
            idx += 1

    # PIL을 사용하여 좌우 반전 문제 방지
    output_image_pil = Image.fromarray((combined_output_image * 255).astype(np.uint8))
    output_image_pil.save(out_path, quality=95)


def create_video(image_dir, output_video_path, fps=30):
    if not os.path.exists(image_dir):
        print(f"디렉토리가 존재하지 않습니다: {image_dir}")
        return

    images = natsorted([img for img in os.listdir(image_dir) if img.endswith(".jpg")])
    if not images:
        print("디렉토리에 이미지가 없습니다.")
        return

    frame = cv2.imread(os.path.join(image_dir, images[0]))
    if frame is None:
        print(f"첫 번째 프레임 읽기 오류: {images[0]}")
        return

    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        img_path = os.path.join(image_dir, image)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"프레임 읽기 오류: {img_path}")

    video.release()


def main(video_state, user_edit_name):
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetInpainting(in_channels=4, out_channels=3).to(device)
    generator.load_state_dict(torch.load(r"gan_inpainting_generator_epoch_280.pth", map_location=device))
    # generator.load_state_dict(torch.load(r"C:\Users\Admin\Downloads\gan_inpainting_generator_07_22_02.pth", map_location=device))

    generator.eval()
 
    # 사용자 정의 변환 함수
    transform = CustomTransforms(train=False)

    # 각 이미지 처리
    frame_files = natsorted(os.listdir(video_state['inpainting_images_path']))
    mask_files = natsorted(os.listdir(video_state['mask_images_path']))

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_path = os.path.join(video_state['inpainting_images_path'], frame_file)
        mask_path = os.path.join(video_state['mask_images_path'], frame_file)  # 마스크 파일이 프레임 파일과 같은 이름이라고 가정
        out_path = os.path.join(video_state['inpainting_images_path'], frame_file)

        if frame_file in mask_files:
            inpaint_image(frame_path, mask_path, out_path, generator, device, transform)
        else:
            image = Image.open(frame_path)
            image.save(out_path)

    # 완성된 프레임으로 동영상 생성
    create_video(video_state['inpainting_images_path'], video_state['inpainting_videos']+f"/{user_edit_name}.mp4")
    convert_video_with_moviepy(input_video_path=video_state['inpainting_videos']+f"/{user_edit_name}.mp4", 
                               output_video_path=video_state['web_inpainting_videos']+f"/{user_edit_name}.mp4")