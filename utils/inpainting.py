import argparse
from PIL import Image
import torch
import torchvision.transforms as T
import os
import cv2
from tqdm import tqdm
from natsort import natsorted
from utils.tools import convert_video_with_moviepy



def inpaint_image(image_path, mask_path, out_path, generator, device):
    # 이미지와 마스크 불러오기
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # 입력 준비
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    image = (image * 2 - 1.).to(device)  # 이미지 값을 [-1, 1] 범위로 매핑
    mask = (mask > 0.5).to(dtype=torch.float32, device=device)  # 1.: masked, 0.: unmasked

    image_masked = image * (1. - mask)  # 이미지에 마스크 적용

    ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
    x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # 채널 결합

    with torch.inference_mode():
        _, x_stage2 = generator(x, mask)

    # 이미지를 완성
    image_inpainted = image * (1. - mask) + x_stage2 * mask

    # 완성된 이미지 저장
    img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    img_out = img_out.cpu().numpy()
    img_out = Image.fromarray(img_out)
    img_out.save(out_path)

    #print(f"저장된 출력 파일: {out_path}")

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
    

def main(origin_images_path, mask_images_path, ouput_path, inpaint_ouput_path, inpainting_images_path):

    generator_state_dict = torch.load("weights\states_pt_places2.pth")['G']

    if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
        from object_tracking.model.networks import Generator
    else:
        from object_tracking.model.networks_tf import Generator

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available() and use_cuda_if_available else 'cpu')

    # 네트워크 설정

    generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
    generator.load_state_dict(generator_state_dict, strict=True)
    # 각 이미지 처리
    frame_files = natsorted(os.listdir(origin_images_path))
    mask_files = natsorted(os.listdir(mask_images_path))

    

    for frame_file in tqdm(frame_files, desc="Processing frames"):
        frame_path = os.path.join(origin_images_path, frame_file)
        mask_path = os.path.join(mask_images_path, frame_file)  # 마스크 파일이 프레임 파일과 같은 이름이라고 가정
        out_path = os.path.join(inpainting_images_path, frame_file)

        if frame_file in mask_files:
            inpaint_image(frame_path, mask_path, out_path, generator, device)
        else:
            image = Image.open(frame_path)
            image.save(out_path)

    # 완성된 프레임으로 동영상 생성
    create_video(inpainting_images_path, ouput_path)
    convert_video_with_moviepy(input_video_path=ouput_path, output_video_path=inpaint_ouput_path)