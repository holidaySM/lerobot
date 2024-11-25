import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

# 루트 디렉토리 설정
root_dir = '/mnt/data2/pushany_rollouts_as_hdf5'

def process_video(args):
    video_file, frame_dir = args
    
    # 비디오 파일의 고유 아이디 추출
    filename = os.path.basename(video_file)
    unique_id = filename.split('observation.image_episode_')[1].split('.mp4')[0]
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Failed to open video file: {video_file}")
        return
    
    # 원본 FPS와 원하는 FPS 설정
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    desired_fps = 10
    frame_interval = int(max(original_fps / desired_fps, 1))
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 선택 (10Hz에 맞게)
        if frame_count % frame_interval == 0:
            # 프레임 크기 조정 및 색상 변환 (BGR -> RGB)
            frame_resized = cv2.resize(frame, (224, 224))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        frame_count += 1
    
    cap.release()
    
    if frames:
        # NumPy 배열로 변환 후 텐서로 변환
        frames_np = np.stack(frames)  # Shape: (N, 224, 224, 3)
        frames_np = np.transpose(frames_np, (0, 3, 1, 2))  # Shape: (N, 3, 224, 224)
        frames_tensor = torch.from_numpy(frames_np.astype(np.uint8))
        
        # .pt 파일로 저장
        output_path = os.path.join(frame_dir, f"{unique_id}.pt")
        torch.save(frames_tensor, output_path)
    else:
        print(f"No frames extracted from video {video_file}")

import multiprocessing

num_workers = multiprocessing.cpu_count()
print(f"Number of workers: {num_workers}")
# 환경(env) 디렉토리들 가져오기
env_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

for env in env_dirs:
    video_dir = os.path.join(root_dir, env, 'videos')
    frame_dir = os.path.join(root_dir, env, 'frames')
    
    # frames 디렉토리가 없으면 생성
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    # 비디오 파일들 가져오기
    video_files = glob.glob(os.path.join(video_dir, '*.mp4'))
    
    # 병렬 처리 및 tqdm을 활용한 진행 상황 표시
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, [(video_file, frame_dir) for video_file in video_files]), total=len(video_files), desc=f"Processing {env}"))
