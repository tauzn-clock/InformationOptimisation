import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/scratchdata/processed/hgx"
CNT = 107
TARGET = os.path.join(ROOT,"rgb.mp4")
frame_rate = 12  # Frames per second for the video

frame_width, frame_height = 640, 400

ourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
video_writer = cv2.VideoWriter(TARGET, fourcc, frame_rate, (frame_width, frame_height))

rgb_store = []

for frame_cnt in range(CNT):
    img = Image.open(os.path.join(ROOT,"rgb",f"{frame_cnt}.png"))
    
    open_cv_image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    rgb_store.append(open_cv_image)
    
    # Write the frame to the video
    video_writer.write(open_cv_image)

# Release the video writer and finish the video file
video_writer.release()

TARGET = os.path.join(ROOT,"depth.mp4")

video_writer = cv2.VideoWriter(TARGET, fourcc, frame_rate, (frame_width, frame_height))

depth_store = []

for frame_cnt in range(CNT):
    img = Image.open(os.path.join(ROOT,"depth",f"{frame_cnt}.png"))
    
    plt.imsave("tmp.png", np.array(img))

    open_cv_image = cv2.imread("tmp.png")

    depth_store.append(open_cv_image)
    
    # Write the frame to the video
    video_writer.write(open_cv_image)

# Release the video writer and finish the video file
video_writer.release()

TARGET = os.path.join(ROOT,"repair.mp4")
video_writer = cv2.VideoWriter(TARGET, fourcc, frame_rate, (frame_width, frame_height))

repair_store = []

for frame_cnt in range(CNT):
    img = Image.open(os.path.join(ROOT,"repair",f"{frame_cnt}.png"))
    
    plt.imsave("tmp.png", np.array(img))

    open_cv_image = cv2.imread("tmp.png")

    repair_store.append(open_cv_image)
    
    # Write the frame to the video
    video_writer.write(open_cv_image)

# Release the video writer and finish the video file
video_writer.release()

TARGET = os.path.join(ROOT,"combined.mp4")
video_writer = cv2.VideoWriter(TARGET, fourcc, frame_rate, (frame_width*3, frame_height))

for frame_cnt in range(CNT):
    img = np.zeros((frame_height, frame_width*3, 3), dtype=np.uint8)
    img[:,0:frame_width] = rgb_store[frame_cnt]
    img[:,frame_width:frame_width*2] = depth_store[frame_cnt]
    img[:,frame_width*2:frame_width*3] = repair_store[frame_cnt]
    
    plt.imsave("tmp.png", img)

    open_cv_image = cv2.imread("tmp.png")
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)

    # Write the frame to the video
    video_writer.write(open_cv_image)

# Release the video writer and finish the video file
video_writer.release()