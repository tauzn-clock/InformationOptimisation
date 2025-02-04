from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/scratchdata/nyu_plane"

frame_cnt = 0

img = Image.open(os.path.join(ROOT,"rgb",f"{frame_cnt}.png"))
depth = Image.open(os.path.join(ROOT,"depth",f"{frame_cnt}.png"))
print(np.array(depth).max())
label = Image.open(os.path.join(ROOT,"labels",f"{frame_cnt}.png"))
plane = Image.open(os.path.join(ROOT,"original_gt",f"{frame_cnt}.png"))

plt.imsave("img.png", np.array(img))
plt.imsave("depth.png", np.array(depth))
plt.imsave("label.png", np.array(label))
plt.imsave("plane.png", np.array(plane))