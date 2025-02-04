import sys
sys.path.append('/HighResMDE/segment-anything')

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

IMG_PATH = "/scratchdata/processed/stair_down/rgb/1.png"
DEVICE="cuda:0"

img = Image.open(IMG_PATH)
img = np.array(img)

sam = sam_model_registry["vit_h"](checkpoint="/scratchdata/sam_vit_h_4b8939.pth").to(DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

print(masks)
print(len(masks))
for i in range(len(masks)):
    save_img = img
    plt.imsave(f"./visualise/mask_{i}.png", masks[i]["segmentation"])
