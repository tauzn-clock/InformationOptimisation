import os
import numpy as np
from PIL import Image
import re

ROOT = "/scratchdata/nyu_depth_v2/official_splits/"

#Iterate through the directory and extract the data

def extract_data(ROOT):
    for root, dirs, files in os.walk(ROOT):
        for file in files:
            if file.startswith("rgb"):
                #Get index of image
                index = int(re.findall(r'\d+', file)[0])
                print(index)
                #Open the image
                image = Image.open(os.path.join(root, file))
                #Save the image
                image.save(f"/scratchdata/nyu_plane/rgb/{index}.png")
            if file.startswith("sync_depth"):
                #Get index of image
                index = int(re.findall(r'\d+', file)[0])
                print(index)
                #Open the image
                depth = Image.open(os.path.join(root, file))
                print(np.array(depth).max(), np.array(depth).dtype)
                #Save the image
                depth.save(f"/scratchdata/nyu_plane/depth/{index}.png")

extract_data(ROOT)