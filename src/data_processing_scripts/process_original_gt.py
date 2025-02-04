import os
import numpy as np
from PIL import Image
import csv

ROOT = "/scratchdata/nyu_plane/planar_reconstruct"

for frame_cnt in range(1, 1449+1):
    data = np.load(os.path.join(ROOT, f"plane_instance_{frame_cnt}.npz"), allow_pickle=True)

    plane = data["plane_instance"]
    plane_params = np.array(data["plane_param"])

    plane = plane.astype(np.uint8)
    plane = Image.fromarray(plane)
    plane = plane.resize((640, 480), Image.NEAREST)

    plane.save(f"/scratchdata/nyu_plane/original_gt/{frame_cnt-1}.png")

    with open(f"/scratchdata/nyu_plane/original_gt/{frame_cnt-1}.csv", mode='w') as file:
        writer = csv.writer(file)
        for i in range(plane_params.shape[0]):
            writer.writerow([plane_params[i][0], plane_params[i][1], plane_params[i][2]])