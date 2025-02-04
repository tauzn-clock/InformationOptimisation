import scipy.io
import mat73
from PIL import Image
import numpy as np

mat = scipy.io.loadmat('/scratchdata/nyu_plane/splits.mat')

print(mat.keys())

print(mat['trainNdxs'].shape)
print(mat['testNdxs'].shape)

mat = mat73.loadmat('/scratchdata/nyu_plane/nyu_depth_v2_labeled.mat')

print(mat.keys())

print(mat['depths'].shape)
print(mat['depths'][:,:,0].shape)
print(mat['depths'][:,:,0].dtype)
print(mat['depths'][:,:,0].max())
print(mat['images'].shape)
print(mat['images'][:,:,0].shape)
print(mat['images'][:,:,0].dtype)
print(mat['images'][:,:,0].max())
print(mat['labels'].shape)
print(mat['labels'][:,:,0].shape)
print(mat['labels'][:,:,0].dtype)
print(mat['labels'][:,:,0].max())

for i in range(1449):
    img = Image.fromarray(mat['images'][:,:,:,i])
    img.save(f"/scratchdata/nyu_plane/rgb/{i}.png")
    depth = np.array(mat['depths'][:,:,i]) * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(f"/scratchdata/nyu_plane/depth/{i}.png")
    label = Image.fromarray(mat['labels'][:,:,i])
    label.save(f"/scratchdata/nyu_plane/labels/{i}.png")