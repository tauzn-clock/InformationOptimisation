import numpy as np

def set_depth(depth,intrinsic,mask,normal,distance):
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)

    H,W = depth.shape
    Z = depth.flatten()

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
    x_3d = (x - cx) * Z / fx
    y_3d = (y - cy) * Z / fy
    POINTS = np.vstack((x_3d, y_3d, Z)).T

    DIRECTION_VECTOR = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    distance = (-distance/np.dot(DIRECTION_VECTOR, normal.T)) * DIRECTION_VECTOR[:,2]
    
    distance = distance.reshape(H,W)

    return distance*mask
