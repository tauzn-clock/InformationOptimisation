import numpy as np

def get_3d(depth, INTRINSICS):
    H, W = depth.shape
    depth = depth.flatten()
    # Generate a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()

    # Calculate 3D coordinates
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[1], INTRINSICS[2], INTRINSICS[3]
    z = depth

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Create a point cloud
    points = np.vstack((x_3d, y_3d, z)).T
    
    return points

def get_normal_adj(depth, INTRINSICS):
    H, W = depth.shape
    points = get_3d(depth, INTRINSICS)
    points = points.reshape(H, W, 3)

    #Pad points along the edges
    points = np.pad(points, ((1, 1), (1, 1), (0, 0)))

    dx = points[1:-1, 2:, :] - points[1:-1, :-2, :]
    dy = points[2:, 1:-1, :] - points[:-2, 1:-1, :]
    # Cross product
    normal = np.cross(dx, dy)
    normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-16)
    normal[depth == 0] = 0

    normal = normal * np.where(normal[:, :, 2] >= 0, 1, -1).reshape(H, W, 1)
    
    return normal

def get_normal_svd(depth, INTRINSICS, kernel_size=3):
    H, W = depth.shape
    points = get_3d(depth, INTRINSICS)
    points = points.reshape(H, W, 3)

    # Pad points along the edges
    points = np.pad(points, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2), (0, 0)))
    normal = np.zeros((H, W, 3))

    for i in range(kernel_size//2, H + kernel_size//2):
        for j in range(kernel_size//2, W + kernel_size//2):
            patch = points[i-kernel_size//2:i+kernel_size//2+1, j-kernel_size//2:j+kernel_size//2+1, :].reshape(-1, 3)
            patch = patch[patch[:, 2] > 0]
            if len(patch) < kernel_size**2 // 2:
                continue
            cov_matrix = np.cov(patch.T)
            _, s, vh = np.linalg.svd(cov_matrix)
            if s[2] / s[1] < 0.1:  # Check if the second singular value is much smaller than the third
                normal[i-kernel_size//2, j-kernel_size//2] = vh[2]

    normal = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-16)
    normal[depth == 0] = 0

    normal = normal * np.where(normal[:, :, 2] >= 0, 1, -1).reshape(H, W, 1)

    return normal