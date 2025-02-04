import numpy as np

def depth_to_coord(depth_image, intrinsic):

    """
    Convert a depth image to a point cloud
    :param depth_image: 2D numpy array of depth image, HxW
    :param intrinsic: 1D numpy array of camera intrinsic parameters, [fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 0, 0]
    :return: 2D numpy array of 3D coordinates, (H*W)x3
    """

    # Get dimensions of the depth image
    H, W = depth_image.shape

    # Generate a grid of (x, y) coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Flatten the arrays
    x = x.flatten()
    y = y.flatten()
    depth = depth_image.flatten()

    # Calculate 3D coordinates
    fx, fy, cx, cy = intrinsic[0], intrinsic[5], intrinsic[2], intrinsic[6]
    z = depth

    x_3d = (x - cx) * z / fx
    y_3d = (y - cy) * z / fy

    # Create a point cloud
    points = np.vstack((x_3d, y_3d, z)).T
    
    index = np.column_stack((x, y))

    return points, index
