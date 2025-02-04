import numpy as np

def get_plane(R, EPSILON, param=[0, 0, 1, 0], noise = 3):
    # Parameters for the flat plane
    num_points = int(R//EPSILON)  # Number of points on the flat plane
    plane_size = R  # Size of the plane (e.g., 10x10 units)
    plane_height = 0.0  # The Z value for the flat plane

    # Create a flat plane (a grid of points)
    x_vals = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    y_vals = np.linspace(-plane_size / 2, plane_size / 2, num_points)
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    param = np.array(param)
    normal = param[:3] / np.linalg.norm(param[:3])
    z_vals = - param[3] + (- normal[0] * x_grid - normal[1] * y_grid) / (normal[2] + 1e-7)
    # Round z_vals
    z_vals = z_vals//EPSILON * EPSILON

    # Flatten the grid to create a 1D array of points
    plane_points = np.vstack((x_grid.flatten(), y_grid.flatten(), z_vals.flatten())).T
    
    # Stay within R cube
    mask = (plane_points[:,2] < R/2) & (plane_points[:,2] > -R/2)
    plane_points = plane_points[mask]

    # Add noise to the plane
    noise_range = noise
    noise = np.random.randint(-noise_range, noise_range, [plane_points.shape[0], 3]) * EPSILON
    noisy_points = plane_points + noise

    # Add additional random noisy points (points far from the plane)
    num_noisy_points = int((num_points**2) * 1)
    random_points = np.random.randint(low=-num_points/2, high=num_points/ 2, size=(num_noisy_points, 3)) * EPSILON

    # Combine the noisy plane points with the random noisy points
    all_points = np.vstack((noisy_points, random_points))
    
    return all_points

def get_plane(H,W,INTRINSICS):
    depth = np.zeros((H,W))

    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
    Z = np.ones_like(x)
    x_3d = (x - cx) * Z / fx
    y_3d = (y - cy) * Z / fy
    POINTS = np.vstack((x_3d, y_3d, Z)).T
    direction_vector = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    direction_vector = direction_vector.reshape(H,W,3)

    distance = -5
    normal = np.array([0,0,1])
    normal = normal / np.linalg.norm(normal)

    for i in range(H):
        for j in range(W//2):
            depth[i,j] = -distance/(np.dot(direction_vector[i,j], normal)+1e-7)*direction_vector[i,j,2]
    
    distance = 2
    normal = np.array([1,0,-1])
    normal = normal / np.linalg.norm(normal)

    for i in range(H):
        for j in range(W//2, W):
            depth[i,j] = -distance/(np.dot(direction_vector[i,j], normal)+1e-7)*direction_vector[i,j,2]
    
    return depth

