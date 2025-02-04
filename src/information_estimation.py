import numpy as np
from tqdm import tqdm

def default_ransac(POINTS, R, EPSILON, SIGMA, CONFIDENCE=0.99, INLIER_THRESHOLD=0.01, MAX_PLANE=1, valid_mask=None, verbose=False):
    assert(POINTS.shape[1] == 3)
    assert(MAX_PLANE > 0), "MAX_PLANE must be greater than 0"
    N = POINTS.shape[0]
    if valid_mask is not None: TOTAL_NO_PTS = valid_mask.sum()
    else: TOTAL_NO_PTS = POINTS.shape[0]
    SPACE_STATES = np.log(R/EPSILON)
    PER_POINT_INFO = 0.5 * np.log(2*np.pi) + np.log(SIGMA/EPSILON) - SPACE_STATES

    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - (INLIER_THRESHOLD)**3))

    information = np.full(MAX_PLANE+1, np.inf, dtype=float)
    mask = np.zeros(N, dtype=int)
    plane = np.zeros((MAX_PLANE+1, 4), dtype=float)
    availability_mask = np.ones(N, dtype=bool)
    if valid_mask is not None:
        availability_mask = valid_mask
    
    # O
    information[0] = TOTAL_NO_PTS * 3 * SPACE_STATES

    # nP + 0
    for plane_cnt in range(1, MAX_PLANE+1):
        #BEST_INLIERS_CNT = 0
        BEST_INLIERS_MASK = np.zeros(N, dtype=bool)
        BEST_ERROR = 0
        BEST_PLANE = np.zeros(4, dtype=float)

        available_index = np.linspace(0, N-1, N, dtype=int)
        available_index = np.where(availability_mask)[0]

        information[plane_cnt] =  information[plane_cnt-1]
        information[plane_cnt] -= TOTAL_NO_PTS * np.log(plane_cnt) # Remove previous mask 
        information[plane_cnt] += TOTAL_NO_PTS * np.log(plane_cnt+1) # New Mask that classify points
        information[plane_cnt] += 3 * SPACE_STATES # New Plane

        if (availability_mask).sum() < 3:
            break

        for _ in tqdm(range(ITERATION), disable=not verbose):
            # Get 3 random points
            idx = np.random.choice(available_index, 3, replace=False)

            # Get the normal vector and distance
            A = POINTS[idx[0]]
            B = POINTS[idx[1]]
            C = POINTS[idx[2]]

            AB = B - A
            AC = C - A
            normal = np.cross(AB, AC)
            normal = normal / (np.linalg.norm(normal) + 1e-7)
            distance = -np.dot(normal, A)

            # Count the number of inliers
            error = np.abs(np.dot(POINTS, normal.T)+distance) 
            error = 0.5 * error**2 / SIGMA**2 + PER_POINT_INFO
            trial_mask = error < 0
            trial_mask = trial_mask & availability_mask
            trial_error = error[trial_mask].sum()

            if BEST_ERROR > trial_error:
                #BEST_INLIERS_CNT = trial_cnt
                BEST_INLIERS_MASK = trial_mask
                BEST_PLANE = np.concatenate((normal, [distance]))
                BEST_ERROR = trial_error
        
        information[plane_cnt] += BEST_ERROR
        mask[BEST_INLIERS_MASK] = plane_cnt
        plane[plane_cnt] = BEST_PLANE

        availability_mask[BEST_INLIERS_MASK] = 0
    
    return information, mask, plane

def plane_ransac(DEPTH, INTRINSICS, R, EPSILON, SIGMA, CONFIDENCE=0.99, INLIER_THRESHOLD=0.01, MAX_PLANE=1, valid_mask=None, verbose=False):

    def find_inliers(normal, distance):
        error = ((-distance/(np.dot(direction_vector, normal.T)+1e-7))*direction_vector[:,2] - Z) ** 2
        error = error / TWO_SIGMA_SQUARE + PER_POINT_INFO
        trial_mask = error < 0
        trial_mask = trial_mask & availability_mask
        trial_error = error[trial_mask].sum()
        return trial_mask, trial_error

    assert(MAX_PLANE > 0), "MAX_PLANE must be greater than 0"
    H, W = DEPTH.shape
    N = H * W
    if valid_mask is not None: TOTAL_NO_PTS = valid_mask.sum()
    else: TOTAL_NO_PTS = H * W

    # Direction vector, all projection rays go through origin
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.flatten()
    y = y.flatten()
    fx, fy, cx, cy = INTRINSICS[0], INTRINSICS[5], INTRINSICS[2], INTRINSICS[6]
    Z = DEPTH.flatten()
    x_3d = (x - cx) * Z / fx
    y_3d = (y - cy) * Z / fy
    POINTS = np.vstack((x_3d, y_3d, Z)).T

    direction_vector = POINTS / (np.linalg.norm(POINTS, axis=1)[:, None]+1e-7)

    SPACE_STATES = np.log(R/EPSILON)
    PER_POINT_INFO = np.log(SIGMA/(EPSILON+1e-7) + 1e-7)
    PER_POINT_INFO += 0.5 * np.log(2*np.pi) - SPACE_STATES
    TWO_SIGMA_SQUARE = 2 * (SIGMA**2 + 1e-7)

    ITERATION = int(np.log(1 - CONFIDENCE) / np.log(1 - INLIER_THRESHOLD**3))

    information = np.full(MAX_PLANE+1, np.inf, dtype=float)
    mask = np.zeros(N, dtype=int)
    plane = np.zeros((MAX_PLANE+1, 4), dtype=float)
    availability_mask = np.ones(N, dtype=bool)
    if valid_mask is not None:
        availability_mask = valid_mask
    
    # O
    information[0] = TOTAL_NO_PTS * SPACE_STATES

    # nP + 0
    for plane_cnt in range(1, MAX_PLANE+1):
        #BEST_INLIERS_CNT = 0
        BEST_INLIERS_MASK = np.zeros(N, dtype=bool)
        BEST_ERROR = 0
        BEST_PLANE = np.zeros(4, dtype=float)

        available_index = np.linspace(0, N-1, N, dtype=int)
        available_index = np.where(availability_mask)[0]

        information[plane_cnt] =  information[plane_cnt-1]
        information[plane_cnt] -= TOTAL_NO_PTS * np.log(plane_cnt) # Remove previous mask 
        information[plane_cnt] += TOTAL_NO_PTS * np.log(plane_cnt+1) # New Mask that classify points
        information[plane_cnt] += 3 * SPACE_STATES # New Plane

        if (availability_mask).sum() < 3:
            break

        for _ in tqdm(range(ITERATION), disable=not verbose):
            # Get 3 random points
            idx = np.random.choice(available_index, 3, replace=False)

            # Get the normal vector and distance
            A = POINTS[idx[0]]
            B = POINTS[idx[1]]
            C = POINTS[idx[2]]

            AB = B - A
            AC = C - A
            normal = np.cross(AB, AC)
            normal = normal / (np.linalg.norm(normal) + 1e-7)
            distance = -np.dot(normal, A)

            # Count the number of inliers
            trial_mask, trial_error = find_inliers(normal, distance)

            if  trial_error < BEST_ERROR:
                
                
                #SVD to find normal and distance
                inliers = POINTS[trial_mask]
                normal, distance = fit_plane(inliers)
                trial_mask, trial_error = find_inliers(normal, distance)
                

                BEST_INLIERS_MASK = trial_mask
                BEST_PLANE = np.concatenate((normal, [distance]))
                BEST_ERROR = trial_error
        
        information[plane_cnt] += BEST_ERROR
        mask[BEST_INLIERS_MASK] = plane_cnt
        plane[plane_cnt] = BEST_PLANE

        availability_mask[BEST_INLIERS_MASK] = 0
    
    return information, mask, plane

def fit_plane(points):
    # Compute the centroid (mean) of the points
    centroid = np.mean(points, axis=0)
    
    # Shift the points to the centroid
    shifted_points = points - centroid
    
    # Compute the covariance matrix
    cov_matrix = np.dot(shifted_points.T, shifted_points) / len(points)
    
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # The normal vector is the eigenvector corresponding to the smallest eigenvalue
    normal_vector = eigenvectors[:, np.argmin(eigenvalues)]
    
    # The equation of the plane is normal_vector . (x - centroid) = 0
    D = -np.dot(normal_vector, centroid)
    
    return normal_vector, D
