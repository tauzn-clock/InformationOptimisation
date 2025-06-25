import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def mask_to_hsv(mask):
    def hsv_colors(n):
        # Generate n distinct colors in HSV color space and convert to RGB
        hsv = np.zeros((n, 3))
        hsv[:, 0] = np.linspace(0, 1, n+1)[:-1]  # Hue values
        hsv[:, 1] = 1
        hsv[:, 2] = 1
        def hsv_to_rgb(h, s, v):
            """
            Convert HSV to RGB.

            :param h: Hue (0 to 360)
            :param s: Saturation (0 to 1)
            :param v: Value (0 to 1)
            :return: A tuple (r, g, b) representing the RGB color.
            """
            h = h   # Normalize hue to [0, 1]
            c = v * s  # Chroma
            x = c * (1 - abs((h * 6) % 2 - 1))  # Temporary value
            m = v - c  # Match value

            if 0 <= h < 1/6:
                r, g, b = c, x, 0
            elif 1/6 <= h < 2/6:
                r, g, b = x, c, 0
            elif 2/6 <= h < 3/6:
                r, g, b = 0, c, x
            elif 3/6 <= h < 4/6:
                r, g, b = 0, x, c
            elif 4/6 <= h < 5/6:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            # Adjust to match value
            r = (r + m) 
            g = (g + m) 
            b = (b + m)

            return r, g, b
        
        # Convert HSV to RGB
        rgb = np.zeros((n, 3))

        for i in range(n):
            r, g, b = hsv_to_rgb(hsv[i, 0], hsv[i, 1], hsv[i, 2])
            rgb[i, 0] = r
            rgb[i, 1] = g
            rgb[i, 2] = b

        return (rgb * 255).astype(np.uint8)

    rgb = hsv_colors(mask.max())

    img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] > 0:
                img[i, j] = rgb[mask[i, j]-1]
            else:
                img[i, j] = [0, 0, 0]

    return img

def mask_over_img(img, mask, path):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(mask_to_hsv(mask), alpha=0.5)
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)

def img_over_pcd(points, img, filepath=None):

    # Visualize the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    color = img.reshape(-1, 3) / 255.0  # Normalize color values to [0, 1]

    point_cloud.colors = o3d.utility.Vector3dVector(color)

    tf = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    point_cloud.transform(tf)

    if filepath is not None:
        # Visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # Add the point cloud to the visualizer
        vis.add_geometry(point_cloud)

        opt = vis.get_render_option()
        opt.point_size = 2

        view_control = vis.get_view_control()
        view_control.set_zoom(0.6) 
        view_control.rotate(0, -100)

        img = np.array(vis.capture_screen_float_buffer(True))
        left = img.shape[1]
        right = 0
        top = img.shape[0]
        bottom = 0

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.sum(img[i,j]) < 3:
                    left = min(left, j)
                    right = max(right, j)
                    top = min(top, i)
                    bottom = max(bottom, i)

        output = img[top:bottom, left:right]
        
        plt.imsave(filepath, output)
    else:
        return point_cloud