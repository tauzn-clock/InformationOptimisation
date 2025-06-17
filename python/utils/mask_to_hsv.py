import numpy as np

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