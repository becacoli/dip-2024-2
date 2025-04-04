# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def translate_image(img: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    h, w = img.shape
    translated = np.zeros_like(img)
    
    dx = min(shift_x, w)
    dy = min(shift_y, h)

    translated[dy:, dx:] = img[:h-dy, :w-dx]
    return translated

def rotated_image(img):
    return np.rot90(img, k=-1)

def stretched_image(img, scale=1.5):
    height, width = img.shape
    new_width = int(width * scale)
    stretched = np.zeros((height, new_width), dtype=img.dtype)

    x_old = np.linspace(0, width-1, new_width)
    x0 = np.floor(x_old).astype(int)
    x1 = np.ceil(x_old).astype(int)
    alpha = x_old - x0

    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)

    stretched = (1 - alpha) * img[:, x0] + alpha * img[:, x1]

    return stretched


def mirrored_image(img):
    return np.fliplr(img)


def distorted_image(img, k: float = 0.0005):
    h, w = img.shape
    y, x = np.indices((h, w))
    cx, cy = w // 2, h // 2

    x_shifted = x - cx
    y_shifted = y - cy

    r_squared = x_shifted**2 + y_shifted**2
    factor = 1 + k * r_squared

    x_distorted = (x_shifted * factor + cx).astype(int)
    y_distorted = (y_shifted * factor + cy).astype(int)

    x_distorted = np.clip(x_distorted, 0, w - 1)
    y_distorted = np.clip(y_distorted, 0, h - 1)

    return img[y_distorted, x_distorted]

def apply_geometric_transformations(img: np.ndarray) -> dict:
    return {
        "translated": translate_image(img, shift_x=50, shift_y=30),
        "rotated": rotated_image(img),
        "stretched": stretched_image(img),
        "mirrored": mirrored_image(img),
        "distorted": distorted_image(img)
    }
    