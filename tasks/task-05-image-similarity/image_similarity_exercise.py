# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def mse(i1: np.ndarray, i2: np.ndarray):
    return np.mean((i1 - i2) ** 2)

def psnr(mse, max_pixel=1.0):
    if mse == 0:
        return float('inf')
    return 10 * np.log10((max_pixel ** 2) / mse)

def ssim(i1: np.ndarray, i2: np.ndarray):
    c1 = (0.01)**2
    c2 = (0.03)**2

    mean1 = np.mean(i1)
    mean2 = np.mean(i2)
    var1 = np.var(i1)
    var2 = np.var(i2)
    covariance = np.mean((i1 - mean1) * (i2 - mean2))

    numerator = (2 * mean1 * mean2 + c1) * (2 * covariance + c2)
    denominator = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)

    return numerator / denominator

def npcc(i1: np.ndarray, i2: np.ndarray):
    return np.corrcoef(i1.flatten(), i2.flatten())[0, 1]

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:
    mse_val = mse(i1, i2)
    psnr_val = psnr(mse_val)
    ssim_val = ssim(i1, i2)
    npcc_val = npcc(i1, i2)
    
    return {
        "mse": float(mse_val),
        "psnr": float(psnr_val),
        "ssim": float(ssim_val),
        "npcc": float(npcc_val)
    }

