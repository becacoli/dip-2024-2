import argparse
import numpy as np
import cv2 as cv
import requests

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    
    ### START CODE HERE ###
    response_url = requests.get(url)
    response_url.raise_for_status()
    image_array = np.asarray(bytearray(response_url.content), dtype=np.uint8)

    image = cv.imdecode(image_array, cv.IMREAD_COLOR)  
    cv.imwrite("image.jpg", image)
    ### END CODE HERE ###
    
    return image

load_image_from_url("https://www.kalmbachfeeds.com/cdn/shop/articles/person-holding-two-chinchillas-in-their-arms.jpg?v=1706873615")
