import scipy.io as sio
import cv2
import numpy as np
import os
"""
This file use OpenCV for local histogram equalization of an image.
"""

this_path = os.path.dirname(os.path.abspath(__file__))      # __file__ is exactly the current file that we are working on

def AdaptiveHist(img_name, dst_folder_path):

    # Get the grayscale image
    img = cv2.imread(img_name, 0)

    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img = clahe.apply(img)

    # Save with the same name in another folder.
    base_name = os.path.basename(img_name)
    cv2.imwrite(os.path.join(dst_folder_path, base_name), equalized_img)
    print(os.path.join(dst_folder_path, base_name))