import cv2
import numpy as np
import os
import glob
import multiprocessing
from joblib import Parallel, delayed
from HistEqualize_Fn import AdaptiveHist

"""
This main function is for histogram equalization of images lie in a folder.
This file will call another file while distributing the process on different CPUs.
"""

num_cores = multiprocessing.cpu_count()
print('Total number of cores', num_cores)

this_path = os.path.dirname(os.path.abspath(__file__))
src_folder_path = 'Path/to/Source'
dst_folder_path = 'Path/to/Output'


Parallel(n_jobs=num_cores)(
    delayed(AdaptiveHist)(f, dst_folder_path) for f in glob.glob(os.path.join(src_folder_path, "*.jpg")))
