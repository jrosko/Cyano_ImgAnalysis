import matplotlib.pyplot as plt
import numpy as np
import pims
import cv2

def pre_process(input_frame,  thr_val, denoise =  True, op_ker_d = 2):
    """
    Denoise and threshold a frame.
    Uses a non-local means filter to denoise (pass denoise = False to not).
    Applies a binary threshold of thr_val.
    Performs morphological opening to remove any leftover noise patches.
    op_ker_d is the kernel dimension for opening.
    """
    if denoise == True:
        input_frame = cv2.fastNlMeansDenoising(input_frame,templateWindowSize=7, searchWindowSize=21, h=0.7)

    ret_bin, thr_bin = cv2.threshold(input_frame, thr_val, 255, cv2.THRESH_BINARY)

    kernel = np.ones((op_ker_d,op_ker_d), np.uint8)
    opening = cv2.morphologyEx(thr_bin, cv2.MORPH_OPEN,kernel, iterations = 1)

    return opening
