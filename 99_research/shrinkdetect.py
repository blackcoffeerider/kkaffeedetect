import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time
import glob
import math

import libkaffeedetect as lkd



def process_image(img_file, draw=False, drawsteps=False):

    img_src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    w = img_src.shape[1]
    h = img_src.shape[0]
    img_src, invdim = lkd.rescale(img_src, scale_percent=50)
    img_pref = lkd.prefilter_colors(img_src)
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY)
    thres, _ = cv2.threshold(img_grey, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_grey = cv2.threshold(img_grey, thres-1, 255, cv2.THRESH_BINARY)
    org = np.array([int(img_grey.shape[1]/2),int(img_grey.shape[0]/2)])
    ret = lkd.process_image_center(img_src,img_grey,thres,org,draw=False,drawsteps=drawsteps)
    org = np.array([int((ret[0][0]+ret[2][0])/2),int((ret[0][1]+ret[2][1])/2)])
    ret = lkd.process_image_center(img_src,img_grey,thres,org,draw=draw,drawsteps=drawsteps)
    ret = np.multiply(ret, invdim, out=ret, casting='unsafe', dtype=np.int32)
    # print(img_file)
    return ret, w, h

