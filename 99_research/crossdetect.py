import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time
import functools
import glob
import math
import os
import sys
sys.path.insert(1, os.path.abspath('.'))
# import shrinkdetect

import libkaffeedetect as lkd


def process_image(img_file, draw=False):

    img_src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    w = img_src.shape[1]
    h = img_src.shape[0]
    img_src, invdim = lkd.rescale(img_src, scale_percent=50)
    img_pref = lkd.prefilter_colors(img_src)
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY)
    thres, _ = cv2.threshold(img_grey, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_grey = cv2.threshold(img_grey, thres-1, 255, cv2.THRESH_BINARY)

    
    ret = 0


    cv2.imshow("GREY",img_grey)

    ret, _, _ = lkd.get_borders_from_grey_smear(img_src,img_grey, draw=draw)

    if ret is not None:
        if draw:
            minr = lkd.get_scaled_rect(ret, percentage=-4) 
            maxr = lkd.get_scaled_rect(ret, percentage=7) 
            
            lkd.drawimg(img_src, [ret], red=maxr, green=minr)
            cv2.waitKey(0)
        ret = np.multiply(ret, invdim, out=ret, casting='unsafe', dtype=np.int32)
    # print(img_file)
    return ret, w, h


def test():
    # print(makecross([20,20], 10, [800, 600]))
    # print(makecross([5,40], 10, [800, 600]))
    
    g = []
    g.append("C:\\tmp\\tweets\\2019_03_1*_tweet.jpg")
    g.append("C:\\tmp\\tweets\\2019_0[3-5]_*_tweet.jpg")
    draw = True
    if 1==1:
        for p in sorted(glob.glob(g[0])):
            #img = cv2.imread(p,0)
            print(">>>"+p+"============================")
            d, w, h = process_image(p,draw=draw)        
            print(d)
            print("<<<"+p+"============================")
    else:
        for p in ["C:\\tmp\\tweets\\2019_02_11_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_02_09_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_02_15_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_02_18_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_02_19_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_02_20_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_03_16_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_04_21_tweet.jpg",
                    "C:\\tmp\\tweets\\2019_03_29_tweet.jpg"]:
            #img = cv2.imread(p,0)
            print(">>>"+p+"============================")
            d, w, h = process_image(p,draw=draw)        
            print(d)
            print("<<<"+p+"============================")
    if draw:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test()