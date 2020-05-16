import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time
import functools
import glob

def rescale(img_process, scale_percent=20):
    width = int(img_process.shape[1] * scale_percent / 100)
    height = int(img_process.shape[0] * scale_percent / 100)
    dim = (width, height)
    invdim = np.array([img_process.shape[1]/width, img_process.shape[0]/height])
    
    res = cv2.resize(img_process,dim)
    return (res, invdim)


def drawimg(img_src, v, future=None):
    draw_img = img_src.copy()
    for c in v:
        cv2.drawContours(draw_img, [c], 0,[255,0,0], 3)
    if future is not None:
        cv2.drawContours(draw_img, [future], 0,[0,0,255], 3)
    cv2.imshow("SRC",draw_img)
    draw_img = img_src.copy()
    for c in v:
        cv2.drawContours(draw_img, [c], 0,[255,0,0], 3)
    cv2.imshow("GREY",draw_img)


def grow_x(v, s = 5, idx = range(8)):
    ret = v.copy() 
    if 0 in idx:
        ret[0] += [-s,0]
    if 2 in idx:
        ret[2] += [s,0] 
    if 3 in idx:
        ret[3] += [s,0]
    if 4 in idx:
        ret[4] += [s,0]
    if 6 in idx:
        ret[6] += [-s,0]
    if 7 in idx:
        ret[7] += [-s,0]
    return ret   


def grow_y(v, s = 5, idx = range(8)):
    ret = v.copy() 
    if 0 in idx:
        ret[0] += [0,-s]
    if 1 in idx:
        ret[1] += [0,-s]
    if 2 in idx:
        ret[2] += [0,-s] 
    if 4 in idx:
        ret[4] += [0,s]
    if 5 in idx:
        ret[5] += [0,s]
    if 6 in idx:
        ret[6] += [0,s]
    return ret   


def grow(v, s, idx = range(8)):
    ret = v.copy() 
    if 0 in idx:
        ret[0] += [-s,-s]
    if 1 in idx:
        ret[1] += [0,-s]
    if 2 in idx:
        ret[2] += [s,-s] 
    if 3 in idx:
        ret[3] += [s,0]
    if 4 in idx:
        ret[4] += [s,s]
    if 5 in idx:
        ret[5] += [0,s]
    if 6 in idx:
        ret[6] += [-s,s]
    if 7 in idx:
        ret[7] += [-s,0]
    return ret   


def get_pixel(img_grey, countour):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(img_grey)
    cv2.drawContours(cimg, [countour], 0, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array 
    pts = np.where(cimg == 255)
    return img_grey[pts[0], pts[1]]



growdirs = [[0,1,2],[2,3,4],[4,5,6],[6,7,0]]

step = 10

def process_image(img_file):

    img_src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img_src, invdim = rescale(img_src, scale_percent=50) # pylint: disable=unused-variable
    img_grey = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    thres, _ = cv2.threshold(img_grey, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_grey = cv2.threshold(img_grey, thres-1, 255, cv2.THRESH_BINARY)
    org = np.array([int(img_grey.shape[0]/2),int(img_grey.shape[1]/2)])
    orgarray = np.array([org,org,org,org,org,org,org,org])
    # print(org)
    # print(grow(orgarray,5))
    # print(img_grey.shape)

    start = grow(orgarray,int(img_grey.shape[0]/4))
    c = np.array([start.copy(),start.copy(),start.copy(),start.copy()])


    def makefuture_rect(v,fn,idx):
        a = v[idx].copy()
        b = fn(v)
        b = b[idx]
        b = np.flip(b,0)
        # res = np.append(a,[[int((a[2][0]+b[0][0])/2),int((a[2][1]+b[0][1])/2)]],axis=0)
        res = np.append(a,b,axis=0)
        # res = np.append(res,[[int((a[0][0]+b[2][0])/2),int((a[0][1]+b[2][1])/2)]],axis=0)
        return res

    for idx, v in enumerate(c):
        # if idx != 3:
        #     continue
        p = None
        avg = 255
        while step <= v.min() and v.max() <=  img_src.shape[1]:
            future = None
            if idx == 0 or idx == 2:
                v = grow(v,step ,idx=growdirs[idx])
                future = makefuture_rect(v, functools.partial(grow_y, s=step*3, idx=growdirs[idx]), growdirs[idx])
            else:
                v = grow(v,step ,idx=growdirs[idx])
                future = makefuture_rect(v, functools.partial(grow_x, s=step*3, idx=growdirs[idx]), growdirs[idx])
            c[idx] = v
            # print(v)
            drawimg(img_src, c, future)
            # cv2.waitKey(200)
            p = get_pixel(img_grey,future)
            avg = np.average(p)
            print(avg)
            cv2.waitKey(100)
            if avg < ( thres ):
                break


    cv2.waitKey(1000)
    cv2.destroyAllWindows()


g = []
g.append("tweetimgs/*_kktweet.jpg")
# g.append("tweetimgs/tweet_001*.jpg")
# g.append("tweetimgs/tweet_002*.jpg")
# g.append("tweetimgs/tweet_003*.jpg")
# g.append("tweetimgs/tweet_004*.jpg")
# g.append("tweetimgs/tweet_005*.jpg")
# g.append("tweetimgs/tweet_01[2-3]*.jpg")

for p in glob.glob(g[0]):
    #img = cv2.imread(p,0)
    process_image(p)
