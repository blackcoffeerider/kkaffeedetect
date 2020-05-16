import cv2
import numpy as np
from scipy.spatial import distance as dist
import sys
import os
import time
import glob
import libkaffeedetect as lkd

font = cv2.FONT_HERSHEY_PLAIN

def get_contours(img_process, lower=150, upper=255):
    _, threshold = cv2.threshold(img_process, lower, upper, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    
    return contours

def get_approxs(img_grey, minsize, theta=0.01, minvert=4, maxvert=20, scale_percent = 20, draw=False, counter=0):
    img_blurred_grey = lkd.blur_image(img_grey,5)
    img_blurred_grey_scaled, invdim = lkd.rescale(img_blurred_grey, scale_percent=scale_percent)
    if draw:
        cv2.imshow("Grey", img_blurred_grey_scaled)    
    #contours = get_contours(img_blurred_grey_scaled, 130, 200) # 140-255 || 170-255
    contours = get_contours(img_blurred_grey_scaled, 0, 255) # 140-255 || 170-255
    retapprox = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, theta*cv2.arcLength(cnt, True), True)
        if minvert <= len(approx) and len(approx) <= maxvert:
            approx = np.multiply(approx, invdim, out=approx, casting='unsafe', dtype=np.int32) 
            csize = cv2.contourArea(approx)
            if csize > minsize:
                retapprox.append(approx)
    return retapprox

def filter_max_approx(approxs):
    approxsize = [ (cv2.contourArea(l), l) for l in approxs ]
    if approxsize:
        approxsize.sort(key=lambda x: x[0], reverse=True)
        return approxsize[0][1]
    return None


def make_clockwise(pts):
    # find topleft 
    tl = pts[0] # pylint: disable=unused-variable
    tlm = pts[0][0]+pts[0][1]
    tlidx = 0
    for cidx, pt in enumerate(pts):
        ctlm = pt[0]+pt[1]
        if ctlm < tlm:
            tl = pt
            tlm = ctlm
            tlidx = cidx

    #make top left first element
    split = np.split(pts,[tlidx])
    split = np.flip(split,0)
    conc = np.concatenate(split,axis=0)
    res = conc

    # check vector orientation of second and last point to normal y by angle
    normyor = lkd.get_normy_orientation(np.take(res,-1,axis=0),res[1]) 
    print(normyor)
    if ( 90.0 < normyor and normyor < 180.0 ) or ( 270.0 < normyor and normyor < 360.0 ):
        
        # reverse chains with orientation from top-left to  bottom-right or bottom-left to top-right
        first = np.copy(conc[0])
        nofirst = np.delete(conc, 0, axis=0)
        flipped = np.flip(nofirst, 0)
        res = np.insert(flipped, 0, first, axis=0)
    # else if ( 0.0 < normyor and normyor < 45.0 )

    return res # fix! 
    

def process_single_file(img_filename, theta=0.01,minvert=4, maxvert=20,scale_percent=20,show=True):
    img_src = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
    img_prefilt = lkd.prefilter_colors(img_src)
    # img_prefilt = img_process.copy() 
    img_grey = cv2.cvtColor(img_prefilt, cv2.COLOR_BGR2GRAY)
    ret = process_single(img_src, img_grey, imgfilename=img_filename, theta=theta, minvert=minvert, maxvert=maxvert, scale_percent=scale_percent, draw=show)
    if show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return ret


def process_single(img_src, img_grey, imgfilename="unknowfile", theta=0.01,minvert=4, maxvert=20,scale_percent=20,draw=False):
    quality = 0

    rects = []

    minsize = int(img_src.shape[1]*img_src.shape[0]*.25)

    approxs = get_approxs(img_grey,minsize,theta=theta,minvert=minvert, maxvert=maxvert,scale_percent=scale_percent,draw=draw)

    approx = filter_max_approx(approxs) # find biggest approximations
    if approx is not None:   
        shape = np.copy(approx)
        shape = np.reshape(shape,(-1,2))
        shape = make_clockwise(shape)
        shape = np.reshape(shape,(-1,2))
        shape = np.insert(shape,0,shape[-1]) # append last to front
        shape = np.reshape(shape,(-1,2))
        shape = np.append(shape, shape[1]) # append original front to back
        shape = np.reshape(shape,(-1,2))

        x,y,w,h = cv2.boundingRect(approx) # pylint: disable=unused-variable
        rect = cv2.minAreaRect(approx) #  optimized box
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_src, [box], 0,[255,0,0], 3)

        cv2.drawContours(img_src, [shape], 0, [0,0, 255], 5) # original detection

        quality = cv2.contourArea(shape) / cv2.contourArea(box)
        cv2.putText(img_src, "Quality: {:0.2f}".format(quality) , (30,30), font, 2, [0,0,0], thickness=8)
        cv2.putText(img_src, "Quality: {:0.2f}".format(quality) , (30,30), font, 2, [255,0,0], thickness=2)
                

        for x in range(1, len(shape)-1):
            v = np.take(shape,(x-1,x,x+1),axis=0)
            # print(x)
            ang = lkd.get_angle_3vec(v)
            if ( 85 <= ang <= 95 ):
                print(ang)
                cv2.putText(img_src, "{0:d}: ({1:d},{2:d}) @{3:3.2f}".format(x,v[1][0],v[1][1],ang) , (v[1][0],v[1][1]), font, 2, [0,0,0], thickness=8)
                cv2.putText(img_src, "{0:d}: ({1:d},{2:d}) @{3:3.2f}".format(x,v[1][0],v[1][1],ang) , (v[1][0],v[1][1]), font, 2, [0,255,0], thickness=2)
        
        rects.append(box)

    scale_percent_disp = 66
    width = int(img_src.shape[1] * scale_percent_disp / 100)
    height = int(img_src.shape[0] * scale_percent_disp / 100)
    dim = (width, height)
    if draw:
        img_src_scale = cv2.resize(img_src,dim)
        cv2.imshow(imgfilename + " Shapes", img_src_scale)
        #cv2.imshow("Threshold", threshold)
        
    return rects, img_src.shape[1], img_src.shape[0],  quality

    


if __name__ == "__main__":
    # test_get_normy_orientation()


    if len(sys.argv) > 1:
        print(sys.argv[1])
        p = os.path.abspath(sys.argv[1])
        print(p)
        if os.path.exists(p):
            if os.path.isfile(p):
                imgfile = p


        print(imgfile)

        process_single_file(imgfile,scale_percent=20)
    
    else:
        g = []
        g.append("01_downloaded_tweetimgs/2019_02*.jpg")

        for p in glob.glob(g[0]):
            _, _, _, q2 =  _, q = process_single_file(p,theta=0.008,minvert=4,maxvert=50,scale_percent=20)
            if q < .9:
                _, _, _, q2 = process_single_file(p,theta=0.010,minvert=4,maxvert=50,scale_percent=25)
            if q2 < q:
                process_single_file(p,theta=0.015,minvert=4,maxvert=50,scale_percent=25)



