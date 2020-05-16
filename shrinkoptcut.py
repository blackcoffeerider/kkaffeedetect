import cv2 
import numpy as np
from matplotlib import pyplot as plt
import time
import functools
import glob
import math

import libkaffeedetect as lkd


def get_grow_triangangle_along_axis_ref(main_a,axpoint_c,ref_b,minpixels,limits=None):
    """ Create a new triangle of which one axis is enlarged.
        Input is a triangle the main (A) point and
        the reference point (B) which stay fixed.
        Now we shift the third point C along the vector of AC
        minpixels gives us the sizing of that normalized vector.
        Limits are two opposite corners (upperleft, lowerright) of a box which should not be exeeded
    """
    axvec_ac = lkd.get_norm_vec(main_a - axpoint_c)*minpixels
    shifted_c = main_a
    if limits is None or \
       ( limits.shape == (2,2)  and \
         ( limits[0][0] <= shifted_c[0] <= limits[1][0] and \
           limits[0][1] <= shifted_c[1] <= limits[1][1] ) ):   
        shifted_c = main_a+axvec_ac
    ret = np.array([shifted_c, main_a, ref_b])
    return ret


def get_optimized_rect(img_src, img_grey, rect, draw=False, drawsteps=False):
    """ We optimize a given quadrangle towards having a maximum white encompassing area
        by trying to shift its corners around.
        Shifting happens on any given corner A
        A     C
        
        B
        where ABC form a triangle made from parts of the original quadrangle
        A is shifted inwards and outwards by a few pixels along the vectors AC and AB
        Giving a point S_ being "outside" and S_' being "within" the line between A and the counter points
          Sy
        Sx A Sx'  C
          Sy'
           B
        if the area Sx A B contains more then a given threshold of white pixels
        A is moved to Sx
        if Sx' A B is mostly comprised of black pixels A is Moved to Sx'
        Same is repeated for Sy and Sy'
        This process is not repeated on one corner at once but all corners are probed in on direction clockwise
        and then counterclockwise to avoid "overoptimizing" on one corener A to much without probing the counterparts B and C
    """
    maxitter = 1000
    ret = rect
    retprev = np.zeros((4,2))
    retcurr = ret.copy()
    diff = np.average(retcurr - retprev)
    if drawsteps:
        print("Optimizing orientation")
    for growsize in [12,8,5,3,2]: # magic numbers that work well for offset sizes of triangles
        if drawsteps:
            print("Optimizing size: {}".format(growsize), end='')
        retprev = np.zeros((4,2))
        retcurr = ret.copy()
        diff = np.average(retcurr - retprev)
        curriter = 0
        while (diff > 1 and curriter < maxitter): 
            curriter += 1
            if drawsteps:
                print(".", end='')
            retprev = retcurr.copy()
            def growswap(growsize=1, swapoverthres=True):
                for idx, main in enumerate(retcurr):
                    rect = get_grow_triangangle_along_axis_ref(main, 
                                                            retcurr[((idx+1) % 4)], 
                                                            retcurr[idx-1],
                                                            growsize, 
                                                            np.array([[0,0],
                                                                      [img_grey.shape[1],img_grey.shape[0]]
                                                                     ])).astype(np.int32)
                    if drawsteps:
                        lkd.drawimg(img_src, [retcurr], red=rect )
                        cv2.waitKey(100)
                    if swapoverthres:
                        if np.average(lkd.get_pixel(img_grey,rect)) > 255*0.3:
                            retcurr[idx] = rect[0]
                    else:
                        if np.average(lkd.get_pixel(img_grey,rect)) < 255*0.3:
                            retcurr[idx] = rect[0]
                    if drawsteps:
                        lkd.drawimg(img_src, [retcurr] )
                        cv2.waitKey(100)
            #clockwise grow
            growswap(growsize, True)
            #clockwise shrink
            growswap(growsize*-1,False)
            #counterclockwise grow
            retcurr = np.flip(retcurr,0)
            growswap(growsize)
            #counterclockwise shrink
            growswap(growsize*-1,False)
            retcurr = np.flip(retcurr,0)
            diff = math.sqrt(np.max(retcurr - retprev)*np.max(retcurr - retprev))
        if drawsteps:
                print("")
        ret = retcurr
    if drawsteps:
        lkd.drawimg(img_src, [retcurr] )
        cv2.waitKey(100)

    return ret


def thresbin(a,b):
    if a > b:
        return 1
    return 0

thresbinv = np.vectorize(thresbin)


def get_borderboxes(img_src):
    """Returns boxes of upper and lower objects besides the main white middle area
       this removes paintboxes and papertissues from the given sample pictures.

       For concept illustration please see plotgeyvert.py
    """
    img_pref = lkd.prefilter_colors(img_src) # ditch greens and fingers as usual :D
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY) # standard grayscale works best
    vhist_grey = np.average(img_grey,axis=1) # average every horizontal line
    glimit_grey = np.min(vhist_grey) + np.average(vhist_grey)*.25 # we cant consider zero as bottom line but need to check the minumum
                                                                  # and need some offset to the "bottom" so by 
                                                                  # looking at the sampled data 25% of the average gave good results in most cases
    bvhist_grey = thresbinv(vhist_grey,glimit_grey)               # we cant use any grey scale value but need a binary representation tissue/paintbox/painting = 1 "background" = 0

    bvhist_grey = np.insert(bvhist_grey,len(bvhist_grey),[0]) # insert dummy zero at the beginning

    ranges = []  # collection of intervals
    prev = 0     # previouse signal value zero or one
    cvec = None  # current interval
    maxvlen = 0  # longest interval

    # go top down over the values - every continuous block of ones will be returned as a 'range' of the indexes (picture lines/rows)
    for idx, l in enumerate(bvhist_grey):
        if prev != l:                       # change from 0 -> 1 or 1 -> 0
            if cvec is None:                    # 0 -> 1
                cvec = [idx, idx, 0]                # start new box with start current index
            else:                               # 1 -> 0
                cvec[1] = idx - 1                   # close open box with end of previous index
                cvec[2] = cvec[1] - cvec[0] + 1     # calculate length so we can remove the longuest (which should be the drawing in the picture)
                ranges.append(cvec)                 # append to range collection
                if maxvlen < cvec[2]:               # if this is the longes block we have seen?
                    maxvlen = cvec[2]                  # remember the length                   # pylint: disable=unsubscriptable-object 
                cvec = None # reset
        prev = l

    ret = [ [ # return a rectangle shape with the width of the src image (clockwise orientation)
                [0,f[0]], # upper left corner                   
                [img_grey.shape[1],f[0]], # upper right
                [img_grey.shape[1],f[1]], # lower right
                [0,f[1]] # lower left
            ] 
            for f in ranges if f[2] < maxvlen  # for every range block that is smaller than the longest (which should be the drawing in the picture) 
          ]

    retmaxa = [ [ # return a rectangle shape with the width of the src image (clockwise orientation)
            [0,f[0]], # upper left corner                   
            [img_grey.shape[1],f[0]], # upper right
            [img_grey.shape[1],f[1]], # lower right
            [0,f[1]] # lower left
        ] 
        for f in ranges if f[2] == maxvlen  # for every range block that is smaller than the longest (which should be the drawing in the picture) 
        ]
    
    retmax = None
    if len(retmaxa) > 0:
        retmax = np.array(retmaxa[0],dtype=np.int32) 

    return np.array(ret,dtype=np.int32), retmax


def process_image(img_file, scale_percent=50, draw=False, drawsteps=False):
    ret = None
    img_src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    w = img_src.shape[1]
    h = img_src.shape[0]
    img_src, invdim = lkd.rescale(img_src, scale_percent=scale_percent)
    img_pref = lkd.prefilter_colors(img_src)
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY)
    thres, _ = cv2.threshold(img_grey, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_grey = cv2.threshold(img_grey, thres-1, 255, cv2.THRESH_BINARY)

    bbxes, mainbox = get_borderboxes(img_src) # get paintboxes and paper-tissues if they exist
    if bbxes is not None:
        cv2.drawContours(img_grey, bbxes, 0, color=0, thickness=-1) # blacken them
    

    if mainbox is not None: # alternative: blacken everyting
        img_grey, _ = lkd.blacken_outside_shape(img_grey, mainbox)


    if draw:
        lkd.drawimg(img_grey, [], label="GREYBLACKED")

    sdraw = draw

    shrinkrect, _, _ = lkd.process_image_shrink_boxdetect(img_src, thres, img_grey, draw=draw, drawsteps=drawsteps, scale_percent=100)

    ret = shrinkrect

    lim = max(img_grey.shape)
    if ret.min() < 0 or ret.max() > lim:
        ret = np.array([[0,0],[img_grey.shape[1],0],[img_grey.shape[1],img_grey.shape[0]],[0,img_grey.shape[0]]],dtype=np.int32)
    else:
        # minr = lkd.get_scaled_rect(ret, percentage=-7) 
        maxr = lkd.get_scaled_rect(ret, percentage=7) # 7 magic number by try and error

        # blacken out detected rectangle outer parts wit an offset of a certain percentage (see above)
        outerinv = np.zeros(img_grey.shape,dtype=np.uint8)
        outerinv = cv2.drawContours(outerinv, [maxr], 0,[255,255,255], thickness=cv2.FILLED) 
        img_grey = cv2.bitwise_and(img_grey,outerinv)

        ret = get_optimized_rect(img_src, img_grey, ret, draw=draw, drawsteps=drawsteps)

    draw = sdraw
    
    if draw:
        lkd.drawimg(img_src, [ret], green=ret )
    # print(img_file)
    ret = np.multiply(ret, invdim, out=ret, casting='unsafe', dtype=np.int32)
    return ret, w, h
