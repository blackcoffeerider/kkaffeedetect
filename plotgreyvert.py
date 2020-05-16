""""
    Demo program to illustrate the concept behind improoving detection by removing
    paintboxes and paper tissues found in the image.
"""

import sys
import glob
import os
import libkaffeedetect as lkd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import docopt
import pprint


def get_grey_otsu_bw(img_src):
    img_pref = lkd.prefilter_colors(img_src)
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY)
    _, img_bw = cv2.threshold(img_grey, 127, 255, cv2.THRESH_BINARY)
    thres, _ = cv2.threshold(img_grey, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_otsu = cv2.threshold(img_grey, thres-1, 255, cv2.THRESH_OTSU)

    return img_grey, img_otsu, img_bw

def thresbin(a,b):
    if a > b:
        return 1
    return 0

thresbinv = np.vectorize(thresbin)

def main():
    """ {pyname} shows a historgram of the horizontal average pixel value 

    Usage:
        {pyname} <imagefilenameglob>  [--scalepercent=<scalepercent>]
        {pyname} -h
    
    Options:
        -h, --help            Show this help message
        --scalepercent=<scalepercent>  [default: 50]

    """
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    # print(main.__doc__)
    args = docopt.docopt(main.__doc__, help=False)
    pprint.pprint(args)

    sfiles = sorted(glob.glob(args["<imagefilenameglob>"]))

    for num, img_file in enumerate(sfiles):
        plotvhist(img_file,num)
    
        

def rotimg90(img):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 90, 1.0)
    return cv2.warpAffine(img, M, (h, w))



def get_borderboxes_intervals(img_src):
    img_pref = lkd.prefilter_colors(img_src)
    img_grey = cv2.cvtColor(img_pref, cv2.COLOR_BGR2GRAY)
    vhist_grey = np.average(img_grey,axis=1)
    glimit_grey = np.min(vhist_grey) + np.average(vhist_grey)*.25
    bvhist_grey = thresbinv(vhist_grey,glimit_grey)

    # bvhist_grey = np.insert(bvhist_grey,0,[0])
    bvhist_grey = np.insert(bvhist_grey,len(bvhist_grey),[0])
    ranges = []
    prev = 0
    # cvopen = False 
    cvec = None
    maxvlen = 0
    for idx, l in enumerate(bvhist_grey):
        if prev != l:
            if cvec is None:
                cvec = [idx, idx, 0]
            else:
                cvec[1] = idx - 1
                cvec[2] = cvec[1] - cvec[0] + 1 
                ranges.append(cvec)
                if maxvlen < cvec[2]:
                    maxvlen = cvec[2]   # pylint: disable=unsubscriptable-object 
                cvec = None
        prev = l
    ret = [ [f[0],f[1]] for f in ranges if f[2] < maxvlen ]
    return ret

def get_graphforbbis(tarval, max, bbi):
    ret = np.zeros((max),dtype=np.uint8)
    for i in bbi: # not elegant but i am tired so wooden hammer has to do at the moment
        for idx in range(i[0],i[1]+1):
            ret[idx] = tarval
    return ret

def plotvhist(img_file,fignum):
    img_src = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    img_grey, img_otsu, img_bw = get_grey_otsu_bw(img_src)

    vhist_grey = np.average(img_grey,axis=1)
    vhist_otsu = np.average(img_otsu,axis=1)
    vhist_bw = np.average(img_bw,axis=1)

    img_grey = rotimg90(img_grey)
    img_otsu = rotimg90(img_otsu)
    img_bw = rotimg90(img_bw)

    bbis = get_borderboxes_intervals(img_src)
    bbigraph = get_graphforbbis(127, len(vhist_grey), bbis)
    
    
    fig = plt.figure()
    fig.suptitle(img_file)
    print(img_file)
    plt.subplot(331),plt.imshow(img_grey,cmap = 'gray', aspect='auto') #sublot(Anzahl Zeilen Anzahl Spalten Bild Nummer)
    plt.title('Grey'), plt.xticks([]), plt.yticks([])

    plt.subplot(332),plt.imshow(img_otsu,cmap = 'gray', aspect='auto')
    plt.title('Otsu'), plt.xticks([]), plt.yticks([])

    plt.subplot(333),plt.imshow(img_bw,cmap = 'gray', aspect='auto')
    plt.title('Bw'), plt.xticks([]), plt.yticks([])


    glimit_grey = np.min(vhist_grey) + np.average(vhist_grey)*.25
    glimit_otsu = np.min(vhist_otsu) + np.average(vhist_otsu)*.25
    glimit_bw = np.min(vhist_bw) + np.average(vhist_bw)*.25

    plt.subplot(334)
    plt.xlim(0,len(vhist_grey))
    plt.plot(vhist_grey,'b')
    plt.plot(np.repeat(glimit_grey,len(vhist_grey)),'r')
    plt.plot(bbigraph,'g')
    plt.xlim(0,len(vhist_grey))
    plt.subplot(335)
    plt.xlim(0,len(vhist_grey))
    plt.plot(vhist_otsu,'b')
    plt.plot(np.repeat(glimit_otsu,len(vhist_otsu)),'r')
    plt.subplot(336)
    plt.xlim(0,len(vhist_grey))
    plt.plot(vhist_bw,'b')
    plt.plot(np.repeat(glimit_bw,len(vhist_grey)),'r')


    bvhist_grey = thresbinv(vhist_grey,glimit_grey)
    bvhist_otsu = thresbinv(vhist_otsu,glimit_otsu)
    bvhist_bw = thresbinv(vhist_bw,glimit_bw)

    plt.subplot(337)
    plt.xlim(0,len(vhist_grey))
    plt.plot(bvhist_grey)
    plt.subplot(338)
    plt.xlim(0,len(vhist_grey))
    plt.plot(bvhist_otsu)
    plt.subplot(339)
    plt.xlim(0,len(vhist_grey))
    plt.plot(bvhist_bw)

    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

    # img_src, invdim = lkd.rescale(img_src, scale_percent=int(args["--scalepercent"]))

def test():
    srcfiles = ["C:\\tmp\\tweets\\2020_01_09_tweet.jpg",
                "C:\\tmp\\tweets\\2019_01_29_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_05_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_11_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_15_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_18_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_19_tweet.jpg",
                "C:\\tmp\\tweets\\2019_02_20_tweet.jpg",
                "C:\\tmp\\tweets\\2019_03_16_tweet.jpg",
                "C:\\tmp\\tweets\\2019_03_21_tweet.jpg",
                "C:\\tmp\\tweets\\2019_03_29_tweet.jpg",
                "C:\\tmp\\tweets\\2019_03_31_tweet.jpg",
                "C:\\tmp\\tweets\\2019_04_21_tweet.jpg",
                "C:\\tmp\\tweets\\2019_04_21_tweet.jpg",
                "C:\\tmp\\tweets\\2019_04_23_tweet.jpg",
                "C:\\tmp\\tweets\\2019_05_01_tweet.jpg" ]

    sfiles = sorted(srcfiles)

    for num, img_file in enumerate(sfiles):
        plotvhist(img_file,num+1)
    
    if len(sfiles) > 0:
        plt.show()



if __name__ == "__main__":
    test()
    #main()



