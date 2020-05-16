import numpy as np
import argparse
import glob
import cv2
from matplotlib import pyplot as plt


# import the necessary packages

def rescale(img_process, scale_percent=20):
    width = int(img_process.shape[1] * scale_percent / 100)
    height = int(img_process.shape[0] * scale_percent / 100)
    dim = (width, height)
    invdim = np.array([img_process.shape[1]/width, img_process.shape[0]/height])
    
    res = cv2.resize(img_process,dim)
    return (res, invdim)

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def process(img):
    img, _ = rescale(img, scale_percent=20)

    img = cv2.blur(img, (5,5))

    edges = auto_canny(img,sigma=.5)


    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()




g = []
g.append("tweetimgs/tweet_0*.jpg")
g.append("tweetimgs/tweet_001*.jpg")
g.append("tweetimgs/tweet_002*.jpg")
g.append("tweetimgs/tweet_003*.jpg")
g.append("tweetimgs/tweet_004*.jpg")
g.append("tweetimgs/tweet_005*.jpg")
g.append("01_downloaded_tweetimgs/2020*kktweet*.jpg")

for p in glob.glob(g[6]):
    img = cv2.imread(p,0)
    process(img)
