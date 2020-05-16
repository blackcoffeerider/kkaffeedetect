import numpy as np
import cv2
import math
import functools
import scipy.spatial.distance
import imagescaler

get_scaled_image_keeping_aspect = imagescaler.get_scaled_image_keeping_aspect


growdirs = [[0,1],[1,2],[2,3],[0,3]]

shrinkdetectstep = -10

def makefuture_rect(v,fn,idx):
    a = v[idx].copy()
    b = fn(v)
    b = b[idx]
    b = np.flip(b,0)
    # res = np.append(a,[[int((a[2][0]+b[0][0])/2),int((a[2][1]+b[0][1])/2)]],axis=0)
    res = np.append(a,b,axis=0)
    # res = np.append(res,[[int((a[0][0]+b[2][0])/2),int((a[0][1]+b[2][1])/2)]],axis=0)
    return res


def process_image_center(img_src, img_grey, thres, org, draw=False, drawsteps=False):
    orgarray = np.array([org,org,org,org])
    # print(org)
    # print(grow(orgarray,5))
    # print(img_grey.shape)

    start = grow(orgarray,int(((img_grey.shape[0]+img_grey.shape[1])/2)/4))
    c = np.array([start.copy(),start.copy(),start.copy(),start.copy()])
    for idx, gr in enumerate(growdirs):
        for v in gr:
            if idx == 0:
                c[idx][v][1] = 0
            elif idx == 1:
                c[idx][v][0] = img_grey.shape[1]
            elif idx == 2:
                c[idx][v][1] = img_grey.shape[0]
            elif idx == 3:
                c[idx][v][0] = 0

    if draw:
        drawimg(img_src, c, None)
        cv2.waitKey(100)

    for idx, v in enumerate(c):
        # if idx != 3:
        #     continue
        p = None
        avg = 255
        # if draw and drawsteps or 1 == 1:
        #     print(str(idx)+"============================")
        #     print(v)
        #     print((shrinkdetectstep, v.min(), np.max(v, axis=0)[0], np.max(v, axis=0)[1] , img_grey.shape[1], img_grey.shape[0]))#
        while shrinkdetectstep <= v.min() and np.max(v, axis=0)[0] <= img_grey.shape[1] and np.max(v, axis=0)[1] <=  img_grey.shape[0]:
            avg = 0
            future = None
            if idx == 0 or idx == 2:
                v = grow_y(v,shrinkdetectstep ,idx=growdirs[idx])
                future = makefuture_rect(v, functools.partial(grow_y, s=shrinkdetectstep, idx=growdirs[idx]), growdirs[idx])
            elif idx == 1 or idx == 3:
                v = grow_x(v,shrinkdetectstep ,idx=growdirs[idx])
                future = makefuture_rect(v, functools.partial(grow_x, s=shrinkdetectstep, idx=growdirs[idx]), growdirs[idx])
            c[idx] = v
            # print(v)
            if draw and drawsteps:
                drawimg(img_src, c, future)
                cv2.waitKey(50)
                # cv2.waitKey(200)
            p = get_pixel(img_grey,future)
            avg = np.average(p)
            if math.isnan(avg):
                print(avg)
            #print(avg)
            # if draw and drawsteps:
            #     print("Idx: {}, thres:{} avg: {}".format(idx,thres, avg))
            if avg > thres :
                break
    res = np.array([[c[3][0][0],c[0][0][1]],[c[1][2][0],c[0][0][1]],
           [c[1][2][0],c[2][3][1]],[c[3][0][0],c[2][3][1]]
          ])
    #print(res)
    if draw:
        minr = get_scaled_rect(res, percentage=-4) 
        maxr = get_scaled_rect(res, percentage=7) 
        
        drawimg(img_src, [res], red=maxr, green=minr)
        cv2.waitKey(0)
    return res


def process_image_shrink_boxdetect(img_src, thres, img_grey, draw=False, drawsteps=False, scale_percent=50):
    w = img_src.shape[1]
    h = img_src.shape[0]
    img_src, invdim = rescale(img_src, scale_percent=scale_percent)
    img_grey, _ = rescale(img_grey, scale_percent=scale_percent)


    org = np.array([int(img_grey.shape[1]/2),int(img_grey.shape[0]/1.5)])
    ret = process_image_center(img_src,img_grey,thres,org,draw=False,drawsteps=drawsteps)
    org = np.array([int((ret[0][0]+ret[2][0])/2),int((ret[0][1]+ret[2][1])/2)])
    ret = process_image_center(img_src,img_grey,thres,org,draw=draw,drawsteps=drawsteps)
    ret = np.multiply(ret, invdim, out=ret, casting='unsafe', dtype=np.int32)
    # print(img_file)
    return ret, w, h   

def rescale(img_process, scale_percent=20):
    width = int(img_process.shape[1] * scale_percent / 100)
    height = int(img_process.shape[0] * scale_percent / 100)
    dim = (width, height)
    invdim = np.array([img_process.shape[1]/width, img_process.shape[0]/height])
    
    res = cv2.resize(img_process,dim)
    return (res, invdim)

def get_borders_from_grey_smear(img_src,img_grey, draw=False):
    smimg_raw = get_grey_smear(img_grey)
    thresg, _ = cv2.threshold(smimg_raw, 20, 200, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, smimg = cv2.threshold(smimg_raw, thresg*1.3, 255, cv2.THRESH_BINARY)
    ret, w, h = process_image_shrink_boxdetect(img_src,thresg,smimg,draw=False,scale_percent=100)
    if draw:    
        cv2.imshow("SMIMG", smimg_raw)
        cv2.imshow("SMIMGBW", smimg)
        cv2.waitKey(100)

    return ret, w, h



def get_grey_smear(img_grey):

    xa = np.average(img_grey,axis=0)
    ya = np.average(img_grey,axis=1)

    xaf = np.repeat([xa], ya.shape[0], axis=0).reshape(img_grey.shape)
    yaf = np.repeat([ya], xa.shape[0]).reshape(img_grey.shape)

    ret = xaf+yaf
    ret = np.divide(ret,2)
    ret = ret.astype(np.uint8)

    return ret

def guess_has_paintbox(img_grey, thres=0.3):

    ya = np.average(img_grey,axis=1)

    auq = np.average(ya[0:int(ya.shape[0]*.08)])
    afi = np.average(ya)
    auquot = (auq/afi)
    if auquot > thres:
        return True
    return False


def get_scaled_rect(srcrect, percentage):
    ret = srcrect.copy()
    
    xres = int((srcrect[1][0]-srcrect[3][0])*(percentage/100))
    yres = int((srcrect[2][1]-srcrect[0][1])*(percentage/100))
    
    ret = grow_x(ret,s=xres)
    ret = grow_y(ret,s=yres)

    return ret

def get_normy_orientation_sv(ab):
    normy = np.array([0,1])
    # print(a)
    # print(b)
    # print(ab)
    dotprod = np.dot(normy,ab)
    cosine_angle = dotprod  / (np.linalg.norm(normy) * np.linalg.norm(ab))
    angle = np.arccos(cosine_angle)
    angdeg = np.degrees(angle)
    if ab[0] < 0:
         angdeg = 360 - angdeg 
    return angdeg

def get_normy_orientation(a,b):
    normy = np.array([0,1])
    # print(a)
    # print(b)
    ab =  b - a
    # print(ab)
    dotprod = np.dot(normy,ab)
    cosine_angle = dotprod  / (np.linalg.norm(normy) * np.linalg.norm(ab))
    angle = np.arccos(cosine_angle)
    angdeg = np.degrees(angle)
    if ab[0] < 0:
         angdeg = 360 - angdeg 
    return angdeg

def get_angle_3vec(v):
    ba = v[0] - v[1] 
    bc = v[2] - v[1]

    dotprodb = np.dot(ba, bc)

    cosine_angle = dotprodb  / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angdeg = np.degrees(angle)
    # if dotprod < 0:
    #     angdeg = angdeg * -1 
    return angdeg 


def drawimg(img_src, v, red=None, green=None, purple=None, label="SRC", scalefac=1.0):
    if scalefac != 1.0:
        draw_img, _ = rescale(img_src,scalefac*100)
    else:
        draw_img = img_src.copy()
    for c in v:
        cout = (c * scalefac).astype(np.int32)
        cv2.drawContours(draw_img, [cout], 0,[255,0,0], 3)
    if red is not None:
        redout = (red*scalefac).astype(np.int32)
        cv2.drawContours(draw_img, [redout], 0,[0,0,255], 3)
    if green is not None:
        greenout = (green*scalefac).astype(np.int32)
        cv2.drawContours(draw_img, [greenout], 0,[0,255,0], 3)
    if purple is not None:
        purpleout = (purple*scalefac).astype(np.int32)
        cv2.drawContours(draw_img, [purpleout], 0,[255,0,255], 3)
    cv2.imshow(label,draw_img)
    # draw_img = img_grey.copy()
    # for c in v:
    #     cv2.drawContours(draw_img, [c], 0,[255,0,0], 3)
    # cv2.imshow("GREY",draw_img)



def get_pixel(img_grey, countour):
    # Create a mask image that contains the contour filled in
    cimg = np.zeros_like(img_grey)
    cv2.drawContours(cimg, [countour], 0, color=255, thickness=-1)

    # Access the image pixels and create a 1D numpy array 
    pts = np.where(cimg == 255)
    return img_grey[pts[0], pts[1]]


white = np.uint8([255,255,255 ])
lower_green  = np.array([68, 29, 29], dtype = "uint8")
upper_green  = np.array([175, 255, 255], dtype = "uint8")

man_finger_lower = np.array([0, 48, 50], dtype = "uint8")
man_finger_upper = np.array([20, 255, 255], dtype = "uint8")

def prefilter_colors(img_process):
    # img_nogreen = img_process * np.array([1,0,0], dtype = "uint8") +  np.array([0,1,0], dtype = "uint8")
    # cv2.imshow('img_nogreen',img_nogreen)
    img_hsv = cv2.cvtColor(img_process, cv2.COLOR_BGR2HSV) 
    # cv2.imshow('IMGHSV',img_hsv)


    man_mask_finger = cv2.inRange(img_hsv, man_finger_lower, man_finger_upper)
    man_mask_finger_inv = cv2.bitwise_not(man_mask_finger)
    # cv2.imshow('Mask_man_finger',man_mask_finger_inv)

    # mask_finger = cv2.bitwise_not(cv2.inRange(img_hsv, lower_finger, upper_finger))
    # mask_red = cv2.bitwise_not(cv2.inRange(img_hsv,  upper_red, lower_red))
    mask_green = cv2.bitwise_not(cv2.inRange(img_hsv, lower_green,upper_green))
    #cv2.imshow('Mask_Green',mask_green)
    # mask_blue = cv2.bitwise_not(cv2.inRange(img_hsv, lower_blue, upper_blue))


    #cv2.imshow('Mask',mask)
    res = cv2.bitwise_and(img_process, img_process, mask= mask_green)  #-- Contains pixels having the gray color--
    # cv2.imshow('ResMGreen',res)
    res2 = cv2.bitwise_and(res, white, mask= man_mask_finger_inv)  #-- Contains pixels having the gray color--
    return res2 # res2

def blur_image(img_process, blur=0):
    if blur > 0:
        res = cv2.blur(img_process, (blur,blur))
        return res
    return img_process  

def makecross(center, sizehalf, limits):
    #     1 2  
    #  11 0 3 4
    #  10 9 6 5
    #     8 7

    
    ret = np.zeros((12,2),dtype=np.int32)
    
    x = 0
    y = 1
    lx = center[x] - sizehalf # left x
    if lx < 0:
        lx = 0
    rx = center[x] + sizehalf # right x
    if rx > limits[x]:
        rx = limits[x]
    uy = center[y] - sizehalf # upper y
    if uy < 0:
        uy = 0
    ly = center[y] + sizehalf # lower y
    if ly > limits[y]:
        ly = limits[y]        
    ret[1][y]  = ret[2][y] = 0
    ret[8][y]  = ret[7][y] = limits[y] 
    ret[10][x] = ret[11][x] = 0
    ret[4][x]  = ret[5][x] = limits[x] 

    ret[1][x] = ret[0][x] = ret[9][x] = ret[8][x] = lx
    ret[2][x] = ret[3][x] = ret[6][x] = ret[7][x] = rx
 
    ret[11][y] = ret[0][y] = ret[3][y] = ret[4][y] = uy
    ret[10][y] = ret[9][y] = ret[6][y] = ret[5][y] = ly
    
    return ret
    
def get_average_cross(img_src, img_grey, center, sizehalf,  draw=False):
    cross = makecross(center,sizehalf,[img_grey.shape[1],img_grey.shape[0]])
    # print(v)
    if draw:
        drawimg(img_src, [cross])
        cv2.waitKey(10)
        # cv2.waitKey(200)
    p = get_pixel(img_grey,cross)
    avg = np.average(p)
    if math.isnan(avg):
        print(avg)
    #print(avg)

    return avg    


def get_veclen(vec):
    ret = np.sqrt(np.dot(vec,vec))
    return ret


def get_norm_vec(vec):
    ret = vec/np.sqrt(np.dot(vec,vec))
    return ret


def grow_x(v, s = 5, idx = range(4)):
    ret = v.copy() 
    if 1 in idx:
        ret[1] += [s,0] 
    if 2 in idx:
        ret[2] += [s,0] 
    if 0 in idx:
        ret[0] += [-s,0]     
    if 3 in idx:
        ret[3] += [-s,0]
    return ret   

def grow_y(v, s = 5, idx = range(8)):
    ret = v.copy() 
    if 0 in idx:
        ret[0] += [0,-s] 
    if 1 in idx:
        ret[1] += [0,-s] 
    if 2 in idx:
        ret[2] += [0,s]
    if 3 in idx:
        ret[3] += [0,s]
    return ret   

def grow(v, s, idx = range(8)):
    ret = grow_y(grow_x(v,s,idx),s,idx) 
    return ret   



def move_to_center(mtrx,newcenter):
    mtrxcenter = np.average(mtrx,axis=0)
    mtrx = mtrx + (newcenter-mtrxcenter)
    return mtrx.astype(np.int32)


def blacken_outside_shape(img_process,shape):
    stencil = np.zeros(img_process.shape,dtype=np.uint8)
    stencil = cv2.drawContours(stencil, [shape], 0,[255,255,255], thickness=cv2.FILLED)
    retimg = cv2.bitwise_and(img_process, stencil)
    return retimg, stencil

def rescale_img_and_shape(img_process,shape,scalefac=1.0):
    retimg, _ = rescale(img_process,scale_percent=scalefac*100)
    retshp = shape*scalefac
    retshp = retshp.astype(np.int32)
    return retimg, retshp


def rescale_img_and_shape_to_size_keeping_aspect(img_process, shape, maxw, maxh):
    scale = 1 
    # orgshape = np.array([img_process.shape[1], img_process.shape[0]])
    ratiomax = maxw/maxh
    ratioimg = img_process.shape[1] / img_process.shape[0] 

    if ratiomax < ratioimg:
        scale = maxw/img_process.shape[1]  
    else:
        scale = maxh/img_process.shape[0]

    dim = (int(img_process.shape[1]*scale), int(img_process.shape[0]*scale))
    
    
    retimg = cv2.resize(img_process,dim)
    # newshape = np.array([img_process.shape[1], img_process.shape[0]])
    # reshape = orgshape/newshape

    retshp = shape*scale
    retshp = retshp.astype(np.int32)
    return retimg, retshp

def get_shape_cropped(srcimg, shape):
    mins = np.min(shape, axis=0)
    maxs = np.max(shape, axis=0)
    if mins[0] < 0:
        mins[0] = 0
    if mins[1] < 0:
        mins[1] = 0
    if maxs[0] > srcimg.shape[1]:
        maxs[0] = srcimg.shape[1]
    if maxs[1] > srcimg.shape[0]:
        maxs[1] = srcimg.shape[0]
    dim = maxs - mins

    stencilshape = shape - mins
    
    stencil = np.zeros([dim[1],dim[0],srcimg.shape[2]],dtype=np.uint8)
    stencil = cv2.drawContours(stencil, [stencilshape], 0,[255,255,255], thickness=cv2.FILLED)

    
    imgpart = srcimg[mins[1]:maxs[1],mins[0]:maxs[0]]

    retimg = cv2.bitwise_and(imgpart, stencil)

    return retimg, stencil, stencilshape

def copy_shape(srcimg,dstimg,shape,dstupleft):
    croppedsrcimg, stencil, stencilshape = get_shape_cropped(srcimg,shape)
    stencil_inv = cv2.bitwise_not(stencil)

    dstlr = dstupleft + np.array([croppedsrcimg.shape[1],croppedsrcimg.shape[0]])

    bgslice = dstimg[dstupleft[1]:dstlr[1],dstupleft[0]:dstlr[0]]
    bltbg = cv2.bitwise_and(bgslice, stencil_inv)

    bltmerge = cv2.add(bltbg, croppedsrcimg)

    retimg = dstimg.copy()
    retimg[dstupleft[1]:dstlr[1],dstupleft[0]:dstlr[0]] = bltmerge
    retshape = stencilshape + dstupleft
    return retimg, retshape
    

def get_warped_img_shape(img_process, currshape, dstshape):
    matrix = cv2.getPerspectiveTransform(currshape.astype(np.float32), dstshape.astype(np.float32))
    retimg = cv2.warpPerspective(img_process, matrix, (img_process.shape[1], img_process.shape[0]))
    return retimg






def get_img_shape_alpha(imgsrc,shape):
    imgprcs, stencil = blacken_outside_shape(imgsrc, shape)
    #b_channel, g_channel, r_channel = cv2.split(imgprcs)
    alpha_channel, _, _ = cv2.split(stencil)

    img_BGRA = np.dstack([imgprcs, alpha_channel]) 

    #img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA



def test_get_normy_orientation():
    ul = np.array([-2,2])
    lr = np.array([2,-2])
    ll = np.array([-2,-2])
    ur = np.array([2,2])
    zero = np.array([0,0])
    xp1 = np.array([1,0])
    xn1 = np.array([-1,0])
    yp1 = np.array([0,1])
    yn1 = np.array([0,-1])
    print(get_normy_orientation(zero,xp1)) # 90
    print(get_normy_orientation(zero,yn1)) # 180
    print(get_normy_orientation(zero,xn1)) # 270 # -90
    print(get_normy_orientation(zero,yp1)) # 0
    print(get_normy_orientation(ll,ur)) #45
    print(get_normy_orientation(ul,lr)) #135 
    print(get_normy_orientation(ur,ll)) #225
    print(get_normy_orientation(lr,ul)) #315