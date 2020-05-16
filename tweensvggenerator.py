import os
import sys
import glob
import cv2
import re
import numpy as np
import libkaffeedetect as lkd
import svgreader
import svgcreator
import docopt


def get_tweenshp(a,b,factor):
    tw = (b-a)*factor
    ret = a + tw
    return ret.astype(np.int32)


def get_tween_image(src_img, dst_img, src_shape, dst_shape, factor=0.5):
    assert src_img.shape == dst_img.shape and 0 <= factor <= 1
    tweenshp = get_tweenshp(src_shape,dst_shape,factor)
    srctweenimg = lkd.get_warped_img_shape(src_img,src_shape,tweenshp)
    dsttweenimg = lkd.get_warped_img_shape(dst_img,dst_shape,tweenshp)

    alpha = 1 - factor
    beta = factor
    tweenimg = cv2.addWeighted(srctweenimg, alpha, dsttweenimg, beta, 0.0)
    return tweenimg, tweenshp


def main():
    """ {pyname} creates a simple morph between multiple files resolved by one or two file patterns given
    by default the source images are scaled to fit the given out

    Usage:
        {pyname} <imagefilenamegloba> <imagefilenameglobb> <width>x<heigth> [options] 
        {pyname} <imagefilenameglob> <width>x<heigth> [options]
        {pyname} -h<w
    
    Options:
        -h, --help                   Show this help message
        -o, --outputdir=<outputdir>  [default: .] output directory    
        --projected                  project quadrangle to a rectangle trying to estimate the perspective and aspect ratio (takes precedence over zoom)
        --showimages                 Shows images used during processes
        --noscale                    if this is set the source image is not made to fit the specified image size as much as possible but centered if the target image is bigger than the source
        --frames=<frames>            [default: 30]  frames for tweening                       
        --stillframes=<stillframes>  [default: 360] how many frames the last frame is repeated
        --framedigits=<framedigits>  [default: 4]   digits to which the frame number is padded
        --suffix=<suffix>            Suffix appended to the original name yourfile_<suffix>_frameno.svg
        --outputtype=(png|svg)       [default: png] png or svg
        --pathid=<pathid>            [default: cutoutpath]
        --svgtemplate=<svgtemplate>  svgtemplate to use as target [default: svgtemplate.svg]
        
    """
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    args = docopt.docopt(main.__doc__, help=False)
    print(args)

    if args["--help"] == True:
        print(main.__doc__)
        return

    if (args["<imagefilenamegloba>"] is None or args["<imagefilenameglobb>"] is None) \
         and args["<imagefilenameglob>"] is None:
         print(main.__doc__)
         print("ERROR: Please provide file-paths to tween")
         return

    if args["<imagefilenamegloba>"] is not None and args["<imagefilenameglobb>"] is not None:
        farraya = sorted(glob.glob(args["<imagefilenamegloba>"]))
        if len(farraya) == 0:
            print("ERROR: No files found for {}".format(args["<imagefilenamegloba>"]))
            return

        farrayb = sorted(glob.glob(args["<imagefilenameglobb>"]))
        if len(farrayb) == 0:
            print("ERROR: No files found for {}".format(args["<imagefilenameglobb>"]))
            return

        farray = farraya + farrayb

    if args["<imagefilenameglob>"] is not None:
        farray = sorted(glob.glob(args["<imagefilenameglob>"]))
        if len(farray) == 0:
            print("ERROR: No files found for {}".format(args["<imagefilenameglob>"]))
            return

    if len(farray) < 2:
        print("ERROR: Not enough files to tween: {}".format(farray))
        return

    if farray is None:
        print(main.__doc__)
        print("Error: please provide files to tween")
        return        

    width = 1
    heigth = 1
    if re.match(r"\d+x\d+", args["<width>x<heigth>"]):
        width, heigth = args["<width>x<heigth>"].split("x")
        width, heigth = int(width), int(heigth)
    else:
        print(main.__doc__)
        print("Error: Please povide dimensions")
        return

    if args["--suffix"] is None:
        args["--suffix"] = ""

    if args["--outputtype"] != "png" and args["--outputtype"] != "svg":
        print("Error: Only png and svg are supported for option --outputtype")
        return
    if args["--outputtype"] == "svg":
        if not os.path.isfile(args["--svgtemplate"]):
            print("Error: file not found: "+args["--svgtemplate"])
            return

    args["--frames"] = int(args["--frames"])
    args["--stillframes"] = int(args["--stillframes"])
    
    limit = len(farray)
    dim = np.array([heigth,width])

    first = True
    for idx in range(limit-1):
        srcimgfname = farray[idx]
        dstimgfname = farray[idx+1]
        fnamepart = os.path.basename(srcimgfname)
        fnamepart = re.sub(r"(\.jpg)?\.[^\.]*$", "", fnamepart)
        outimgfnamebasestr = "{org}_{frameno:0"+args["--framedigits"]+"d}.{ftype}"
        if args["--suffix"] != '':
            outimgfnamebasestr = "{org}_{suffix}_{frameno:0"+args["--framedigits"]+"d}.{ftype}"

        print("{} => {}".format(srcimgfname, dstimgfname))
        
        srcshapedct, srcimgdatadct = svgreader.parse_from_svg(svgreader.read_svg(srcimgfname),pathids=[args["--pathid"]],imgids=["tweetimage"],fname=srcimgfname)
        srcshape = srcshapedct.get(args["--pathid"])
        srcimgtup = srcimgdatadct.get("tweetimage")
        srcimgdata, w, h = srcimgtup
        if srcimgdata.shape[2] == 4:
            srcimgdata = cv2.cvtColor(srcimgdata, cv2.COLOR_BGRA2BGR)
        if srcimgdata.shape != (h, w):
            srcshape = (srcshape * [srcimgdata.shape[1]/w, srcimgdata.shape[0]/h]).astype(np.int32)

        dstshapedct, dstimgdatadct = svgreader.parse_from_svg(svgreader.read_svg(dstimgfname),pathids=[args["--pathid"]],imgids=["tweetimage"],fname=dstimgfname)
        dstshape = dstshapedct.get(args["--pathid"])
        dstimgtup = dstimgdatadct.get("tweetimage")
        dstimgdata, w, h = dstimgtup
        if dstimgdata.shape[2] == 4:
            dstimgdata = cv2.cvtColor(dstimgdata, cv2.COLOR_BGRA2BGR)
        if dstimgdata.shape != (h, w):
            dstshape = (dstshape * [dstimgdata.shape[1]/w, dstimgdata.shape[0]/h]).astype(np.int32)

        if args["--projected"]:
            srcimgdata = lkd.get_scaled_image_keeping_aspect(srcimgdata,srcshape)
            srcshape = np.array([[0,0],
                                    [srcimgdata.shape[1],0],
                                    [srcimgdata.shape[1],srcimgdata.shape[0]],
                                    [0,srcimgdata.shape[0]]
                                    ],dtype=np.int32)

            dstimgdata = lkd.get_scaled_image_keeping_aspect(srcimgdata,srcshape)
            dstshape = np.array([[0,0],
                                    [dstimgdata.shape[1],0],
                                    [dstimgdata.shape[1],dstimgdata.shape[0]],
                                    [0,dstimgdata.shape[0]]
                                    ],dtype=np.int32)

        if srcshape is not None and dstshape is not None \
            and srcimgdata is not None and dstimgdata is not None:

            srcimgdata, _ = lkd.blacken_outside_shape(srcimgdata,srcshape)
            dstimgdata, _ = lkd.blacken_outside_shape(dstimgdata,dstshape)

            # copy source image to black common size
            srcblk = np.zeros([dim[0],dim[1],3],dtype=np.uint8) # create black image
            if args["--noscale"] == False or  dim[1] < srcimgdata.shape[1] or dim[0] < srcimgdata.shape[0]:
                srcimgdata, srcshape = lkd.rescale_img_and_shape_to_size_keeping_aspect(srcimgdata, srcshape, dim[1], dim[0]) # scale source image to fit target image 
            srcoff = np.array([srcblk.shape[1],srcblk.shape[0]]) - np.array([srcimgdata.shape[1],srcimgdata.shape[0]]) # calculate offset to center original image # pylint: disable=unsubscriptable-object # srcblk.shape is subscriptable
            srcoff = (srcoff/2).astype(np.int32) # calculate offset to center original image
            srcmapul = np.min(srcshape + srcoff, axis=0)
            srcimgdata, srcshape  = lkd.copy_shape(srcimgdata,srcblk,srcshape,srcmapul)
            

            dstblk = np.zeros([dim[0],dim[1],3],dtype=np.uint8) # create black image 
            if args["--noscale"] == False or  dim[1] < dstimgdata.shape[1] or dim[0] < dstimgdata.shape[0]:
                dstimgdata, dstshape = lkd.rescale_img_and_shape_to_size_keeping_aspect(dstimgdata, dstshape, dim[1], dim[0]) # scale destination image to fit target image 
            dstoff = np.array([dstblk.shape[1],dstblk.shape[0]]) - np.array([dstimgdata.shape[1],dstimgdata.shape[0]]) # calculate offset to center original image # pylint: disable=unsubscriptable-object # dstblk.shape is subscriptable
            dstoff = (dstoff/2).astype(np.int32) # calculate offset to center original image
            dstmapul = np.min(dstshape + dstoff, axis=0)
            dstimgdata, dstshape = lkd.copy_shape(dstimgdata,dstblk,dstshape,dstmapul)

            if args["--showimages"]:
                lkd.drawimg(srcimgdata, [srcshape], label="SRC", scalefac=1)#, red=srcshpcntr) 
                lkd.drawimg(dstimgdata, [dstshape], label="DST", scalefac=1)#, red=dstshpcntr)
                lkd.drawimg(srcimgdata, [srcshape], label="MORPH", scalefac=1)

                if first:
                    first = False
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(3000)

            tweenimg = np.zeros(srcimgdata.shape)
            steps = args["--frames"]
            for pos in range(0,steps+1):
                outimgfname = os.path.join(args["--outputdir"], outimgfnamebasestr.format(org=fnamepart,suffix=args["--suffix"],frameno=pos,ftype=args["--outputtype"]))
                #print(outimgfname)
                tweenimg, tweenshp = get_tween_image(srcimgdata, dstimgdata, srcshape, dstshape, pos/steps)
                alhpaimg = lkd.get_img_shape_alpha(tweenimg, tweenshp)    
                if args["--showimages"]:
                    lkd.drawimg(tweenimg, [tweenshp], label="MORPH", scalefac=1)
                    cv2.waitKey(20)
                svgcreator.write_image(outimgfname, alhpaimg, tweenshp, args["--outputtype"], args["--svgtemplate"])
            for pos in range(0,args["--stillframes"]+1):
                cpos = pos+args["--frames"]
                outimgfname = os.path.join(args["--outputdir"],outimgfnamebasestr.format(org=fnamepart,suffix=args["--suffix"],frameno=cpos,ftype=args["--outputtype"]))
                #print(outimgfname)
                svgcreator.write_image(outimgfname, alhpaimg, tweenshp, args["--outputtype"], args["--svgtemplate"])

    if args["--showimages"]:
        cv2.destroyAllWindows()  




if __name__ == "__main__":
    main()
