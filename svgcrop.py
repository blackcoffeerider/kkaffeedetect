import cv2
import numpy as np
import sys
import os
import re
import glob
import docopt
import libkaffeedetect as lkd
import svgreader 
import svgcreator


def main():
    """ {pyname} converts a SVG file to a PNG file cropping out a quadrangle given by a SVG path
    By default the original quadrangles orientation and position is kept as is and the background is made transparent.
    Alternatively the quadrangles can be zoomed to or projected flat to an estimated aspect ratio

    Usage:
        {pyname} <imagefilenameglob> [<outputdir>] [options] 
        {pyname} -h
    
    Options:
        -h, --help                   Show this help message
        --projected                  project quadrangle to a rectangle trying to estimate the perspective and aspect ratio (takes precedence over zoom)
        --zoom                       creates a image of the shape fitted into the minimum encompassing rectangle
        --showimages                 Shows images used during processes
        --autosuffix                 automatically sets "_cropped", "_zoom" or "_project" as a suffix
        --outputtype=(png|svg)       [default: png] png or svg
        --suffix=<suffix>            Suffix appended to the original name yourfile<suffix>.png
        --pathid=<pathid>            [default: cutoutpath]
        --scale=<scalepercent>       [default: 100]
        --svgtemplate=<svgtemplate>  svgtemplate to use as target [default: svgtemplate.svg]
        
    """
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    args = docopt.docopt(main.__doc__, help=True)

    if args["<imagefilenameglob>"] is None or args["--help"] == True:
        print(main.__doc__)
        return
    if args["<outputdir>"] is None:
        args["<outputdir>"] = "."
    if args["--outputtype"] != "png" and args["--outputtype"] != "svg":
        print("Error: Only png and svg are supported for option --outputtype")
        return
    if args["--outputtype"] == "svg":
        if not os.path.isfile(args["--svgtemplate"]):
            print("Error: file not found: "+args["--svgtemplate"])
            return
    if args["--suffix"] is None:
        if args["--autosuffix"]:
            if args["--projected"]:
                args["--suffix"] = "_projected"
            elif args["--zoom"]:
                args["--suffix"] = "_zoom"
            else:
                args["--suffix"] = "_cropped"
        else:
            args["--suffix"] = ""
    if not os.path.isdir(args["<outputdir>"]):
        print(main.__doc__)
        return
    

    files = sorted(glob.glob(args["<imagefilenameglob>"]))
    if files is None or len(files) == 0:
        print("no files matched {}".format(args["<imagefilenameglob>"]))
        return

    for srcimgfname in files:
            fnamepart = os.path.basename(srcimgfname)
            fnamepart = re.sub(r"\.[^\.]*$", "", fnamepart)
            dstimgfname = os.path.join(args["<outputdir>"],fnamepart)+args["--suffix"]+"."+args["--outputtype"]
            print("{} => {}".format(srcimgfname, dstimgfname))
            
            srcshapedct, srcimgdatadct = svgreader.parse_from_svg(svgreader.read_svg(srcimgfname),pathids=["cutoutpath"],imgids=["tweetimage"],fname=srcimgfname)
            srcshape = srcshapedct.get(args["--pathid"])
            srcimgdatatuple = srcimgdatadct.get("tweetimage")
            srcimgdata, w, h = srcimgdatatuple
            if srcimgdata.shape != (h, w):
                srcshape = (srcshape * [srcimgdata.shape[1]/w, srcimgdata.shape[0]/h]).astype(np.int32)

            if args["--scale"] != "100":
                srcimgdata, srcshape = lkd.rescale_img_and_shape(srcimgdata, srcshape, int(args["--scale"])/100)

            if args["--projected"]:
                dstimgdata = lkd.get_scaled_image_keeping_aspect(srcimgdata,srcshape)
                dstimgshape = np.array([[0,0],
                                        [dstimgdata.shape[1],0],
                                        [dstimgdata.shape[1],dstimgdata.shape[0]],
                                        [0,dstimgdata.shape[0]]
                                       ],dtype=np.int32)

            elif args["--zoom"]:
                dstimgdata, _, dstimgshape = lkd.get_shape_cropped(srcimgdata,srcshape)
            else:
                dstimgdata = srcimgdata
                dstimgshape =  srcshape


            alhpaimg = lkd.get_img_shape_alpha(dstimgdata, dstimgshape)
            svgcreator.write_image(dstimgfname, alhpaimg, dstimgshape, args["--outputtype"], args["--svgtemplate"])

            if args["--showimages"]:
                lkd.drawimg(dstimgdata, [dstimgshape], label="DST", scalefac=1)#, red=srcshpcntr) 
                cv2.waitKey(0)

    if args["--showimages"]:
        cv2.destroyAllWindows()  

if __name__ == "__main__":
    main()