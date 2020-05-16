import sys
import os
import docopt
import re
import base64
import numpy as np
import libkaffeedetect as lkd
import glob
import cv2
import svg.path


# regshapetag = re.compile(r"<[^>]*id=\"cutoutpath\"[^>]*/>")

def comptovec(cin):
    assert isinstance(cin,complex)
    return np.array([[cin.real,cin.imag]],dtype=np.int32) 


pathregdict = {}
regshapepath = re.compile(r"\s+d=\"([^\"]*)\"")
def get_path_vecarray(pathid,svgstr):
    """ 
        Get SVG path points as vector array 
        At the moment it only supports Move and Line Commands 
        The points are onle the edge points not the intermediate points between of curves etc!
        !!! This is not proper XML Parsing !!! 
    """
    shape = None
    if pathid not in pathregdict:
        pathregdict[pathid] = re.compile(r"<[^>]*id=\""+pathid+r"\"[^>]*/>") 
    shapetag = pathregdict[pathid].search(svgstr)
    if shapetag is not None:
        mpath = regshapepath.search(shapetag[0])
        if mpath is not None:
            parsedpath = svg.path.parse_path(mpath[0])
            shape = np.array([[0,0]]) # 1 dummy
            c = np.array([[0,0]])

            for e in parsedpath:
                if isinstance(e,svg.path.path.Move):
                    c = c + comptovec(e.point(0))
                    shape = np.append(shape,c,axis=0)
                elif isinstance(e,svg.path.path.Line):
                    c = comptovec(e.end)
                    shape = np.append(shape,c,axis=0)

            shape = np.delete(shape,0,axis=0) # delete dummy
    return shape


imgregdict = {}
regb64filelinkpattern = re.compile(r"href=\"data:image/[^;]*;base64,([^\"]*)\"")
def get_embedded_image_as_cv2(imgid,svgstr):
    """ 
        Extract embedded images by id as a cv2 image
        !!! This is not proper XML Parsing !!! 
    """
    if imgid not in imgregdict:
        imgregdict[imgid] = re.compile(r"<image[^>]*id=\""+imgid+r"\"[^>]*/>") 
    imgtag = imgregdict[imgid].search(svgstr)
    imgdata = None
    w = None
    h = None
    if imgtag is not None:
        mimgdata = regb64filelinkpattern.search(imgtag[0])
        if mimgdata.lastindex == 1:
            imgstr = mimgdata[1].replace('\n','').replace(' ','')
            imgdatabin = np.frombuffer(base64.b64decode(imgstr, validate=True), dtype=np.uint8)
            imgdata = cv2.imdecode(imgdatabin, cv2.IMREAD_UNCHANGED) 
            wm = re.search(r"width[^=]*=[^\"]*\"([^\"]*)\"",imgtag[0])
            hm = re.search(r"height[^=]*=[^\"]*\"([^\"]*)\"",imgtag[0])
            if wm is not None and hm is not None \
                and wm.lastindex == 1 and hm.lastindex == 1:
                w = int(float(wm[1]))
                h = int(float(hm[1]))
                # imgdata = cv2.resize(imgdata,(w,h))
            else:
                w = imgdata.shape[1]
                h = imgdata.shape[0]
    return (imgdata, w, h)


def read_svg(filename):
    svgdata = None
    with open(filename, "r")  as svgfiledesc:
        svgdata = svgfiledesc.read()
    return svgdata


def parse_from_svg(svgdata, pathids=[], imgids=[], fname="notsupplied.svg"):
    shapes = {}
    imgs = {}
    try:
        if svgdata is not None:
            for pid in pathids:
                shape = get_path_vecarray(pid, svgdata) 
                if shape is not None:
                    shapes[pid] = shape 
                else:
                    print("!!!Error in {} can't find shape id \"{}\"".format(fname, pid))

            for imgid in imgids:
                imgdata = get_embedded_image_as_cv2(imgid, svgdata)
                if imgdata is not None:
                    imgs[imgid] = imgdata
                else:
                    print("!!!Error in {} can't find image id \"{}\"".format(fname, imgid))

    except Exception as e:
        print("!!!Unknown Error in {} e: {}".format(fname,e))

    return shapes, imgs


def main():
    """ {pyname} reads a given set of svgs with a path and shows them on the screen

    Usage:
        {pyname} <imagefilenameglob> [--pathid=<pathid>] [--scale=<scalepercent>]
        {pyname} -h
    
    Options:
        -h, --help              Show this help message
        --pathid=<pathid>       [default: cutoutpath]
        --scale=<scalepercent>  [default: 100]
        
    Example:
        python {pyname} 02_shape_detected_imgs/2019_03*.jpg.svg --scale=50

    """    
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    # print(main.__doc__)
    args = docopt.docopt(main.__doc__, help=False)
    # pprint.pprint(args)
    if args["<imagefilenameglob>"] is None or args["--help"] == True:
        print(main.__doc__)
        return

    files = sorted(glob.glob(args["<imagefilenameglob>"]))

    if files is None or len(files) == 0:
        print("no files matched {}".format(args["<imagefilenameglob>"]))
        return

    for srcimgfname in files:
        print("{}".format(srcimgfname))
        
        srcshapedct, srcimgdatadct = parse_from_svg(read_svg(srcimgfname),pathids=[args["--pathid"]],imgids=["tweetimage"],fname=srcimgfname)
        srcshape = srcshapedct.get(args["--pathid"])
        srcimgdatatuple = srcimgdatadct.get("tweetimage")
        srcimgdata, w, h = srcimgdatatuple
        if srcimgdata.shape != (h, w):
            srcshape = (srcshape * [srcimgdata.shape[1]/w, srcimgdata.shape[0]/h]).astype(np.int32)

        lkd.drawimg(srcimgdata, [srcshape], label=srcimgfname, scalefac=int(args["--scale"])/100)#, red=srcshpcntr) 

        cv2.waitKey(0)

        cv2.destroyAllWindows()  


if __name__ == "__main__":
    main()