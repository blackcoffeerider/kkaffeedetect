import os
import re
import base64
import sys
import docopt
import pprint
import glob
import cv2
import shrinkoptcut
import simpleocvshapedetect


def loadfile_as_base64(filename):
    with open(filename,"rb") as file:
        fc = file.read()
    return str(base64.b64encode(fc).decode("utf-8"))


def encode_as_base64(binary):
    return str(base64.b64encode(binary).decode("utf-8"))


def get_svgtemplate(filename="svgtemplate.svg"):
    svgtem = ""
    with open(filename,"r") as f:
        svgtem = f.read()
    return svgtem


def write_image(outfname, outimage, outshape, outtype="png", svgtemplate="svgtemplate.svg"):
    if outtype == "png":
        cv2.imwrite(outfname,outimage)
    elif outtype == "svg":
        template = get_svgtemplate(svgtemplate)
        encoderet, binimage = cv2.imencode(".png", outimage) 
        if encoderet:
            b64file = encode_as_base64(binimage)
            outsvgstr = get_svg_image_str(b64file,outshape,template,outimage.shape[1],outimage.shape[0])
            with open(outfname,"w",encoding="utf-8") as outfile:
                outfile.write(outsvgstr)


def get_date_int(datefield):
    dparts = datefield.split(".")
    return int(dparts[2])*10000+int(dparts[1])*100+int(dparts[0])


regshapepath = re.compile(r"d=\"M[^\"]*\"")
regfilelink = re.compile(r"href=\"[^\"]*\"")
b64filelinkpattern = 'href="data:image/{};base64,{}"'
shapepathpattern = 'd="M {} z"'

def get_svg_image_str(b64file, shape, svgtemplate, width=-1.0, height=-1.0, ftype="jpeg"):
    shapepathstr = " ".join(["{:.2f},{:.2f}".format(f[0],f[1]) for f in shape])
    newshapepath = shapepathpattern.format(shapepathstr)
    ret = svgtemplate.replace("##imgw##",str(width))
    ret = ret.replace("##imgh##",str(height))
    ret = regshapepath.sub(newshapepath,ret)
    newfilelink = b64filelinkpattern.format(ftype,b64file)
    ret = regfilelink.sub(newfilelink,ret)
    return ret


def process_files(files, args):
    template = get_svgtemplate(args["--svgtemplate"])
    outfilepattern = '{}.svg'
    for srcimgfname in files:
        dstimgfname = os.path.join(args["<outputdir>"],outfilepattern.format(os.path.basename(srcimgfname)))
        print("{} => {}".format(srcimgfname, dstimgfname))
       
        if args["--method"] == "shrinkoptcut":
            shape, w, h = shrinkoptcut.process_image(srcimgfname,draw=args["--showimages"], drawsteps=args["--showprocessteps"], scale_percent=int(args["--scale"]))
        if args["--method"] == "simpleocv":
            rets, w, h, _ = simpleocvshapedetect.process_single_file(srcimgfname,theta=0.010,minvert=4,maxvert=50,scale_percent=100,show=args["--showimages"])
            shape = rets[0]
        b64file = loadfile_as_base64(srcimgfname)
        outsvgstr = get_svg_image_str(b64file,shape,template,w,h)
        
        with open(dstimgfname,"w",encoding="utf-8") as outfile:
            outfile.write(outsvgstr)
        

def main():
    """ {pyname} reads 

    Usage:
        {pyname} <imagefilenameglob> [<outputdir>] [--svgtemplate=<svgtemplate>] [--showimages [--showprocessteps]] [--method=<method>] [--scale=<scalepercent>]
        {pyname} -h
    
    Options:
        -h, --help            Show this help message
        --showimages          Shows images used during processes
        --showprocessteps     Shows detection steps
        --method=<method>     [default: shrinkoptcut] (shrinkoptcut|simpleocv)       
        --svgtemplate=<svgtemplate>  svgtemplate to use as target [default: svgtemplate.svg]
        --scale=<scalepercent>  [default: 100]
        
    Example:
        python {pyname} 01_downloaded_tweetimgs/2019_03*.jpg 02_shape_detected_imgs

    """
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    # print(main.__doc__)
    args = docopt.docopt(main.__doc__, help=False)
    # pprint.pprint(args)
    if args["<imagefilenameglob>"] is None or args["--help"] == True:
        print(main.__doc__)
        return
    if args["<outputdir>"] is None:
        args["<outputdir>"] = "."
    if not os.path.isfile(args["--svgtemplate"]) \
        or not os.path.isdir(args["<outputdir>"]):
        print(main.__doc__)
        return
    

    files = sorted(glob.glob(args["<imagefilenameglob>"]))
    if files is None or len(files) == 0:
        print("no files matched {}".format(args["<imagefilenameglob>"]))
        return
    
    process_files(files,args)

    
def test():
    srcfiles = [".\\01_downloaded_tweetimgs\\2020_01_09_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_01_29_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_05_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_11_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_15_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_18_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_19_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_02_20_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_03_16_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_03_21_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_03_29_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_03_31_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_04_21_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_04_21_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_04_23_tweet.jpg",
                ".\\01_downloaded_tweetimgs\\2019_05_01_tweet.jpg" ]

    sfiles = sorted(srcfiles)
    args = {}
    args["<outputdir>"] = "convtest"
    args["--showimages"] = True
    args["--showprocessteps"] = True
    args["--svgtemplate"] = "svgtemplate.svg"
    process_files(sfiles,args)


if __name__ == "__main__":
    main()