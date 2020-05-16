import csv
import collections
import requests
import os
import sys
import docopt
import tweetcsvreader

def download_tweet(srcurl, outfilename):
    if not os.path.exists(outfilename):
        r = requests.get(srcurl)
        open(outfilename , 'wb').write(r.content)
    else:
        print("{} skipped - already exists".format(outfilename))



def get_date_int(datefield):
    dparts = datefield.split(".")
    return int(dparts[2])*10000+int(dparts[1])*100+int(dparts[0])


def main():
    """ {pyname} downloads all images mentioned from a CSV file withe the following header lines:
    IMGLinkURI has to be a link to an image you wish to download
    Date in "European format" (DD.MM.YYYYY)

    Usage:
        {pyname} <csvfilenameglob> [<outputdir>] [--imgurlhdr=<imgurlhdr>] [--datehdr=<datehdr>] [--mindate=<mindate>] [--maxdate=<maxdate>] [--svgtemplate=<svgtemplate>]
        {pyname} -h
    
    Options:
        -h, --help                   Show this help message
        --imgurlhdr=<imgurlhdr>      Header of the image column [default: BildLink] 
        --datehdr=<datehdr>          Header of the date column  [default: Datum]
        --mindate=<mindate>          Minimum date to process [default: 01.01.1900] (inclusive) 
        --maxdate=<maxdate>          Maximum date to process [default: 31.12.9999] (inclusive) 
    """
    main.__doc__ = main.__doc__.format(pyname=os.path.basename(sys.argv[0]))
    args = docopt.docopt(main.__doc__, help=False)
    # print(args)
    if args["<csvfilenameglob>"] is None or args["--help"] == True:
        print(main.__doc__)
        return
    if args["<outputdir>"] is None:
        args["<outputdir>"] = "."
    if not os.path.isfile(args["<csvfilenameglob>"]) \
        or not os.path.isdir(args["<outputdir>"]):
        print(main.__doc__)
        return
    # exit(0)

    for rdict, row, csvrowidx in tweetcsvreader.tweetcsvreader(filename=args["<csvfilenameglob>"],imgcol=args["--imgurlhdr"], datecol=args["--datehdr"]):
        if "http://" in rdict[args["--imgurlhdr"]]:
            if get_date_int(args["--mindate"]) <= get_date_int(rdict[args["--datehdr"]]) <= get_date_int(args["--maxdate"]):
                srcurl = rdict[args["--imgurlhdr"]]
                fname = os.path.join(args["<outputdir>"],tweetcsvreader.get_tweetfilename(rdict,datecol=args["--datehdr"]))
                print("{} => {}".format(srcurl,fname))
                download_tweet(srcurl, outfilename=fname)
            # else:
            #     print("skipping date: {}".format(rdict[args["--datehdr"]]))
        else:
            print("Problem line {}".format(csvrowidx))
            print(row)

if __name__ == "__main__":
    main()