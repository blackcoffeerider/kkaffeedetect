import csv
import collections
import requests
import os

def tweetcsvreader(filename='sampledata.csv', imgcol="BildLink", datecol="Datum"):

    rowidx = collections.OrderedDict()

    with open(filename,encoding="utf-8") as csvfile:
        kkcsvreader = csv.reader(csvfile, delimiter=',')
        firstrow = None
        for csvrowidx, row in enumerate(kkcsvreader):
            # print(', '.join(row))
            if firstrow is None:
                if imgcol in row and datecol in row:
                    firstrow = row
                    for idx, val in enumerate([f for f in row if f != ""]):
                        rowidx[val] = {"idx": idx, "val": val}
            else:
                rdict = collections.OrderedDict()
                for tkey in rowidx:
                    t = rowidx[tkey]
                    rdict[t["val"]] = row[t["idx"]]
                # print(rdict)
                yield (rdict, row, csvrowidx)

def get_tweetfilename(rdict, datecol="Datum"):
    return "{y}_{m}_{d}_tweet.jpg".format(y=rdict[datecol][6:10],m=rdict[datecol][3:5],d=rdict[datecol][0:2])
