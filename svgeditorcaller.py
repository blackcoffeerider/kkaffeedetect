import subprocess
import psutil
import glob
import time

svgfilepattern = 'tweetsvgs/2019_*.svg'

for filename in glob.glob(svgfilepattern):
    doc = subprocess.Popen(["start","/wait",filename],shell=True)
    print(filename)
    while doc.poll() is None:
        time.sleep(0.1)
