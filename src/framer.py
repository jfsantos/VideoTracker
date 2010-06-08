import cv
import sys

samples_dir = '../../samples'

capture = cv.CaptureFromFile(sys.argv[1])

frameskip = sys.argv[2]

for k in range(cv.GetCaptureProperty(CV_CAP_PROP_FRAME_COUNT):
    img = cv.QueryFrame(capture)
    if not (k % frameskip):
        cv.SaveImage(samples_dir+"frame"+str(k)+".jpg", img)
