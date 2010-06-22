import cv
import sys

samples_dir = '../../samples'

file = sys.argv[1]
frameskip = 1 + int(sys.argv[2])
capture = cv.CaptureFromFile(file)
n_frames = cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT)

print("File: %s, frames: %d" % (file, n_frames))

for k in range(n_frames):
    try:
        img = cv.QueryFrame(capture)
        if not (k % frameskip):
            cv.SaveImage(samples_dir+"frame"+str(k)+".jpg", img)
    except(cv.error):
        print "End of file reached."
