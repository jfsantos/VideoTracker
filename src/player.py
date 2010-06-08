import cv
import sys

cv.NamedWindow("camera", 1)

capture = cv.CaptureFromFile(sys.argv[1])

while True:
    img = cv.QueryFrame(capture)
    cv.ShowImage("camera", img)
    if cv.WaitKey(20) == 27:
        break
