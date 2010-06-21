import cv, sys
from pyflann import *
from numpy import *

class TrackedRegion():
    xpos = 0
    ypos = 0
    size = 1
    reference = None

    def set_x(self, x):
        self.xpos = x

    def set_y(self, y):
        self.ypos = y

    def set_size(self, s):
        self.size = s

    def get_rectangle(self):
        return (self.xpos, self.ypos, self.size, self.size)

    def __repr__(self):
        return str(self.get_rectangle())

    def __init__(self, reference, xpos=0, ypos=0, size=1):
        self.reference = reference
        self.xpos = xpos
        self.ypos = ypos
        self.size = size
        

def main(argv=None):
    if argv is None:
        argv = sys.argv
    flann = FLANN()

    sample_dir = sys.argv[1]#'/home/jfsantos/Documents/ufsc/2010.1/ProcImagens/samples/'
    num_samples = 125

    # Loading images
    #print "Reference image:", 'samples/frame0.jpg'
    im = cv.LoadImageM(sample_dir + 'frame0.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Extracting SURF descriptors from reference image
    # and building the index for ANN search
    (k, d) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, 3000, 3, 1))
    d = array(d, dtype=float)
    print "shape(d) = ", shape(d)
    params = flann.build_index(d, target_precision=0.9)

    for x in k:
        pos = x[0]
        fsize = x[2]
        theta = x[3]
        cv.Circle(im, pos, fsize, 0)
        cv.Line(im, pos, (pos[0]+round(fsize*cos(theta)),pos[1]+fsize*sin(theta)),0)

    frames = [im]

    # Create window and controls
    tr = TrackedRegion(im)

    cv.NamedWindow("reference")
    cv.CreateTrackbar('xpos', 'reference', 0, im.width-1, tr.set_x)
    cv.CreateTrackbar('ypos', 'reference', 0, im.height-1, tr.set_y)
    cv.CreateTrackbar('size', 'reference', 0, 200, tr.set_size)

    # Show position of descriptors on reference image
    cv.ShowImage("reference", im)

    while True:
        if cv.WaitKey(100) == 32:
            cv.Rectangle(im, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 0)
            cv.ShowImage("reference", im)
            break

    print tr

    # Extracting descriptors from each target image and
    # calculating the distances to the nearest neighbors
    for x in range(1,num_samples):
        im2 = cv.LoadImageM(sample_dir + 'frame'+str(x)+'.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
        (k2, d2) = cv.ExtractSURF(im2, None, cv.CreateMemStorage(), (0, 3000, 3, 1))
        d2 = array(d2, dtype=float)
        result, dists = flann.nn_index(d2, 1, checks=params['checks'])
        for x in k2:
            pos = x[0]
            fsize = x[2]
            theta = radians(x[3])
            cv.Circle(im2, pos, fsize, 0)
            cv.Line(im2, pos, (pos[0]+round(fsize*cos(theta)),pos[1]+fsize*sin(theta)),0)
        frames.append(im2)

    cv.NamedWindow("target")

    while True:
        for img in frames:
            cv.ShowImage("target", img)
            if cv.WaitKey(100) == 27:
                break

if __name__ == "__main__":
    main()
