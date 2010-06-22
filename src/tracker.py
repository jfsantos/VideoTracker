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

    def get_mask(self):
        width = self.reference.width
        height = self.reference.height
        mask = zeros([height, width], dtype=int)
        cvmask = cv.CreateMat(height, width, cv.CV_8UC1)
        mask[self.ypos:self.ypos+self.size,self.xpos:self.xpos+self.size] = 255
        cv.Convert(mask, cvmask)
        return cvmask

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

    sample_dir = argv[1]#'/home/jfsantos/Documents/ufsc/2010.1/ProcImagens/samples/'
    num_samples = 125

    # Loading images
    #print "Reference image:", 'samples/frame0.jpg'
    im = cv.LoadImageM(sample_dir + 'samplesframe0.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Extracting SURF descriptors from reference image
    # and building the index for ANN search
    (k, d) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, 500, 3, 1))
    d = array(d, dtype=float)
    print "shape(d) = ", shape(d)
    # params = flann.build_index(d, target_precision=0.9)

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
    cv.NamedWindow("mask")
    cv.CreateTrackbar('xpos', 'reference', 0, im.width-1, tr.set_x)
    cv.CreateTrackbar('ypos', 'reference', 0, im.height-1, tr.set_y)
    cv.CreateTrackbar('size', 'reference', 0, im.width-1, tr.set_size)

    # Show position of descriptors on reference image
    cv.ShowImage("reference", im)

    while True:
        key_pressed = cv.WaitKey(100)
        if key_pressed == 32:
            cv.Rectangle(im, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 0)
            cv.DestroyWindow("reference")
            break
        elif key_pressed == 27:
            cv.DestroyAllWindows()
            cv.WaitKey(100)
            return
        else:
            im_copy = cv.CreateMat(im.height, im.width, cv.CV_8UC1)
            cv.Copy(im, im_copy)
            cv.Rectangle(im_copy, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 0)
            cv.ShowImage("reference", im_copy)
            cv.ShowImage("mask", tr.get_mask())

    print tr

    # Extracting descriptors from each target image and
    # calculating the distances to the nearest neighbors
    for x in range(0,num_samples):
        print "x: ", x
        im2 = cv.LoadImageM(sample_dir + 'samplesframe'+str(x)+'.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
        # Creating mask for extracting features from new image
        mask = tr.get_mask()
        print mask
        (k2, d2) = cv.ExtractSURF(im2, mask, cv.CreateMemStorage(), (0, 500, 3, 1))
        print "shape(d2): ", shape(d2)
        d2 = array(d2, dtype=float)
        if x == 0:
            params = flann.build_index(d2, target_precision=0.9)
        else:
            if len(d2) > 0:
                result, dists = flann.nn_index(d2, 1, checks=params['checks'])
                for k in k2:
                    pos = k[0]
                    fsize = k[2]
                    theta = radians(k[3])
                    cv.Circle(im2, pos, fsize, 0)
                    cv.Line(im2, pos, (pos[0]+round(fsize*cos(theta)),pos[1]+fsize*sin(theta)),0)
        frames.append(im2)

    cv.NamedWindow("target")

    esc_pressed = False
    
    while True:
        for img in frames:
            cv.ShowImage("target", img)
            if cv.WaitKey(100) == 27:
                esc_pressed = True
                break
        if esc_pressed:
            cv.DestroyAllWindows()
            break
    return

if __name__ == "__main__":
    main()
