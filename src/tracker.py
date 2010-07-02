import cv, sys
from pyflann import *
from numpy import *

class TrackedRegion():
    xpos = 0
    ypos = 0
    size = 1
    reference = None

    # Setters added so they can be used as callbacks by HighGUI
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
        mask = zeros([height, width], dtype=int32)
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

def draw_features(im, k):
    for x in k:
        pos = x[0]
        fsize = x[2]
        theta = radians(x[3])
        cv.Circle(im, pos, fsize, 0)
        cv.Line(im, pos, (pos[0]+round(fsize*cos(theta)),pos[1]+fsize*sin(theta)),0)

def draw_tracked_region(im, tr):
     cv.Rectangle(im, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 255, thickness=3)

def get_tracked_region(im):
    tr = TrackedRegion(im)

    cv.NamedWindow("reference")
    cv.CreateTrackbar('xpos', 'reference', 0, im.width-1, tr.set_x)
    cv.CreateTrackbar('ypos', 'reference', 0, im.height-1, tr.set_y)
    cv.CreateTrackbar('size', 'reference', 0, im.width-1, tr.set_size)

    # Show position of descriptors on reference image
    cv.ShowImage("reference", im)

    # Selecting tracked region
    while True:
        key_pressed = cv.WaitKey(100)
        if key_pressed == 32:
            cv.Rectangle(im, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 255, thickness=3)
            cv.DestroyWindow("reference")
            break
        elif key_pressed == 27:
            cv.DestroyAllWindows()
            cv.WaitKey(100)
            return
        else:
            im_copy = cv.CreateMat(im.height, im.width, cv.CV_8UC1)
            cv.Copy(im, im_copy)
            cv.Rectangle(im_copy, (tr.xpos, tr.ypos), (tr.xpos+tr.size, tr.ypos+tr.size), 255, thickness=3)
            cv.ShowImage("reference", im_copy)

    return tr

def calc_centroid(keypoints):
    x = [k[0][0] for k in keypoints]
    y = [k[0][1] for k in keypoints]
    n = len(keypoints)
    return (sum(x)/n, sum(y)/n)

def main(argv=None):
    if argv is None:
        argv = sys.argv
    flann = FLANN()

    try:
        sample_dir = argv[1]
        sample_name = argv[2]
        start_frame = int(argv[3])
        end_frame = int(argv[4])
        hessian_threshold = int(argv[5])
    except(IndexError):
        print "Argument error\n Usage: python tracker.py sample_dir sample_filename start_frame end_frame hessian_threshold"
        return

    # Loading first frame
    im = cv.LoadImageM(sample_dir + sample_name + str(start_frame) + '.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Extracting SURF descriptors from reference image
    # to make selecting the tracked region easier
    (k, d) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (0, hessian_threshold, 3, 1))

    draw_features(im, k)

    frames = [im]

    # Create window and controls
    tr = get_tracked_region(im)
    
    # Extracting descriptors from each target image and
    # calculating the distances to the nearest neighbors
    # Tracked region must be updated in each step
    for x in range(start_frame,end_frame):
        im2 = cv.LoadImageM(sample_dir + sample_name +str(x)+'.jpg', cv.CV_LOAD_IMAGE_GRAYSCALE)
        # Creating mask for extracting features from new image
        mask = tr.get_mask()
        (k2, d2) = cv.ExtractSURF(im2, mask, cv.CreateMemStorage(), (0, hessian_threshold, 3, 1))
        #(k2, d2) = cv.ExtractSURF(im2, None, cv.CreateMemStorage(), (0, hessian_threshold, 3, 1))
        d2 = array(d2, dtype=float32)
        result = None
        dists = None
        if x == start_frame:
            params = flann.build_index(d2, target_precision=0.9)
            tr.size = 1.1*tr.size
        else:
            if len(d2) > 0:
                result, dists = flann.nn_index(d2, 1, checks=params['checks'])
                # Creating full neighbor table
                neighbors = []
                nearest_neighbors = []
                for n in range(len(d2)):
                    neighbors.append((d2[n], k2[n], result[n], dists[n]))
                # Removing not-nearest neighbors
                for k in range(len(d2)):
                    k_neighbors = filter(lambda x: x[2] == k, neighbors)
                    num_neighbors = len(k_neighbors)
                    if num_neighbors >= 1:
                        if num_neighbors > 1:
                            nearest = reduce(lambda x,y: x if x[3] < y[3] else y, k_neighbors)
                        else:
                            nearest = k_neighbors[0]
                        nearest_neighbors.append(nearest)
                nearest_keypoints =  [kp[1] for kp in nearest_neighbors]
                if len(nearest_keypoints) > 0:
                    centroid = calc_centroid(nearest_keypoints)
                    # print "centroid: ", centroid
                    tr.xpos = centroid[0] - tr.size/2
                    tr.ypos = centroid[1] - tr.size/2
                #draw_features(im2, nearest_keypoints)
                draw_tracked_region(im2, tr)
                #print "Frame %d had %d feature(s)!" % (x, len(d2))
            else:
                #print "Frame %d had no features!" % x
                pass
        
        frames.append((im2, k2, d2, result, dists))

    cv.NamedWindow("target")

    esc_pressed = False
    
    while True:
        for frame in frames:
            cv.ShowImage("target", frame[0])
            if cv.WaitKey(50) == 27:
                esc_pressed = True
                break
        if esc_pressed:
            cv.DestroyAllWindows()
            break
    return frames, params

if __name__ == "__main__":
    main()
