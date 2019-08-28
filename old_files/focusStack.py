import numpy as np
#import mahotas as mh
import os
from scipy import ndimage
from scipy import misc
from Queue import Queue
from threading import Thread
from scipy.ndimage.filters import gaussian_filter

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getImages():
    global bw
    while not files.empty():
        fn = files.get()
        image = ndimage.imread(fn, flatten=bw)
        global im_arr
        global im
        im_arr[im] = image
        im += 1
        if not bw:
            image = rgb2gray(image)
        images.put(image)
 #       print "imported {}".format(fn)
        files.task_done()

def smoothBest(smooth=0):
    global best
    global im_arr
    global bw
    stack,h,w = im_arr.shape[:3]
    gauss = gaussian_filter(best, sigma=smooth)
    im_arr2 = np.copy(im_arr)
    if bw:
        im_arr2 = im_arr2.reshape((stack,h*w)) # image is now (stack, nr_pixels)
        im_arr2 = im_arr2.transpose() # image is now (nr_pixels, stack)
        r = im_arr2[np.arange(len(im_arr2)), gauss.ravel()] # Select the right pixel at each location
        r = r.reshape((h,w)) # reshape to get final result
    else:
        im_arr2 = im_arr2.reshape((stack,h*w,3)) # image is now (stack, nr_pixels)
        im_arr2 = im_arr2.transpose() # image is now (nr_pixels, stack)
        r = im_arr2[:, np.arange(h*w), gauss.ravel()] # Select the right pixel at each location
        r = r.transpose().reshape((h,w,3))
    return r

def focusImages(saveImage=False):
    global foc
    while foc < len(fns) - 1:
        image = images.get()
        sx = ndimage.filters.sobel(image,axis=0, mode='constant')
        sy = ndimage.filters.sobel(image,axis=1, mode='constant')
        sob = np.hypot(sx, sy)
#        sob = mh.sobel(image, just_filter=True)
#        sob = ndimage.filters.sobel(image, mode='mirror')
        global best
        better = np.greater(sob, comb)
        comb[better] = sob[better]
        best[better] = foc
        foc += 1
#        print "processed {}".format(foc)
    if saveImage:
#        global best
        global r
        r = smoothBest(0)
        misc.imsave('focus_stack.jpg', r)

def progReport():
    from time import sleep
    import sys
    global foc
    while foc < len(fns) - 1:
 #       global foc
        global im
        im_p = float(im)/float(len(fns))
        foc_p = float(foc)/float(len(fns))
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] %d%%\t[%-20s] %d%%" % ("="*int(20*im_p), 100*im_p, "="*int(20*foc_p), 100*foc_p))
        sys.stdout.flush()
        sleep(0.25)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%\t[%-20s] %d%%" % ("="*int(20), 100, "="*int(20), 100))

def smoothPreview():
    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.widgets import Slider, Button, RadioButtons
    fig, ax = plt.subplots()
    plt.subplots_adjust(left = .25, bottom=.25)
    r = smoothBest()
    l = plt.imshow(r, 'gray')
    smooth_loc = plt.axes([.25, .1, .65, .03])

    smooth = Slider(smooth_loc, 'Smoothness', 0, 100, valint=0)
    xticks = np.logspace(0, 7, 7, base=2)

    def update(val):
        v = smooth.val
        l.set_data(smoothBest(int(v)))
        fig.canvas.draw_idle()
    smooth.on_changed(update)

    plt.show()

global bw
bw = False

cwd = os.getcwd()
fns = os.listdir(cwd)
fns = [f for f in fns if f.endswith(('.jpg', '.png', '.JPG'))]

if "focus_stack.jpg" in fns:
    fns.remove("focus_stack.jpg")
if "best.png" in fns:
    fns.remove("best.png")

fns.sort()

files = Queue()
[files.put(f) for f in fns]

first = ndimage.imread(files.get(), flatten=bw)
first = first.astype('uint8')

global im_arr
if not bw:
    im_arr = np.zeros((len(fns), first.shape[0], first.shape[1], first.shape[2]), dtype= first.dtype)
else:
    im_arr = np.zeros((len(fns), first.shape[0], first.shape[1]), dtype= first.dtype)

images = Queue(maxsize=len(fns))
im_arr[0] = first
first = rgb2gray(first)
images.put(first)
im = 1
foc = 0
best = np.zeros(first.shape, dtype='uint16')
comb = np.zeros(first.shape, dtype='float64')

get = Thread(target = getImages)
focus1 = Thread(target = focusImages, args = (True,))
focus2 = Thread(target = focusImages, args = (True,))
focus3 = Thread(target = focusImages, args = (True,))
get.setDaemon(True)
focus1.setDaemon(True)
focus2.setDaemon(True)
focus3.setDaemon(True)
get.start()
focus2.start()
focus3.start()
focus1.start()
progress = Thread(target=progReport)
progress.setDaemon(True)
progress.start()
