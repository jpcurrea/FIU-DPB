# ! /usr/bin/env python

# make a small target tracker program - try with pylab

from sys import argv
import os
import matplotlib
from matplotlib import pyplot as plt
import PIL
# matplotlib.use('gtk3agg')
import pylab
import numpy as np
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle

from numpy import load, save, zeros

cwd = os.getcwd()+"/"


class tracker_window():

    def __init__(self, dirname=cwd):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0

        # markers and data file
        self.num_markers = 10
        self.range_markers = np.array(range(self.num_markers))
        self.curr_marker = 0
        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                       (0, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0)]
        self.marker_view = 0
        self.fn = self.dirname + 'data.npy'
        if os.path.isfile(self.fn):
            self.markers = load(self.fn)
            # remove markers for files that have been deleted
            if os.path.isfile(self.dirname + 'order.npy'):
                self.old_order = np.load(self.dirname + 'order.npy')
                self.old_order.sort()
                keepers = [fn in self.filenames for fn in self.old_order]
                self.old_order = self.old_order[keepers]
                self.markers = self.markers[:, keepers]
            else:
                self.old_order = np.array([])
            if self.markers.shape[1] < self.num_frames:
                old_files = [fn in self.old_order for fn in
                             self.filenames]
                z = zeros((self.num_markers, self.num_frames, 2))
                z[:, old_files] = self.markers
                self.markers = z
        else:
            self.markers = zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1
            self.imarkers = zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1

        np.save(self.dirname + "./order.npy", self.filenames)
        self.data_changed = False

        # the figure
        self.load_image()
        figsize = self.image.shape[1]/90, self.image.shape[0]/90
        self.figure = plt.figure(1, figsize=(
            figsize[0]+1, figsize[1]+2), dpi=90)
        xmarg, ymarg = .1, .1
        axim = plt.axes([xmarg, ymarg, 1-2*xmarg, 1-2*ymarg])
        self.implot = plt.imshow(self.image, cmap='gray')
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        for i in range(self.num_markers):
            plt.plot([-1], [-1], 'o', mfc=self.colors[i])
        plt.plot([-1], [-1], 'o', mfc=(1, 1, 1))
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]

        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider for selecting frames
        axframe = plt.axes([0.5-.65/2, 0.04, 0.65, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)

        # the slider for selecting min intensity bval
        self.minframe = plt.axes([0.1, .97, 0.1, 0.02], axisbg='k')
        self.minframe.plot([0], [.5], 'ko', mec='w')
        self.min = Slider(self.minframe, 'min', 0, 255,
                          valinit=0, valfmt='%1.2f', color='w')
        self.min.on_changed(self.update_sliders)
        # slider for selecting mid val
        self.midframe = plt.axes([0.1, .94, 0.1, 0.02], axisbg='0.4')
        self.midframe.plot([127], [.5], 'o', color='0.5')
        self.mid = Slider(self.midframe, 'mid', 0, 255,
                          valinit=127, valfmt='%1.2f', color='0.6')
        self.mid.on_changed(self.update_sliders)
        # slider for max val
        self.maxframe = plt.axes([0.1, .91, 0.1, 0.02], axisbg='k')
        self.maxframe.plot([255], [.5], 'wo')
        self.max = Slider(self.maxframe, 'max', 0, 255,
                          valinit=255, valfmt='%1.2f', color='w')
        self.max.on_changed(self.update_sliders)
        # set a colorbar
        axframe = plt.axes([0.1, .88, 0.1, 0.02])
        axframe.set_xticks([])
        axframe.set_yticks([])
        c = plt.colorbar(orientation='horizontal', fraction=1)
        c.set_ticks([])

        # buttons for slider contrast
        self.bresetframe = plt.axes([.25, .95, .06, .04])
        self.breverseframe = plt.axes([.25, .89, .06, .04])
        self.breset = Button(self.bresetframe, 'reset')
        self.breverse = Button(self.breverseframe, 'reverse')
        self.breset.on_clicked(self.reset_contrast)
        self.breverse.on_clicked(self.reverse_contrast)

        # radio buttons for marker selection
        self.radioframe = plt.axes([0.02, .60, 0.10, 0.20], frameon=False)
        self.radioframe.set_title("marker", ha='right')
        labs = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
        self.radiobuttons = RadioButtons(
            self.radioframe, labs, activecolor='k')
        # color the buttons
        for i in np.arange(len(labs)):
            self.radiobuttons.labels[i].set_color(self.colors[i])
        self.radiobuttons.on_clicked(self.marker_button)

        # radio buttons for selecting marker view
        self.visframe = plt.axes([0.02, .40, 0.10, 0.15], frameon=False)
        self.visframe.set_title("visible", ha='right')
        self.visbuttons = RadioButtons(
            self.visframe, ['curr', 'all'], activecolor='k')
        self.visbuttons.on_clicked(self.vis_button)

        # radio buttons for selecting view
        self.viewframe = plt.axes([0.02, .30, 0.10, 0.15], frameon=False)
        self.viewbuttons = RadioButtons(
            self.viewframe, ['frame', 'all'], activecolor='k')
        self.viewbuttons.on_clicked(self.view_button)

        # connect some keys
        self.cidk = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

    def load_filenames(self):
        ls = os.listdir(self.dirname)
#        print ls
        self.filenames = [self.dirname + f for f in ls if f.endswith(
            ('.png', '.jpg', '.BMP', '.JPG', '.TIF', '.TIFF'))]
        self.filenames.sort()

    def load_image(self):
        print self.curr_frame_index
#        print len(self.filenames)
        # self.image = plt.imread(self.filenames[self.curr_frame_index])
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first the image
        # self.im = pylab.interp(self.image, pylab.array([self.min.val,self.mid.val,self.max.val]), pylab.array([0,127,255]))
        # self.im = pylab.array(self.im, dtype=pylab.uint8)
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # then the markers
        for i in range(self.num_markers):
            self.figure.axes[0].lines[i].set_xdata(
                self.markers[i, self.curr_frame_index, 0:1])
            self.figure.axes[0].lines[i].set_ydata(
                self.markers[i, self.curr_frame_index, 1:2])
        middle = self.markers[:, self.curr_frame_index]
        middle = middle[middle.sum(1) > 0].mean(0)
        self.figure.axes[0].lines[i+1].set_xdata(
            middle[0])
        self.figure.axes[0].lines[i+1].set_ydata(
            middle[1])
        self.minframe.lines[0].set_xdata(self.min.val)
        self.midframe.lines[0].set_xdata(self.mid.val)
        self.maxframe.lines[0].set_xdata(self.max.val)
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        plt.draw()

    def change_frame(self, new_frame):
        print('change_frame {} {}'.format(new_frame, int(new_frame)))
        self.curr_frame_index = int(new_frame)-1
        self.load_image()
        # self.xlim = self.figure.axes[0].get_xlim()
        # self.ylim = self.figure.axes[0].get_ylim()
        self.show_image()
        # self.figure.axes[0].set_xlim(*self.xlim)
        # self.figure.axes[0].set_ylim(*self.ylim)
        if self.data_changed:
            self.save_data()
            self.data_changed = False

    def nudge(self, direction):
        self.markers[self.curr_marker,
                     self.curr_frame_index, 0] += direction.real
        self.markers[self.curr_marker,
                     self.curr_frame_index, 1] += direction.imag
        self.show_image()
        # self.change_frame(mod(self.curr_frame, self.num_frames))
        self.data_changed = True

    def on_key_release(self, event):
        # frame change
        if event.key in ("pageup", "alt+v", "alt+tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index, self.num_frames))
        elif event.key in ("pagedown", "alt+c", "tab"):
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 2, self.num_frames))
            print self.curr_frame_index
        elif event.key == "alt+pageup":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index - 9, self.num_frames))
        elif event.key == "alt+pagedown":
            self.curr_frame.set_val(
                np.mod(self.curr_frame_index + 11, self.num_frames))
        elif event.key == "home":
            self.curr_frame.set_val(1)
        elif event.key == "end":
            self.curr_frame.set_val(self.num_frames)

        # marker change
        elif event.key in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
            # white out the old marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (1.0, 1.0, 1.0, 1.0))
            # update the marker index
            if int(event.key) > 0:
                self.curr_marker = int(event.key) - 1
            else:
                self.curr_marker = 9
            # place the marker
            self.markers[self.curr_marker, self.curr_frame_index, :] = [
                event.xdata, event.ydata]
            # black in the new marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (0.0, 0.0, 0.0, 1.0))
            self.data_changed = True
            self.show_image()

        elif event.key == " ":
            self.data_changed = True
            self.markers[self.curr_marker, self.curr_frame_index, :] = [
                event.xdata, event.ydata]
            self.show_image()

        elif event.key in ('ctrl+1', 'ctrl+2', 'ctrl+3', 'ctrl+4', 'ctrl+5', 'ctrl+6'):
            # white out the old marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (1.0, 1.0, 1.0, 1.0))
            # update the marker index
            self.curr_marker = int(event.key[-1]) - 1
            # place the marker
            self.markers[self.curr_marker, self.curr_frame_index, :] = [-1, -1]
            # black in the new marker
            self.radiobuttons.circles[self.curr_marker].set_facecolor(
                (0.0, 0.0, 0.0, 1.0))
            self.data_changed = True
            self.show_image()

        elif event.key == "backspace":
            self.data_changed = True
            self.markers[self.curr_marker, self.curr_frame_index, :] = [-1, -1]
            self.show_image()

        # marker move
        elif event.key == "left":
            self.nudge(-1)
        elif event.key == "right":
            self.nudge(1)
        elif event.key == "up":
            self.nudge(-1j)
        elif event.key == "down":
            self.nudge(1j)
        elif event.key == "alt+left":
            self.nudge(-10)
        elif event.key == "alt+right":
            self.nudge(10)
        elif event.key == "alt+up":
            self.nudge(-10j)
        elif event.key == "alt+down":
            self.nudge(10j)

    def update_sliders(self, val):
        self.show_image()

    def reset_contrast(self, event=None):
        self.min.set_val(0)
        self.mid.set_val(127)
        self.max.set_val(255)

    def reverse_contrast(self, event=None):
        self.min.set_val(255)
        self.mid.set_val(127)
        self.max.set_val(0)

    def on_mouse_release(self, event):
        self.markers[self.curr_marker, self.curr_frame_index, :] = [
            event.xdata, event.ydata]
        # self.change_frame(0)

    def marker_button(self, lab):
        self.curr_marker = int(lab)-1
        self.show_image()

    def vis_button(self, lab):
        self.show_image()
        self.vis = lab

    def view_button(self, lab):
        self.show_image()
        self.view = lab

    def save_data(self):
        print('save')
        save(self.fn, self.markers)


if __name__ == '__main__':
   # dirname = '/home/jamie/data/long_legged_flies/TS3-D9-LEFT_000000/'
    #    print ('__main__')
    #    dirname = '07_17_14_00_cr/'

    # if len(argv)>1:
    #     dirname = argv[1]   #first arg is the dirname
    # else:
    #     dirname = './'
    import os
    dirname = os.getcwd()


t = tracker_window()


plt.show()
