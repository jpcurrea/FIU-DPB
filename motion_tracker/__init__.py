from scipy import spatial
from collections import namedtuple
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle
# from MinimumBoundingBox import MinimumBoundingBox
import matplotlib
from matplotlib import pyplot as plt
import math
import numpy as np
import os
import PIL
from scipy import interpolate, ndimage, spatial, optimize
from skvideo import io
from skimage.feature import peak_local_max
from sklearn import cluster
import subprocess
import sys

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
# from points_GUI import *

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

colors = [
    'tab:red',
    'tab:green',
    'tab:blue',
    'tab:orange',
    'tab:purple',
]


def print_progress(part, whole):
    prop = float(part)/float(whole)
    sys.stdout.write('\r')
    sys.stdout.write("[%-20s] %d%%" % ("="*int(20*prop), 100*prop))
    sys.stdout.flush()


filetypes = [
    ('mpeg videos', '*.mpg *.mpeg *.mp4'),
    ('avi videos', '*.avi'),
    ('quicktime videos', '*.mov *.qt'),
    ('h264 videos', '*.h264'),
    ('all files', '*')
]

# format the filetypes for the pyqt file dialog box
ftypes = []
for (fname, ftype) in filetypes:
    ftypes += [f"{fname} ({ftype})"]
ftypes = ";;".join(ftypes)


def rotate(arr, theta, axis=0):
    """Generate a rotation matrix and rotate input array along a single axis. 
    If only one axis, it will rotate counter-clockwise"""
    # theta = -theta
    rot_matrix = np.array(
        [[np.cos(theta), -np.sin(theta)],
         [np.sin(theta), np.cos(theta)]])
    nx, ny, = np.dot(arr, rot_matrix).T
    nx = np.squeeze(nx)
    ny = np.squeeze(ny)
    return np.array([nx, ny]).T


def save_thumbnail(fn, new_fn=None):
    if new_fn is None:
        ftype = fn.split(".")[-1]
        ftype = f".{ftype}"
        new_fn = fn.replace(ftype, "_thumbnail.jpg")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-nostats",
        "-i", fn,
        "-vf", "select=gte(n\,100)",
        "-vframes", "1", new_fn
    ]
    subprocess.call(cmd)


def unit_vector(pt0, pt1):
    # returns an unit vector that points in the direction of pt0 to pt1
    dis_0_to_1 = math.sqrt((pt0[0] - pt1[0])**2 + (pt0[1] - pt1[1])**2)
    return (pt1[0] - pt0[0]) / dis_0_to_1, \
           (pt1[1] - pt0[1]) / dis_0_to_1


def orthogonal_vector(vector):
    # from vector returns a orthogonal/perpendicular vector of equal length
    return -1 * vector[1], vector[0]


def bounding_area(index, hull):
    unit_vector_p = unit_vector(hull[index], hull[index+1])
    unit_vector_o = orthogonal_vector(unit_vector_p)

    dis_p = tuple(np.dot(unit_vector_p, pt) for pt in hull)
    dis_o = tuple(np.dot(unit_vector_o, pt) for pt in hull)

    min_p = min(dis_p)
    min_o = min(dis_o)
    len_p = max(dis_p) - min_p
    len_o = max(dis_o) - min_o

    return {'area': len_p * len_o,
            'length_parallel': len_p,
            'length_orthogonal': len_o,
            'rectangle_center': (min_p + len_p / 2, min_o + len_o / 2),
            'unit_vector': unit_vector_p,
            }


def to_xy_coordinates(unit_vector_angle, point):
    # returns converted unit vector coordinates in x, y coordinates
    angle_orthogonal = unit_vector_angle + math.pi / 2
    return point[0] * math.cos(unit_vector_angle) + point[1] * math.cos(angle_orthogonal), \
           point[0] * math.sin(unit_vector_angle) + point[1] * math.sin(angle_orthogonal)


def rotate_points(center_of_rotation, angle, points):
    # Requires: center_of_rotation to be a 2d vector. ex: (1.56, -23.4)
    #           angle to be in radians
    #           points to be a list or tuple of points. ex: ((1.56, -23.4), (1.56, -23.4))
    # Effects: rotates a point cloud around the center_of_rotation point by angle
    rot_points = []
    ang = []
    for pt in points:
        diff = tuple([pt[d] - center_of_rotation[d] for d in range(2)])
        diff_angle = math.atan2(diff[1], diff[0]) + angle
        ang.append(diff_angle)
        diff_length = math.sqrt(sum([d**2 for d in diff]))
        rot_points.append((center_of_rotation[0] + diff_length * math.cos(diff_angle),
                           center_of_rotation[1] + diff_length * math.sin(diff_angle)))

    return rot_points


def rectangle_corners(rectangle):
    # Requires: the output of mon_bounding_rectangle
    # Effects: returns the corner locations of the bounding rectangle
    corner_points = []
    for i1 in (.5, -.5):
        for i2 in (i1, -1 * i1):
            corner_points.append((rectangle['rectangle_center'][0] + i1 * rectangle['length_parallel'],
                            rectangle['rectangle_center'][1] + i2 * rectangle['length_orthogonal']))

    return rotate_points(rectangle['rectangle_center'], rectangle['unit_vector_angle'], corner_points)


BoundingBox = namedtuple('BoundingBox', ('area',
                                         'length_parallel',
                                         'length_orthogonal',
                                         'rectangle_center',
                                         'unit_vector',
                                         'unit_vector_angle',
                                         'corner_points'
                                        )
)


# use this function to find the listed properties of the minimum bounding box of a point cloud
def MinimumBoundingBox(points):
    # Requires: points to be a list or tuple of 2D points. ex: ((5, 2), (3, 4), (6, 8))
    #           needs to be more than 2 points
    # Effects:  returns a namedtuple that contains:
    #               area: area of the rectangle
    #               length_parallel: length of the side that is parallel to unit_vector
    #               length_orthogonal: length of the side that is orthogonal to unit_vector
    #               rectangle_center: coordinates of the rectangle center
    #                   (use rectangle_corners to get the corner points of the rectangle)
    #               unit_vector: direction of the length_parallel side. RADIANS
    #                   (it's orthogonal vector can be found with the orthogonal_vector function
    #               unit_vector_angle: angle of the unit vector
    #               corner_points: set that contains the corners of the rectangle

    if len(points) <= 2: raise ValueError('More than two points required.')

    hull_ordered = [points[index] for index in spatial.ConvexHull(points).vertices]
    hull_ordered.append(hull_ordered[0])
    hull_ordered = tuple(hull_ordered)

    min_rectangle = bounding_area(0, hull_ordered)
    for i in range(1, len(hull_ordered)-1):
        rectangle = bounding_area(i, hull_ordered)
        if rectangle['area'] < min_rectangle['area']:
            min_rectangle = rectangle

    min_rectangle['unit_vector_angle'] = math.atan2(min_rectangle['unit_vector'][1], min_rectangle['unit_vector'][0])
    min_rectangle['rectangle_center'] = to_xy_coordinates(min_rectangle['unit_vector_angle'], min_rectangle['rectangle_center'])

    # this is ugly but a quick hack and is being changed in the speedup branch
    return BoundingBox(
        area = min_rectangle['area'],
        length_parallel = min_rectangle['length_parallel'],
        length_orthogonal = min_rectangle['length_orthogonal'],
        rectangle_center = min_rectangle['rectangle_center'],
        unit_vector = min_rectangle['unit_vector'],
        unit_vector_angle = min_rectangle['unit_vector_angle'],
        corner_points = set(rectangle_corners(min_rectangle))
    )


class tracker_window():

    def __init__(self, dirname="./", num_markers=10, fn='data.npy'):
        # m.pyplot.ion()
        self.dirname = dirname
        self.load_filenames()
        self.num_frames = len(self.filenames)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0

        # markers and data file
        self.num_markers = num_markers
        self.range_markers = np.arange(self.num_markers)
        self.curr_marker = 0

        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1),
                  (0, 1, 1), (.5, 0, 0), (0, .5, 0), (0, 0, .5), (.5, .5, 0)]
        self.colors = []
        for x in self.range_markers:
            self.colors += [colors[x % len(colors)]]

        self.marker_view = 0
        self.fn = os.path.join(self.dirname, fn)
        if os.path.isfile(self.fn):
            self.markers = np.load(self.fn)
            # remove markers for files that have been deleted
            if os.path.isfile(os.path.join(self.dirname, 'order.npy')):
                self.old_order = np.load(
                    os.path.join(self.dirname, 'order.npy'))
                self.old_order.sort()
                keepers = [fn in self.filenames for fn in self.old_order]
                self.old_order = self.old_order[keepers]
                self.markers = self.markers[:, keepers]
            else:
                self.old_order = np.array([])
            if self.markers.shape[1] < self.num_frames:
                old_files = [fn in self.old_order for fn in
                             self.filenames]
                z = np.zeros((self.num_markers, self.num_frames, 2))
                z[:, old_files] = self.markers
                self.markers = z
        else:
            self.markers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1
            self.imarkers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1

        np.save(os.path.join(self.dirname, "order.npy"), self.filenames)
        self.data_changed = False

        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 8
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height

        # self.figure = plt.figure(1, figsize=(
        #     figsize[0]+1, figsize[1]+2), dpi=90)
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        # xmarg, ymarg = .2, .1
        fig_left, fig_bottom, fig_width, fig_height = .15, .1, .75, .85
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='gray')
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        for i in range(self.num_markers):
            plt.plot([-1], [-1], 'o', mfc=self.colors[i])
        self.figure.axes[0].set_xlim(*self.xlim)
        self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]

        # title
        self.title = self.figure.suptitle(
            '%d - %s' % (self.curr_frame_index + 1, self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        # the slider ing frames
        # axframe = plt.axes([0.5-.65/2, 0.04, 0.65, 0.02])
        axframe = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            axframe, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)

        # radio buttons for marker selection
        self.radioframe = plt.axes([0.04, .60, 0.10, 0.20], frameon=False)
        self.radioframe.set_title("marker", ha='right')
        labs = [str(x + 1) for x in range(self.num_markers)]
        labs = tuple(labs)
        self.radiobuttons = RadioButtons(
            self.radioframe, labs, activecolor='k')
        # color the buttons
        for i in np.arange(len(labs)):
            self.radiobuttons.labels[i].set_color(self.colors[i])
        self.radiobuttons.on_clicked(self.marker_button)

        # connect some keys
        self.cidk = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        # self.cidm = self.figure.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        # self.cidm = self.figure.canvas.mpl_connect('', self.on_mouse_release)
        # self.figure.canvas.toolbar.home = self.show_image

        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data

        # make a list of objects and filenames to save
        self.objects_to_save = {}
        self.objects_to_save[self.fn] = self.markers

        # plt.tight_layout()

    def load_filenames(self):
        ls = os.listdir(self.dirname)
#        print(ls)
        self.filenames = [os.path.join(self.dirname, f) for f in ls if f.endswith(
            ('.png', '.jpg', '.BMP', '.JPG', '.TIF', '.TIFF'))]
        self.filenames.sort()

    def load_image(self):
        print(self.curr_frame_index)
#        print(len(self.filenames))
        # self.image = plt.imread(self.filenames[self.curr_frame_index])
        self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plotthe image
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[:,
                                                                  self.curr_frame_index - 1]
        # then plot the markers
        for i in range(self.num_markers):
            self.figure.axes[0].lines[i].set_xdata(
                self.markers[i, self.curr_frame_index, 0:1])
            self.figure.axes[0].lines[i].set_ydata(
                self.markers[i, self.curr_frame_index, 1:2])
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
            print(self.curr_frame_index)
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
            # elif int(event.key) in self.range_markers:
            if self.curr_marker + 1 <= self.markers.shape[0]:
                # white out the old marker
                self.radiobuttons.circles[self.curr_marker].set_facecolor(
                    (1.0, 1.0, 1.0, 1.0))

            # update the marker index
            if int(event.key) > 0:
                self.curr_marker = int(event.key) - 1
            else:
                self.curr_marker = 9

            if self.curr_marker + 1 > self.markers.shape[0]:
                print(
                    f"marker = {self.curr_marker} is out of the range of {self.markers.shape[0]} defined markers.")
            else:
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

    def on_mouse_release(self, event):
        self.markers[self.curr_marker, self.curr_frame_index, :] = [
            event.xdata, event.ydata]
        # self.change_frame(0)

    def marker_button(self, lab):
        self.curr_marker = int(lab)-1
        self.show_image()

    def save_data(self):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)


class distance_calibration_GUI(tracker_window):
    """Special tracker window for selecting a size reference and exporting 
    the pixel to distance conversion per image.
    """

    def __init__(self, dirname, scale=10):
        super().__init__(dirname, num_markers=2, fn='calibration_markers.npy')
        self.scale = scale      # these will be in centimeters, but could be whatever
        self.scale_fn = os.path.join(self.dirname, "calibration_scales.npy")
        self.lengths_fn = os.path.join(self.dirname, "calibration_lengths.npy")
        if os.path.exists(self.scale_fn) is False:
            self.scales = np.zeros(self.markers.shape[1])
        else:
            self.scales = np.load(self.scale_fn)
            # remove markers for files that have been deleted
            assert len(self.scales) == len(self.filenames), (
                f"{self.scale_fn} should have the same number of entries as"
                f" image files in {self.dirname}. Instead there are {len(self.filenames)} images"
                f" and {len(self.scales)} entries in {self.scale_fn}.")
        if os.path.exists(self.lengths_fn) is False:
            self.lengths = np.zeros(self.markers.shape[1])
        else:
            self.lengths = np.load(self.lengths_fn)
            # remove markers for files that have been deleted
            assert len(self.lengths) == len(self.filenames), (
                f"{self.lengths_fn} should have the same number of entries as"
                f" image files in {self.dirname}. Instead there are {len(self.filenames)} images"
                f" and {len(self.lengths)} entries in {self.lengths_fn}.")

        # make input box for the distance used for calibration
        self.input_frame = plt.axes([0.04, .40, 0.10, 0.10], frameon=False)
        self.input_frame.set_title("Scale", ha='right')
        self.input_box = TextBox(self.input_frame,
                                 'cm', initial=str(self.scale),
                                 label_pad=-.65)
        self.input_box.on_submit(self.set_scale)

        # add to objects_to_save
        self.objects_to_save[self.scale_fn] = self.scales
        self.objects_to_save[self.lengths_fn] = self.lengths

    def set_scale(self, text):
        num = float(text)
        self.scale = num

    def show_image(self, *args):
        print('show_image')
        # first plot the image
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[:,
                                                                  self.curr_frame_index - 1]
        # then plot the markers
        for i in range(self.num_markers):
            self.figure.axes[0].lines[i].set_xdata(
                self.markers[i, self.curr_frame_index, 0:1])
            self.figure.axes[0].lines[i].set_ydata(
                self.markers[i, self.curr_frame_index, 1:2])
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        self.scales[self.curr_frame_index] = self.scale
        # calculate the pixel length
        current_marker = self.markers[:, self.curr_frame_index]
        positive_vals = (current_marker > 0).mean(1) == 1
        if positive_vals.sum() > 1:
            dist_tree = spatial.KDTree(current_marker[positive_vals])
            dists, inds = dist_tree.query(current_marker, k=2)
            self.lengths[self.curr_frame_index] = self.scale / \
                dists[:, 1].mean()
        self.data_changed = True
        plt.draw()


class ROI_GUI(tracker_window):
    """Special window for selecting points of interest with an optional hidden
    radius. The hidden zone is used to remove errors in the motion tracker,
    assuming that trajectories ocurring only in these areas are not of interest.
    """

    def __init__(self, dirname="./", num_markers=5, radius=10, pixel_length=1):
        super().__init__(dirname, num_markers=num_markers,
                         fn='ROI_markers.npy')
        self.pixel_length = pixel_length
        assert isinstance(pixel_length, (int, float, list, tuple, np.ndarray)), (
            f"Pixel length variable, type = {type(self.pixel_length)}, is "
            "not understood.")
        self.radius = radius      # these will be in centimeters, but could be whatever
        self.radius_fn = os.path.join(self.dirname, "ROI_radii.npy")
        if os.path.exists(self.radius_fn) is False:
            self.radii = np.zeros(self.markers.shape[1])
        else:
            self.radii = np.load(self.radius_fn)
            # remove markers for files that have been deleted
            assert len(self.radii) == len(self.filenames), (
                f"{self.radius_fn} should have the same number of entries as"
                f" image files in {self.dirname}. Instead there are {len(self.filenames)} images"
                f" and {len(self.radii)} entries in {self.radius_fn}.")

        # make input box for the distance used for calibration
        self.input_frame = plt.axes([0.04, .40, 0.10, 0.10], frameon=False)
        self.input_frame.set_title("Radius", ha='right')
        self.input_box = TextBox(self.input_frame,
                                 'cm', initial=str(self.radius),
                                 label_pad=-.65)
        self.input_box.on_submit(self.set_radius)

        self.circles = []
        if isinstance(self.pixel_length, (int, float)):
            pixel_length = self.pixel_length
        else:
            pixel_length = self.pixel_length[self.curr_frame_index]
        radius = radius / pixel_length  # convert to number of pixels
        for marker, color in zip(self.markers[:, 0], self.colors):
            x, y = marker
            circle = plt.Circle((x, y), radius, color=color, fill=False)
            self.figure.axes[0].add_artist(circle)
            self.circles.append(circle)

        # add objects to objects_to_save
        self.objects_to_save[self.radius_fn] = self.radii

    def set_radius(self, text):
        num = float(text)
        self.radius = num

    def show_image(self, *args):
        print('show_image')
        # first plot the image
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[:,
                                                                  self.curr_frame_index - 1]
        # get pixel length for this frame
        if isinstance(self.pixel_length, (int, float)):
            pixel_length = self.pixel_length
        else:
            pixel_length = self.pixel_length[self.curr_frame_index]
        print(f"pixel length: {pixel_length}")
        # get circle radius in pixels
        self.radii[self.curr_frame_index] = self.radius
        radius = self.radius / pixel_length  # convert to number of pixels
        # then update the markers and circles
        # for i in range(self.num_markers):
        for marker, circle, line in zip(self.markers[:, self.curr_frame_index],
                                        self.circles,
                                        self.figure.axes[0].lines):
            x, y = marker
            line.set_xdata(x)
            line.set_ydata(y)
            circle.set_center((x, y))
            circle.set_radius(radius)
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))

        self.data_changed = True
        plt.draw()


class rectangle_bounding_box_GUI(tracker_window):
    """Special window for finding a minimum area bounding box around
    the user-selected points to define a useful frame.
    """

    def __init__(self, dirname="./", num_markers=10):
        super().__init__(dirname, num_markers=num_markers,
                         fn='frame_markers.npy')
        self.frames_fn = os.path.join(self.dirname, 'frame_corners.npy')
        # check if the data has been saved previously
        if os.path.exists(self.frames_fn) is False:
            # 4 X num_markers X 2 dimensions
            self.frames = np.zeros((4, self.markers.shape[1], 2))
        # if so, load the data to self.rectangles
        else:
            self.frames = np.load(self.frames_fn)
            # remove markers for files that have been deleted
            assert self.frames.shape[1] == len(self.filenames), (
                f"{self.frames_fn} should have the same number of entries as"
                f" image files in {self.dirname}. Instead there are {len(self.filenames)} images"
                f" and {len(self.frames)} entries in {self.frames_fn}.")
        # add rectangles to objects_to_save
        self.objects_to_save[self.frames_fn] = self.frames
        # plot bounding rectangle
        positive_responses = self.markers[:, self.curr_frame_index] > 0
        if positive_responses.mean() > 0:
            positive_responses = positive_responses.mean(1)
            markers = self.markers[positive_responses ==
                                   1, self.curr_frame_index]
            self.bounding_rectangle = MinimumBoundingBox(markers)
            rectangle = np.array(tuple(self.bounding_rectangle.corner_points))
            rectangle_centered = rectangle - rectangle.mean(0)
            x, y = rectangle_centered.T
            angles = np.arctan2(y, x)
            order = np.argsort(angles)
            self.rectangle = rectangle[order]
            x, y = self.rectangle.T
        else:
            self.bounding_rectangle = None
            negs = np.repeat(-1, 4)
            x, y = negs, negs
            self.rectangle = np.array([x, y]).T
        self.frames[:, self.curr_frame_index] = self.rectangle
        x, y = np.append(x, x[0]), np.append(y, y[0])
        self.rectangle_line = self.axis.plot(x, y, 'k.-')[0]

    def show_image(self, *args):
        print('show_image')
        # first plot the image
        self.im = self.image
        self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[:,
                                                                  self.curr_frame_index - 1]
        # update the markers
        for marker, line in zip(self.markers[:, self.curr_frame_index],
                                self.figure.axes[0].lines):
            x, y = marker
            line.set_xdata(x)
            line.set_ydata(y)
        # and the title
        self.title.set_text('%d - %s' % (self.curr_frame_index + 1,
                                         self.filenames[self.curr_frame_index].rsplit('/')[-1]))
        # update the bounding box
        markers = self.markers[:, self.curr_frame_index]
        positive_responses = (markers > 0).mean(1)
        print(positive_responses.sum())
        if positive_responses.sum() > 2:
            self.bounding_rectangle = MinimumBoundingBox(
                markers[positive_responses == 1])
            rectangle = np.array(tuple(self.bounding_rectangle.corner_points))
            rectangle_centered = rectangle - rectangle.mean(0)
            x, y = rectangle_centered.T
            angles = np.arctan2(y, x)
            order = np.argsort(angles)
            self.rectangle = rectangle[order]
            self.frames[:, self.curr_frame_index] = self.rectangle
            x, y = self.rectangle.T
            x, y = np.append(x, x[0]), np.append(y, y[0])
            self.rectangle_line.set_xdata(x)
            self.rectangle_line.set_ydata(y)
        self.data_changed = True
        plt.draw()


class FileSelector(QWidget):
    """Offers a file selection dialog box with filters based on common image filetypes.
    """

    def __init__(self, filetypes=ftypes):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        super().__init__()
        self.title = 'Select the videos you want to process.'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480
        self.filetypes = filetypes
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.openFileNamesDialog()
        self.show()

    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        self.files, self.ftype = QFileDialog.getOpenFileNames(
            self,
            "QFileDialog.getOpenFileNames()",
            "",
            self.filetypes,
            options=options)


class Kalman_Filter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed jerk, (2) gaussian
    jerk noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    jerk : float
        Jerk is modeled as normally distributed. This is the mean.
    jerk_std : float
        Jerk distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''

    def __init__(self, num_objects, sampling_interval=30**-1,
                 jerk=0, jerk_std=50,
                 measurement_noise_x=.1, measurement_noise_y=.1,
                 width=None, height=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0],
             [0,             self.dt**6/36, 0,
                 self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt **
                 4/4,  0,             self.dt**3/2, 0],
             [0,             self.dt**5/12, 0,
                 self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt **
                 3/2,  0,             self.dt**2,   0],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = self.Ex
        # define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        # we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0],
             [0, 0, 0,       1,       0,            self.dt],
             [0, 0, 0,       0,       1,            0],
             [0, 0, 0,       0,       0,            1]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        # initialize result variables
        self.Q_local_measurement = []  # point detections
        # initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        self.Q_loc_estimateX = []
        self.Q_loc_estimateY = []
        self.track_strikes = np.zeros(self.max_tracks)
        self.num_tracks = self.num_objects
        self.num_detections = self.num_objects
        self.frame_num = 0

    def get_prediction(self):
        '''
        Get next predicted coordinates using current state and measurement information.

        Returns
        -------
        estimated points : ndarray
            approximated positions with shape.
        '''
        # kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + \
            (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        self.K = self.P @ self.C.T @ np.linalg.inv(
            self.C @ self.P @ self.C.T + self.Ez)
        # now assign the detections to estimated track positions
        # make the distance (cost) matrix between all pairs; rows = tracks and
        # cols = detections
        self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
        if self.height is not None:
            np.clip(self.estimate_points[0], 0,
                    self.height-1, out=self.estimate_points[0])
        if self.width is not None:
            np.clip(self.estimate_points[1], 0,
                    self.width-1, out=self.estimate_points[1])
        return self.estimate_points.T  # shape should be (num_objects, 2)

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                            "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        self.Q_loc_estimateX.append(self.Q_estimate[0])
        self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        # detections matrix
        breakpoint()
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                            "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        no_nans_meas = no_nans_meas.max(1)
        assigned_measurements = np.empty((self.num_tracks, 2))
        assigned_measurements.fill(np.nan)
        if self.num_tracks > 1:
            self.est_dist = spatial.distance_matrix(
                self.estimate_points.T,
                self.Q_loc_meas[:, :self.num_tracks][no_nans_meas])
            # use hungarian algorithm to find best pairings between estimations and measurements
            asgn = optimize.linear_sum_assignment(self.est_dist)
            for num, val in zip(asgn[0], asgn[1]):
                assigned_measurements[num] = self.Q_loc_meas[no_nans_meas][val]
        else:
            # if one track, only one possible assignment
            assigned_measurements[:] = self.Q_loc_meas
        # remove problematic cases
        # close_enough = self.est_dist[asgn] < 25
        no_nans = np.logical_not(np.isnan(assigned_measurements)).max(1)
        # good_cases = np.logical_and(close_enough, no_nans)

        # if self.width is not None:
        #     in_bounds_x = np.logical_and(
        #         assigned_measurements.T[1] > 0,
        #         assigned_measurements.T[1] < self.width)
        # if self.height is not None:
        #     in_bounds_y = np.logical_and(
        #         assigned_measurements.T[0] > 0,
        #         assigned_measurements.T[0] < self.height)
        good_cases = no_nans
        # good_cases = no_nans * in_bounds_x * in_bounds_y
        # apply assignemts to the update
        for num, (good, val) in enumerate(zip(good_cases, assigned_measurements)):
            if good:
                self.Q_estimate[:, num] = self.Q_estimate[:, num] + self.K @ (
                    val.T - self.C @ self.Q_estimate[:, num])
                self.track_strikes[num] = 0
            else:
                self.track_strikes[num] += 1
                self.Q_estimate[2:, num] = 0
        # update covariance estimation
        self.P = (
            np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        # store data
        self.Q_loc_estimateX.append(self.Q_estimate[0])
        self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1
