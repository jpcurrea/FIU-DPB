from motion_tracker import FileSelector
from scipy import spatial
from collections import namedtuple, Counter
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import Arrow, Circle, Polygon, Rectangle
from matplotlib.animation import FuncAnimation

# from MinimumBoundingBox import MinimumBoundingBox
import matplotlib
from matplotlib import pyplot as plt
import math
import time
import numpy as np
import os
import PIL
from scipy import interpolate, ndimage, spatial, optimize
from skvideo import io
from skimage.feature import peak_local_max
from sklearn import cluster
import subprocess
import threading
import sys

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
    'tab:cyan',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive'
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


class VideoTrackerWindow():

    def __init__(self, filename, num_markers=6, tracking_folder="tracking_data",
                 data_fn_suffix='_track_data.npy', fps=30):
        # m.pyplot.ion()
        self.filename = filename
        self.dirname = os.path.dirname(filename)
        self.basename = os.path.basename(filename)
        self.tracking_folder = os.path.join(
            self.dirname,
            tracking_folder)
        self.ftype = self.filename.split(".")[-1]
        self.tracking_fn = os.path.join(
            self.tracking_folder,
            self.basename.replace(f".{self.ftype}", data_fn_suffix))
        self.load_file()
        self.num_frames = len(self.video)
        self.range_frames = np.array(range(self.num_frames))
        self.curr_frame_index = 0
        self.inherent_fps = fps

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
        if os.path.isfile(self.tracking_fn):
            self.markers = np.load(self.tracking_fn)
            self.markers = self.markers
        else:
            self.markers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1
            self.imarkers = np.zeros(
                (self.num_markers, self.num_frames, 2), dtype='float')-1
        self.data_changed = False
        self.interpolated_data = np.zeros(self.markers.shape[:-1], dtype=bool)

        # the figure
        self.load_image()
        # figsize = self.image.shape[1]/90, self.image.shape[0]/90
        h, w = self.image.shape[:2]
        if w > h:
            fig_width = 10
            fig_height = h/w * fig_width
        else:
            fig_height = 8
            fig_width = w/h * fig_height
        self.figure = plt.figure(1, figsize=(fig_width, fig_height), dpi=90)
        fig_left, fig_bottom, fig_width, fig_height = .15, .15, .75, .85
        axim = plt.axes([fig_left, fig_bottom, fig_width, fig_height])
        self.implot = plt.imshow(self.image, cmap='gray', animated=True)
        axim.set_xticks([])
        axim.set_yticks([])
        self.xlim = self.figure.axes[0].get_xlim()
        self.ylim = self.figure.axes[0].get_ylim()
        self.axis = self.figure.get_axes()[0]
        self.marks = []
        for i in range(self.num_markers):
            line = plt.plot([-1], [-1], '+', color=self.colors[i])
            self.marks.append(line[0])
        self.marker_lines_ax = self.figure.axes[0]
        self.marker_lines_ax.set_xlim(*self.xlim)
        self.marker_lines_ax.set_ylim(*self.ylim)
        # self.figure.axes[0].set_xlim(*self.xlim)
        # self.figure.axes[0].set_ylim(*self.ylim)
        self.image_data = self.axis.images[0]
        # title
        self.title = self.figure.suptitle(
            f"frame #{self.curr_frame_index + 1}")
        self.slider_ax = plt.axes([fig_left, 0.04, fig_width, 0.02])
        self.curr_frame = Slider(
            self.slider_ax, 'frame', 1, self.num_frames, valinit=1, valfmt='%d', color='k')
        self.curr_frame.on_changed(self.change_frame)
        # setup selection bar
        self.selected_frames_ax = plt.axes([fig_left, 0.06, fig_width, 0.08], frameon=False)
        self.selected_frames_ax.set_xticks([])
        self.selected_frames_ax.set_yticks([])
        self.selected_frames_ax.spines['bottom'].set_visible(False)
        self.selected_in = 0
        # self.selected_out = self.num_frames - 1
        self.selected_out = 0
        self.update_selection_bar()
        # radio buttons for marker selection
        self.radioframe = plt.axes([0.02, .75, 0.08, 0.20], frameon=False)
        self.radioframe.set_title("marker", ha='center')
        lbls = [str(x + 1) for x in range(self.num_markers)]
        lbls = tuple(lbls)
        self.radiobuttons = RadioButtons(
            self.radioframe, lbls, activecolor='k')
        # color the buttons
        for i in np.arange(len(lbls)):
            self.radiobuttons.labels[i].set_color(self.colors[i])
        self.radiobuttons.on_clicked(self.marker_button)

        # one radio button and input box per marker
        ypad = .015
        box_height = (.20 - 2*ypad) / self.num_markers
        box_width = .08 / 2
        box_x = .02
        start_y = .75+ ypad
        ys = (box_height * np.arange(self.num_markers) + start_y)[::-1]  # from top to bottom
        self.swap_inputs = []
        # self.radiobuttons = []
        for num, (lbl, color, y) in enumerate(zip(lbls, self.colors, ys)):
            input_ax = plt.axes([box_x + box_width, y,
                                 box_width-.02, box_height],
                                frameon=True)
            swap_input = TextBox(input_ax, "")
            self.swap_inputs.append(swap_input)
        # swap buttons
        self.swap_button_ax = plt.axes(
            [0.02, start_y - 3 * box_height, 2.7 * box_width, 1.5 * box_height],
            frameon=True)
        self.swap_button = Button(self.swap_button_ax, 'Swap (in:out)')
        self.swap_button.on_clicked(self.swap_selection)
        # swap to end button
        self.swap_to_end_button_ax = plt.axes(
            [0.02, start_y - 4.5 * box_height, 2.7 * box_width, 1.5 * box_height],
            frameon=True)
        self.swap_to_end_button = Button(self.swap_to_end_button_ax, 'Swap (in:end)')
        self.swap_to_end_button.on_clicked(self.swap_to_end)
        # merge (in:out) button
        self.merge_button_ax = plt.axes(
            [0.02, start_y - 6 * box_height, 2.7 * box_width, 1.5 * box_height],
            frameon=True)
        self.merge_button = Button(self.merge_button_ax, 'Merge (in:out)')
        self.merge_button.on_clicked(self.merge_selection)
        # connect some keys
        self.cidk = self.figure.canvas.mpl_connect(
            'key_release_event', self.on_key_release)
        # change the toolbar functions
        NavigationToolbar2.home = self.show_image
        NavigationToolbar2.save = self.save_data
        # make a list of objects and filenames to save
        self.objects_to_save = {}
        self.objects_to_save[self.tracking_fn] = self.markers
        # plot markers
        self.playing = False
        self.show_image()
        # input box for setting the framerate
        self.framerate = 30
        self.input_frame = plt.axes([0.01, .45, 0.07, 0.05], frameon=True)
        self.input_frame.set_title("F.Rate:", ha='center', va='center')
        self.input_box = TextBox(self.input_frame,
                                 'fps', initial=str(self.framerate),
                                 label_pad=-.75)
        self.input_box.on_submit(self.set_framerate)
        # make the following buttons:
        # save
        self.save_button_ax = plt.axes([0.01, .30, .06, .05])
        self.save_button = Button(self.save_button_ax, 'Save')
        self.save_button.on_clicked(self.save_data)
        # play/pause
        self.play_pause_button_ax = plt.axes([0.07, .30, .06, .05])
        self.play_pause_button = Button(self.play_pause_button_ax, 'Play')
        self.play_pause_button.on_clicked(self.play)
        # interpolate
        self.interpolate_button_ax = plt.axes([0.01, .25, .06, .05])
        self.interpolate_button = Button(self.interpolate_button_ax, 'Fix')
        self.interpolate_button.on_clicked(self.interpolate_nans)
        # delete
        self.delete_button_ax = plt.axes([0.07, .25, .06, .05])
        self.delete_button = Button(self.delete_button_ax, 'Remove')
        self.delete_button.on_clicked(self.remove_selection)

        # set in point at start frame
        # set in point at current frame
        # set out point at current frame
        # set out point at end frame


    def load_file(self):
        self.video = np.squeeze(io.vread(self.filename, as_grey=True)).astype('int16')
        # self.video = self.video.transpose((0, 2, 1))

    def load_image(self):
        print(self.curr_frame_index)
        # print(len(self.filenames))
        self.image = self.video[self.curr_frame_index]
        # self.image = PIL.Image.open(self.filenames[self.curr_frame_index])
        # self.image = np.asarray(self.image)

    def show_image(self, *args):
        print('show_image')
        # first plot the image
        self.im = self.image
        # self.figure.axes[0].get_images()[0].set_data(self.im)
        self.implot.set_data(self.im)
        # if no markers are set, use previous markers as default
        if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
            self.markers[:, self.curr_frame_index] = self.markers[
                :, self.curr_frame_index - 1]
        # then plot the markers
        for num, mark in enumerate(self.marks):
            mark.set_xdata(self.markers[num, self.curr_frame_index, 0])
            mark.set_ydata(self.markers[num, self.curr_frame_index, 1])
        # for i in range(self.num_markers):
        #     self.axis.lines[i].set_xdata(
        #         self.markers[i, self.curr_frame_index, 0:1])
        #     self.axis.lines[i].set_ydata(
        #         self.markers[i, self.curr_frame_index, 1:2])
            # self.figure.axes[0].lines[i].set_xdata(
            #     self.markers[i, self.curr_frame_index, 0:1])
            # self.figure.axes[0].lines[i].set_ydata(
            #     self.markers[i, self.curr_frame_index, 1:2])
        # and the title
        self.title.set_text(f"{self.filename} - frame #{self.curr_frame_index + 1}")
        plt.draw()

    def set_framerate(self, text):
        num = float(text)
        self.framerate = num

    def playing_thread(self):
        # setup seems to have an inherent framerate of 10 fps
        # let's convert desired framerate to something near 10 fps but with desired jumps
        step = self.framerate / self.inherent_fps
        inherent_fps = self.framerate / step
        self.playing = True
        # self.animated_show()
        self.animation = FuncAnimation(self.figure, self.animated_show,
                                       interval=1000*self.framerate**-1,
                                       blit=False, repeat=True)
        self.animation.event_source.start()
        # while self.playing:
        #     pass
        # self.animation.event_source.stop()
        # while self.playing:
        #     self.curr_frame.set_val(
        #         np.mod(self.curr_frame_index + 1 + step, self.num_frames))
        #     self.show_image()
        #     time.sleep(inherent_fps ** -1)

    def animated_show(self, event=None):
        self.curr_frame_index += 2
        self.curr_frame_index = np.mod(self.curr_frame_index, self.num_frames)
        self.curr_frame.set_val(self.curr_frame_index)
        # self.image = self.video[self.curr_frame_index - 1]
        # self.im = self.image
        self.implot.set_data(self.video[self.curr_frame_index - 1])
        # self.figure.axes[0].get_images()[0].set_data(self.im)
        # if no markers are set, use previous markers as default
        # if (self.markers[:, self.curr_frame_index] > 0).sum() == 0:
        #     self.markers[:, self.curr_frame_index] = self.markers[:,
        #                                                           self.curr_frame_index - 1]
        # then plot the markers
        for num, mark in enumerate(self.marks):
            mark.set_xdata(self.markers[num, self.curr_frame_index, 0])
            mark.set_ydata(self.markers[num, self.curr_frame_index, 1])
        # for i in range(self.num_markers):
        #     self.axis.lines[i].set_xdata(
        #         self.markers[i, self.curr_frame_index, 0:1])
        #     self.axis.lines[i].set_ydata(
        #         self.markers[i, self.curr_frame_index, 1:2])
        # and the title
        self.title.set_text(f"{self.filename} - frame #{self.curr_frame_index + 1}")
        # plt.draw()
        # time.sleep(self.framerate**-1)
        

    def play(self, event=None):
        if self.playing:
            self.playing = False
            # self.player.exit()
            self.animation.event_source.stop()
            self.load_image()
        else:
            self.player = threading.Thread(target=self.playing_thread, daemon=True)
            self.player.start()

    def change_frame(self, new_frame):
        if not self.playing:
            print('change_frame {} {}'.format(new_frame, int(new_frame)))
            self.curr_frame_index = int(new_frame)-1
            self.load_image()
            self.show_image()
            if self.data_changed:
                self.data_changed = False

    def set_in_point(self):
        self.selected_in = self.curr_frame_index + 1
        self.update_selection_bar()

    def set_out_point(self):
        self.selected_out = self.curr_frame_index + 1
        self.update_selection_bar()

    def update_selection_bar(self):
        original = np.isnan(self.markers).mean(-1) == 0
        ys = np.arange(self.num_markers)
        # yvals = np.repeat(ys[:, np.newaxis], self.num_frames, axis=-1)
        frames = np.arange(self.num_frames)
        # remove old data
        self.selected_frames_ax.clear()
        # plot span of frames that are in selection
        self.selected_frames_ax.axvspan(self.selected_in, self.selected_out,
                                        color='grey', alpha=.5, zorder=0)
        # plot lines where frames are not nan
        linewidths = np.ones(ys.shape)
        linewidths[ys == self.curr_marker] = 2
        linewidths = 2 * linewidths
        not_nan = np.isnan(self.markers).mean(-1) == 0
        interpolated = np.logical_and(self.interpolated_data, not_nan)
        not_interpolated = self.interpolated_data == False
        not_nan = not_nan * not_interpolated
        for y, track, color, lw in zip(ys, not_nan, self.colors, linewidths):
            diff = np.diff(track)
            changes = np.where(diff)[0]
            if track[0]:
                starts = changes[1::2]
                stops = changes[::2]
                starts = np.append(0, starts)
            else:
                starts = changes[::2]
                stops = changes[1::2]
            if track[-1]:
                stops = np.append(stops, self.num_frames - 1)
            for start, stop in zip(starts, stops):
                self.selected_frames_ax.plot([start, stop], [y, y], color=color, linewidth = lw)
        # plot dashed lines for interpolated data
        for y, track, color, lw in zip(ys, interpolated, self.colors, linewidths):
            diff = np.diff(track)
            changes = np.where(diff)[0]
            if track.sum() > 0:
                if track[0]:
                    starts = changes[1::2]
                    stops = changes[::2]
                    starts = np.append(0, starts)
                else:
                    starts = changes[::2]
                    stops = changes[1::2]
                if track[-1]:
                    stops = np.append(stops, self.num_frames - 1)
                for start, stop in zip(starts, stops):
                    self.selected_frames_ax.plot([start, stop], [y, y], color=color, linewidth = lw, linestyle=':')
        # make sure out line is >= in line
        if self.selected_in > self.selected_out:
            new_out = np.copy(self.selected_in)
            self.selected_in = np.copy(self.selected_out)
            self.selected_out = new_out
        # plot in line
        self.selected_frames_ax.plot([self.selected_in, self.selected_in],
                                     [-.5, self.num_markers + .5], color = 'k',
                                     linestyle=':')
        self.selected_frames_ax.text(self.selected_in, -.5,
                                     'IN', horizontalalignment='right')
        # plot out line
        self.selected_frames_ax.plot([self.selected_out, self.selected_out],
                                     [-.5, self.num_markers + .5], color = 'k',
                                     linestyle='-')
        self.selected_frames_ax.text(self.selected_out,  -.5,
                                     'OUT', horizontalalignment='left')
        self.selected_frames_ax.set_xlim(0, self.num_frames)
        self.selected_frames_ax.set_ylim(-.5, self.num_markers - .5)
        self.selected_frames_ax.set_yticks([])
        self.selected_frames_ax.set_xticks([])
        self.selected_frames_ax.invert_yaxis()
        plt.draw()

    def interpolate_nans(self, event=None):
        for num, track in enumerate(self.markers):
            nans = np.isnan(track).mean(-1) == 1
            if np.any(nans):
                frame_nums = np.arange(self.num_frames)
                xs, ys = track[..., 0], track[..., 1]
                xmodel = interpolate.interp1d(frame_nums[nans == False], xs[nans == False],
                                              kind='cubic', fill_value="extrapolate",
                                              bounds_error=False)
                new_xs = xmodel(frame_nums[nans])
                ymodel = interpolate.interp1d(frame_nums[nans == False], ys[nans == False],
                                              kind='cubic', fill_value="extrapolate",
                                              bounds_error=False)
                new_ys = ymodel(frame_nums[nans])
                track[frame_nums[nans], 0], track[frame_nums[nans], 1] = new_xs, new_ys
                self.interpolated_data[num, nans] = True
        self.show_image()
        self.update_selection_bar()

    def remove_selection(self, event=None):
        self.markers[self.curr_marker, self.selected_in:self.selected_out] = np.nan
        self.update_selection_bar()
        self.show_image()

    def swap_selection(self, event=None, end=False):
        new_vals = []
        for num, swap_input in enumerate(self.swap_inputs):
            text = swap_input.text
            try:
                new_vals.append(int(text)-1)
            except:
                new_vals.append(num)
        new_vals = np.array(new_vals)
        marker_range = np.arange(self.num_markers)
        if (new_vals != marker_range).sum() > 1:
            missing_markers = sorted(set(marker_range) - set(new_vals))
            missing_markers = [str(marker + 1) for marker in missing_markers]
            missing_markers = ", ".join(missing_markers)
            if len(missing_markers) > 0:
                print("swap locations are missing the following markers: "
                      f"{missing_markers}")
            else:
                if end:
                    self.markers[:, self.selected_in:] = self.markers[new_vals, self.selected_in:]
                else:
                    self.markers[:, self.selected_in:self.selected_out] = self.markers[new_vals, self.selected_in:self.selected_out]
                self.show_image()

    def swap_to_end(self, event=None):
        self.swap_selection(event=event, end=True)

    def merge_selection(self, event=None):
        # new_vals = {}
        old_to_new = {}
        for num, swap_input in enumerate(self.swap_inputs):
            text = swap_input.text
            try:
                input_num = int(text) - 1
                if input_num not in old_to_new.keys():
                    old_to_new[input_num] = [num]
                else:
                    old_to_new[input_num].append(num)
                # new_vals[num] = (int(text)-1)
            except:
                pass
        # new_val_counts = Counter(new_vals.values())
        for destination, sources in old_to_new.items():
            if len(sources) > 1 :
                if destination in sources:
                    self.markers[destination, self.selected_in:self.selected_out] = np.nanmean(
                        self.markers[sources, self.selected_in:self.selected_out],
                        axis=0)
                    sources.remove(destination)
                    self.markers[sources, self.selected_in:self.selected_out] = np.nan
                else:
                    print(f"Error: destination marker ({destination + 1}) should be included in source markers ({list(np.array(sources) + 1)})")
            else:
                print("Error: there should be more than one source for merging")
        self.show_image()
        self.update_selection_bar()

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
        elif event.key == 'i':
            self.set_in_point()
        elif event.key == 'o':
            self.set_out_point()

        # marker change
        elif event.key in ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0'):
            ymax, xmax = self.implot.get_size()
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
                if event.inaxes == self.axis:
                    self.markers[self.curr_marker, self.curr_frame_index, :] = [
                        event.xdata, event.ydata]
                    self.data_changed = True
                self.radiobuttons.circles[self.curr_marker].set_facecolor(
                    (0.0, 0.0, 0.0, 1.0))
                self.show_image()
                self.update_selection_bar()

        elif event.key == " ":
            self.play()
        elif event.key == "p":
            self.playing = False
        elif event.key == "e":
            self.swap_to_end()
        elif event.key == "s":
            self.swap_selection()
        elif event.key == "d":
            self.remove_selection()
        elif event.key == "f":
            self.interpolate_nans()
        elif event.key == "m":
            self.merge_selection()
        

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
            if event.inaxes == self.axis:
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
        self.update_selection_bar()

    def save_data(self, event=None):
        print('save')
        for fn, val in zip(self.objects_to_save.keys(), self.objects_to_save.values()):
            np.save(fn, val)

if __name__ == "__main__":
    file_UI = FileSelector()
    file_UI.close()
    fn = file_UI.files[0]
    tracker = VideoTrackerWindow(fn)
    plt.show()
