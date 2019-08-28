from skvideo import io
import os
import numpy as np
from scipy import ndimage, spatial, interpolate
from skimage.feature import peak_local_max
from sklearn import cluster
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication

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


def spline_predict(xs, ys, ts=None, num_frames=30):
    # if ts is None:
    #     ts = np.arange(0, len(xs))

    # # ignore nans for all three dimensions
    # nans = np.logical_or(np.isnan(xs), np.isnan(ys), np.isnan(ts))
    # no_nans = nans == False
    # xs, ys, ts = xs[no_nans], ys[no_nans], ts[no_nans]

    # num_frames = min(num_frames, len(ts))
    # xs, ys, ts = xs[:num_frames], ys[:num_frames], ts[:num_frames]
    # new_t = np.array(ts.max() + np.diff(ts).min())

    # # spline model
    # x_model = interpolate.splrep(ts, xs, s=1.5)
    # y_model = interpolate.splrep(ts, ys, s=1.5)
    # new_x = interpolate.splev(new_t, x_model, ext=0)
    # new_y = interpolate.splev(new_t, y_model, ext=0)

    # x_model = interpolate.interp1d(ts, xs,
    #                                fill_value='extrapolate',
    #                                bounds_error=False,
    #                                kind='slinear',
    #                                assume_sorted=True)
    # y_model = interpolate.interp1d(ts, ys,
    #                                fill_value='extrapolate',
    #                                bounds_error=False,
    #                                kind='slinear',
    #                                assume_sorted=True)
    # new_x = x_model(new_t)
    # new_y = y_model(new_t)
    no_nans = np.isnan(xs) == False
    new_x = xs[no_nans][-1]
    new_y = ys[no_nans][-1]

    return new_x, new_y


def track_video(fn, num_points=3, movement_threshold=90, make_video=False):
    fn_head = os.path.split(fn)[-1].split(".")[0]

    vid = np.squeeze(io.vread(fn, as_grey=True))
    back = vid[::100].mean(0).astype(int)

    if make_video:
        new_vid = np.zeros(
            (vid.shape[0], vid.shape[1], vid.shape[2], 3), dtype='uint8')

    num_frames, height, width = vid.shape
    coords = np.zeros([num_frames, num_points, 2])
    coords[:] = np.nan

    shape = list(vid.shape[1:]) + [3]
    clusterer = cluster.KMeans(num_points)
    l = 3
    for num, frame in enumerate(vid):
        diff = abs(frame.astype(int) - back).astype('uint8')
        thresh = diff > movement_threshold
        ys, xs = np.where(thresh)
        arr = np.array([xs, ys]).T
        try:
            clusterer.fit(arr)
            labels = clusterer.labels_
            points = clusterer.cluster_centers_
            coords[num] = points
            future_points = []
            for path in coords.transpose(1, 2, 0):
                px, py = path
                future_points += [spline_predict(px, py)]
            clusterer.init = np.array(future_points)
            clusterer.n_init = 1
        except:
            points = np.repeat(np.nan, num_points)

        if make_video:
            new_vid[num] = np.repeat(frame[..., np.newaxis], 3, axis=-1)
            if np.isnan(points).sum() == num_points:
                pass
            else:
                for label, color in zip(sorted(set(labels)), colors):
                    i = labels == label
                    x, y = int(
                        np.round(xs[i].mean())), int(np.round(ys[i].mean()))
                    new_vid[num, y - l:y + l, x - l: x + l] = 255 * np.array(
                        plt.cm.colors.to_rgb(color))

        print_progress(num, num_frames)

    coords = np.array(coords)

    ftype = "." + fn.split(".")[-1]
    npy_fn = fn.replace(ftype, "_track_data.npy")
    np.save(npy_fn, coords)

    if make_video:
        vid_fn = fn.replace(ftype, "_track_video.mp4")
        io.vwrite(vid_fn, new_vid)


filetypes = [
    ('mpeg videos', '*.mpg *.mpeg *.mp4'),
    ('avi videos', '*.avi'),
    ('quicktime videos', '*.mov *.qt')]

ftypes = [f"{fname} ({ftype})" for (fname, ftype) in filetypes]
ftypes = ";;".join(ftypes)


class fileSelector(QWidget):
    """Offers a file selection dialog box with filters based on common image filetypes.
    """

    def __init__(self, filetypes=ftypes):
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


if __name__ == "__main__":
    num_points = int(input("How many objects are moving, maximum? "))
    make_video = input(
        "Do you want to save a video of the tracking data? Type 1 for yes and 0 for no: ")
    while make_video not in ["0", "1"]:
        make_video = input(
            "The response must be a 0 or a 1")
    make_video = bool(int(make_video))

    app = QApplication([])
    file_UI = fileSelector()
    file_UI.close()
    fns = file_UI.files
    for fn in fns:
        ftype = "." + fn.split(".")[-1]
        npy_fn = fn.replace(ftype, "_track_data.npy")
        if os.path.exists(npy_fn) is False:
            print(f"Tracking file {fn}:")
            track_video(fn, num_points, make_video=make_video)
            print('/n')
    app.exec_()
