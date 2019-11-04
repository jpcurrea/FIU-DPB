from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import interpolate, ndimage, spatial
from skvideo import io
from skimage.feature import peak_local_max
from sklearn import cluster
import subprocess
import sys

from PyQt5.QtWidgets import QWidget, QFileDialog, QApplication
from points_GUI import *

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


class FileSelector(QWidget):
    """Offers a file selection dialog box with filters based on common image filetypes.
    """

    def __init__(self, filetypes=ftypes):
        super().__init__()
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
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
