from motion_tracker import *
import motion_tracker as mt
from scipy import signal, optimize, ndimage


def predict(xs, ys, ts=None):
    if ts is None:
        ts = np.arange(len(xs))
    no_nans = np.isnan(xs) == False
    # if len(xs) < 11:
    if True:
        x_new = xs[no_nans][-1]
        y_new = ys[no_nans][-1]
    return x_new, y_new


def get_backgrounds(fns, backgrounds_folder="backgrounds", backgrounds_suffix="_background.npy"):
    for num, fn in enumerate(fns):
        dirname = os.path.dirname(fn)
        basename = os.path.basename(fn)
        vid_ftype = fn.split(".")[-1]
        background_folder = os.path.join(dirname, backgrounds_folder)
        if not os.path.isdir(background_folder):
            os.mkdir(background_folder)
        background_fn = os.path.join(backgrounds_folder,
                                     basename.replace(f".{vid_ftype}", backgrounds_suffix))
        if not os.path.exists(background_fn):
            vid = np.squeeze(io.vread(fn, as_grey=True)).astype('int16')
            back = np.median(vid[::100], axis=0).astype(vid.dtype)
            np.save(background_fn, back)
        print_progress(num, len(fns))

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
    def __init__(self, num_objects, num_frames=None, sampling_interval=30**-1,
                 jerk=0, jerk_std=125,
                 measurement_noise_x=5, measurement_noise_y=5,
                 width=None, height=None):
        self.width = width
        self.height = height
        self.num_objects = num_objects
        self.num_frames = num_frames
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.jerk = jerk
        self.jerk_std = jerk_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix (constant jerk)
        self.Ex = np.array(
            [[self.dt**6/36, 0,             self.dt**5/12, 0,             self.dt**4/6, 0           ],
             [0,             self.dt**6/36, 0,             self.dt**5/12, 0,            self.dt**4/6],
             [self.dt**5/12, 0,             self.dt**4/4,  0,             self.dt**3/2, 0           ],
             [0,             self.dt**5/12, 0,             self.dt**4/4,  0,            self.dt**3/2],
             [self.dt**4/6,  0,             self.dt**3/2,  0,             self.dt**2,   0           ],
             [0,             self.dt**4/6,  0,             self.dt**3/2,  0,            self.dt**2  ]])
        self.Ex *= self.jerk_std**2
        # set initial position variance
        self.P = self.Ex
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0,       self.dt**2/2, 0           ],
             [0, 1, 0,       self.dt, 0,            self.dt**2/2],
             [0, 0, 1,       0,       self.dt,      0           ],
             [0, 0, 0,       1,       0,            self.dt     ],
             [0, 0, 0,       0,       1,            0           ],
             [0, 0, 0,       0,       0,            1           ]])
        self.control_matrix = np.array(
            [self.dt**3/6, self.dt**3/6, self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        dimension = self.state_update_matrix.shape[0]
        self.Q_estimate = np.empty((dimension, self.max_tracks))
        self.Q_estimate.fill(np.nan)
        if self.num_frames is not None:
            self.Q_loc_estimateX = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateX.fill(np.nan)
            self.Q_loc_estimateY = np.empty((self.num_frames, self.max_tracks))
            self.Q_loc_estimateY.fill(np.nan)
        else:
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
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.jerk)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
        ## now assign the detections to estimated track positions
        # make the distance (cost) matrix between all pairs; rows = tracks and
        # cols = detections
        self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
        np.clip(self.estimate_points[0], 0, self.height-1, out=self.estimate_points[0])
        np.clip(self.estimate_points[1], 0, self.width-1, out=self.estimate_points[1])
        return self.estimate_points.T  # shape should be (num_objects, 2)

    def add_starting_points(self, points):
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_estimate.fill(0)
        self.Q_estimate[:2] = points.T
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1

    def add_measurement(self, points):
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        no_nans_meas = no_nans_meas.max(1)
        assigned_measurements = np.empty((self.num_tracks, 2))
        assigned_measurements.fill(np.nan)
        self.est_dist = spatial.distance_matrix(
            self.estimate_points.T,
            self.Q_loc_meas[:, :self.num_tracks][no_nans_meas])
        # use hungarian algorithm to find best pairings between estimations and measurements
        asgn = optimize.linear_sum_assignment(self.est_dist)
        for num, val in zip(asgn[0], asgn[1]):
            assigned_measurements[num] = self.Q_loc_meas[no_nans_meas][val]
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
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
        if self.num_frames is not None:
            self.Q_loc_estimateX[self.frame_num] = self.Q_estimate[0]
            self.Q_loc_estimateY[self.frame_num] = self.Q_estimate[1]
        else:
            self.Q_loc_estimateX.append(self.Q_estimate[0])
            self.Q_loc_estimateY.append(self.Q_estimate[1])
        self.frame_num += 1


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_window(side_length):
    xs, ys = np.arange(side_length), np.arange(side_length)
    xs, ys = np.meshgrid(xs, ys)
    mean, sd = side_length/2., side_length/4
    arr = gaussian(xs, mean, sd) * gaussian(ys, mean, sd)
    return arr


def smooth(arr, sigma=5):
    """A 2d smoothing filter for the heights array"""
    # arr = arr.astype("float32")
    fft2 = np.fft.fft2(arr)
    ndimage.fourier_gaussian(fft2, sigma=sigma, output=fft2)
    positive = np.fft.ifft2(fft2).real
    fft2 = np.fft.fft2(arr)
    ndimage.fourier_gaussian(fft2, sigma=1.5*sigma, output=fft2)
    negative = np.fft.ifft2(fft2).real
    return positive - negative


def find_peaks(frame, expected_points, max_num_peaks=None,
               min_distance=5, movement_threshold=50):
    min_num_peaks = len(expected_points)
    if max_num_peaks is None:
        max_num_peaks = 2 * min_num_peaks
    assert min_num_peaks <= max_num_peaks, print(
        "min_num_peaks should be less than number of expected points")
    # find repeated inds and replace with next nearest but not mapped points
    points = peak_local_max(frame, num_peaks=max_num_peaks,
                            min_distance=min_distance)
    # exclude points with low values
    vals = frame[points[:, 0], points[:, 1]]
    fast_enough = vals > movement_threshold
    if fast_enough.sum() >= min_num_peaks:
        points = points[fast_enough]
    else:
        order = np.argsort(vals)[::-1]
        points = points[order[:min_num_peaks]]
    # use hungarian algorithm to find min distance assignements
    dists = spatial.distance_matrix(points, expected_points)
    assignments = optimize.linear_sum_assignment(dists)
    return points[assignments[1]], dists[assignments]


def track_video(vid, num_objects=3, movement_threshold=50, object_side_length=20,
                background=None):
    if background is None:
        # subtract averaged background
        back = np.median(vid[::100], axis=0).astype(vid.dtype)
    else:
        back = background.astype(vid.dtype)
    np.subtract(vid, back[None, ...], out=vid)
    np.abs(vid, out=vid)
    num_frames, height, width = vid.shape
    shape = list(vid.shape[1:]) + [3]
    # setup KMeans to find individual objects
    clusterer = cluster.KMeans(num_objects)  # replace with peak_local_max detection
    np.clip(vid, .8 * movement_threshold, 300, out=vid)
    # make kalman filter to smooth and predict location of K-mean centers
    kalman_filter = Kalman_Filter(num_objects=num_objects, width=width,
                                  height=height, num_frames=num_frames)
    # get first measurement
    errors = []
    for num, frame in enumerate(vid):
        frame_smoothed = smooth(frame, object_side_length/4)
        # ys, xs = np.where(frame_smoothed > movement_threshold)
        # if len(xs) > num_objects * 3:
        #     arr = np.array([ys, xs]).T
        #     clusterer.fit(arr)
        #     measured_centers = clusterer.cluster_centers_
        #     kalman_filter.add_starting_points(measured_centers)
        #     break
        points = peak_local_max(frame_smoothed, num_peaks=2 * num_objects,
                                min_distance=round(object_side_length/4))
        vals = frame_smoothed[points[:, 0], points[:, 1]]
        order = np.argsort(vals)[::-1]
        points = points[order][:num_objects]
        if not np.any(np.isnan(points)):
            measured_centers = points
            kalman_filter.add_starting_points(measured_centers)
            break
        kalman_filter.frame_num += 1
    # make new KMeans clusterer using points as seed
    if kalman_filter.frame_num < num_frames:
        clusterer = cluster.KMeans(num_objects, init=measured_centers, n_init=1)
        for num, frame in enumerate(vid[kalman_filter.frame_num:]):
            frame_smoothed = smooth(frame, sigma=object_side_length/4)
            # frame[:] = frame_smoothed
            # predict object centers using kalman filter
            expected_centers = kalman_filter.get_prediction()
            clusterer.init = expected_centers
            # measure object centers using KMeans
            ys, xs = np.where(frame_smoothed > movement_threshold)
            if len(xs) > num_objects * 3:
                arr = np.array([ys, xs]).T
                clusterer.fit(arr)
                measured_centers = clusterer.cluster_centers_
                # combine points if they're within side length of each other
                tree = spatial.KDTree(measured_centers)
                for center in measured_centers:
                    nearby_inds = tree.query_ball_point(center, r=object_side_length/2)
                    if np.any(np.isnan(center)) and len(nearby_inds) > 1:
                        inds = nearby_inds[1:]
                        midpoint = measured_centers[nearby_inds].mean(0)
                        center = midpoint
                        measured_centers[inds] = np.nan
                # find measurements that are unreasonable
                y_centers, x_centers = np.round(measured_centers.T).astype(int)
                vals = frame_smoothed[y_centers, x_centers]
                # bad_vals = ((x_centers < 0) + (x_centers > width) +
                #             (y_centers < 0) + (y_centers > height) +
                #             (vals < movement_threshold))
                # bad_vals = vals < movement_threshold
                # measured_centers[bad_vals] = np.nan
            # add measurements to kalman filter for future prediction
            else:
                measured_centers.fill(np.nan)
            # good_measurements, dists = find_peaks(frame_smoothed, expected_centers,
            #                                       movement_threshold=movement_threshold,
            #                                       min_distance=int(object_side_length/8))
            # good_measurements = good_measurements.astype(float)
            # good_measurements[dists > 30] = np.nan
            # good_measurements[dists == 0] = np.nan
            # errors.append(dists)
            # measured_centers = good_measurements
            kalman_filter.add_measurement(measured_centers)
            print_progress(num, num_frames)
    xs, ys = kalman_filter.Q_loc_estimateX, kalman_filter.Q_loc_estimateY
    coords = np.array([xs, ys]).T
    return coords


class VideoTracker():
    def __init__(self, num_objects=1, video_files=None,
                 tracks_folder='tracking_data', movement_threshold=90,
                 dt=30**-1, use_neighboring_backgrounds=False,
                 backgrounds_folder="backgrounds"):
        self.num_objects = num_objects
        self.movement_threshold = movement_threshold
        self.app = QApplication.instance()
        self.dt = dt
        if self.app is None:
            self.app = QApplication([])
        # grab the video files
        if video_files is None:
            print("Select the video files you want to motion track:")
            file_UI = FileSelector()
            file_UI.close()
            self.video_files = file_UI.files
        else:
            self.video_files = video_files
        # figure out the parent directory
        self.folder = os.path.dirname(self.video_files[0])
        os.chdir(self.folder)
        # find/make tracks_folder
        self.tracks_folder = os.path.join(self.folder, tracks_folder)
        if os.path.isdir(self.tracks_folder) is False:
            os.mkdir(self.tracks_folder)
        # store video backgrounds if wanted
        self.use_neighboring_backgrounds = use_neighboring_backgrounds
        if use_neighboring_backgrounds:
            self.backgrounds_folder = os.path.join(self.folder, backgrounds_folder)
            get_backgrounds(self.video_files, backgrounds_folder=backgrounds_folder)
            fns = os.listdir(self.backgrounds_folder)
            fns = [fn for fn in fns if fn.endswith(".npy")]
            self.background_fns = os.listdir(self.backgrounds_folder)
            self.backgrounds = []
            for vid_fn in self.video_files:
                basename = os.path.basename(vid_fn)
                vid_ftype = basename.split(".")[-1]
                background_fn = os.path.join(
                    self.folder,
                    self.backgrounds_folder,
                    basename.replace(f".{vid_ftype}", "_background.npy"))
                back = np.load(background_fn)
                self.backgrounds.append(back)
            self.backgrounds = np.array(self.backgrounds)
            self.avg_backgrounds = self.backgrounds.mean(0)

    def track_files(self, save_video=False, point_length=7, object_side_length=20):
        for num, fn in enumerate(self.video_files):
            # make a new filename for the tracking data
            print(f"Tracking file {fn}:")
            ftype = "." + fn.split(".")[-1]
            new_fn = os.path.basename(fn)
            new_fn = new_fn.replace(ftype, "_track_data.npy")
            self.new_fn = os.path.join(self.tracks_folder, new_fn)
            vid_fn = self.new_fn.replace("_data.npy", "_video.mp4")
            self.vid = None
            if not os.path.exists(self.new_fn):
                self.vid = np.squeeze(io.vread(fn, as_grey=True)).astype('int16')
                if self.use_neighboring_backgrounds:
                    low = max(0, num - 1)
                    high = min(num + 2, len(self.backgrounds))
                    back = self.backgrounds[[num - 1, num + 1]].mean(0)
                    # back = self.avg_backgrounds
                else:
                    back = None
                coords = track_video(
                    self.vid, num_objects=self.num_objects,
                    movement_threshold=self.movement_threshold,
                    object_side_length=object_side_length,
                    background=back)
                np.save(self.new_fn, coords[..., [1, 0]])
                # xs, ys = kalman_filter(coords, self.dt)
                # self.coords = np.array([xs, ys]).transpose(1, 2, 0)
                coords = coords.transpose(1, 0, 2)
                coords = coords[..., [1, 0]]
                self.coords = coords
            else:
                self.coords = np.load(self.new_fn)
            if save_video and not os.path.exists(vid_fn) and np.isnan(self.coords).mean() < 1:
                if self.vid is None:
                    self.vid = np.squeeze(io.vread(fn, as_grey=True)).astype('int16')
                new_vid = mt.make_video(
                    self.vid, self.coords[:, :5], point_length=point_length)
                io.vwrite(vid_fn, new_vid)
                print(f"Tracking video saved at {vid_fn}")


if __name__ == "__main__":
    num_objects = int(input("How many objects are moving, maximum? "))
    object_side_length = int(input("How many pixels long is the object, approximately? "))
    movement_threshold = int(input("What is the minimum motion value needed to be detected? "))
    save_video = input(
        "Do you want to save a video of the tracking data? Type 1 for "
        "yes and 0 for no: ")
    while save_video not in ["0", "1"]:
        save_video = input(
            "The response must be a 0 or a 1")
    save_video = bool(int(save_video))
    # option for using neighboring videos' backgrounds
    neighbor_background = input(
        "Do you want to use neighboring videos to extract background information? "
        "yes and 0 for no: ")
    while neighbor_background not in ["0", "1"]:
        neighbor_background = input(
            "The response must be a 0 or a 1")
    neighbor_background = bool(int(neighbor_background))
    # save_video = True
    # num_objects = 5
    # object_side_length = 15
    # movement_threshold = 10
    video_tracker = VideoTracker(
        num_objects=num_objects, movement_threshold=movement_threshold,
        use_neighboring_backgrounds=neighbor_background)
    video_tracker.track_files(save_video=save_video, object_side_length=object_side_length)
