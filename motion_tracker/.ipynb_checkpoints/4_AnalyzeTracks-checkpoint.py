from motion_tracker import *
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


class Kalman_Filter():
    '''
    2D Kalman filter, assuming constant acceleration.

    Use a 2D Kalman filter to return the estimated position of points given 
    linear prediction of position assuming (1) fixed acceleration, (2) gaussian
    acceleration noise, and (3) gaussian measurement noise.

    Parameters
    ----------
    num_objects : int
        Number of objects to model as well to expect from detector.
    sampling_interval : float
        Sampling interval in seconds. Should equal (frame rate) ** -1.
    acceleration : float
        Acceleration is modeled as normally distributed. This is the mean.
    acceleration_std : float
        Acceleration distribution standard deviation.
    measurement_noise_x : float
        Variance of the x component of measurement noise.
    measurement_noise_y : float
        Variance of the y component of measurement noise.

    '''
    def __init__(self, num_objects, sampling_interval=30**-1,
                 acceleration=0, acceleration_std=50,
                 measurement_noise_x=.1, measurement_noise_y=.1):
        self.num_objects = num_objects
        self.sampling_interval = sampling_interval
        self.dt = self.sampling_interval
        self.acceleration = acceleration
        self.acceleration_std = acceleration_std
        self.measurement_noise_x = measurement_noise_x
        self.measurement_noise_y = measurement_noise_y
        self.tkn_x, self.tkn_y = self.measurement_noise_x, self.measurement_noise_y
        # process error covariance matrix
        self.Ez = np.array(
            [[self.tkn_x, 0         ],
             [0,          self.tkn_y]])
        # measurement error covariance matrix
        self.Ex = np.array(
            [[self.dt**4/4, 0,            self.dt**3/2, 0           ],
             [0,            self.dt**4/4, 0,            self.dt**3/2],
             [self.dt**3/2, 0,            self.dt**2,   0           ],
             [0,            self.dt**3/2, 0,            self.dt**2  ]])
        self.Ex *= self.acceleration_std**2
        # set initial position variance
        self.P = self.Ex
        ## define update equations in 2D as matrices - a physics based model for predicting
        # object motion
        ## we expect objects to be at:
        # [state update matrix (position + velocity)] + [input control (acceleration)]
        self.state_update_matrix = np.array(
            [[1, 0, self.dt, 0      ],
             [0, 1, 0,       self.dt],
             [0, 0, 1,       0      ],
             [0, 0, 0,       1      ]])
        self.control_matrix = np.array(
            [self.dt**2/2, self.dt**2/2, self.dt, self.dt])
        # measurement function to predict next measurement
        self.measurement_function = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]])
        self.A = self.state_update_matrix
        self.B = self.control_matrix
        self.C = self.measurement_function
        ## initialize result variables
        self.Q_local_measurement = []  # point detections
        ## initialize estimateion variables for two dimensions
        self.max_tracks = self.num_objects
        self.Q_estimate = np.empty((4, self.max_tracks))
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
        ## kalman filter
        # predict next state with last state and predicted motion
        self.Q_estimate = self.A @ self.Q_estimate + (self.B * self.acceleration)[:, None]
        # predict next covariance
        self.P = self.A @ self.P @ self.A.T + self.Ex
        # Kalman Gain
        self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.Ez)
        ## now assign the detections to estimated track positions
        # make the distance (cost) matrix between all pairs; rows = tracks and
        # cols = detections
        self.estimate_points = self.Q_estimate[:2, :self.num_tracks]
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
        ## detections matrix
        assert points.shape == (self.num_objects, 2), print("input array should have "
                                                           "shape (num_objects X 2)")
        self.Q_loc_meas = points
        # find nans, exclude from the distance matrix
        no_nans_meas = np.isnan(self.Q_loc_meas[:, :self.num_tracks]) == False
        no_nans_meas = no_nans_meas.max(1)
        self.est_dist = spatial.distance_matrix(
            self.estimate_points.T,
            self.Q_loc_meas[:, :self.num_tracks][no_nans_meas])
        # use hungarian algorithm to find best pairings between estimations and measurements
        asgn = optimize.linear_sum_assignment(self.est_dist)
        # remove problematic cases
        close_enough = self.est_dist[asgn] < 25
        no_nans = np.logical_not(np.isnan(self.est_dist[asgn]))
        good_cases = np.logical_and(close_enough, no_nans)
        bad_cases = np.logical_not(good_cases)
        # apply assignemts to the update
        for num, good, ind in zip(asgn[0], good_cases, asgn[1]):
            if good:
                self.Q_estimate[:, num] = self.Q_estimate[:, num] + self.K @ (
                    self.Q_loc_meas[:, :self.num_tracks][no_nans_meas][ind, :].T 
                    - self.C @ self.Q_estimate[:, num])
                self.track_strikes[num] = 0
            else:
                # add new object tracking
                # new_estimate = np.zeros(4)
                # new_estimate[:2] = Q_loc_meas[ind]
                # self.Q_estimate[:, self.num_tracks] = new_estimate
                self.track_strikes[num] += 1
                # self.num_tracks += 1
                # slow down estimate so that tracking doesn't drift away
                self.Q_estimate[2:, num] = 0
        # too_many_strikes = track_strikes >= 6
        # # give a strike to any tracking that didn't get matched up to a detection
        # if sum(too_many_strikes) > 0:
        #     self.Q_estimate[:, too_many_strikes] = np.nan
        # update covariance estimation
        self.P = (np.eye((self.K @ self.C).shape[0]) - self.K @ self.C) @ self.P
        ## store data
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
    out = np.fft.ifft2(fft2).real
    return out


def track_video(vid, num_objects=3, movement_threshold=50, make_video=True,
                pointer_length=3, object_side_length=20):
    # subtract averaged background
    back = np.median(vid[::100], axis=0).astype(vid.dtype)
    np.subtract(vid, back[None, ...], out=vid)
    np.abs(vid, out=vid)
    num_frames, height, width = vid.shape
    shape = list(vid.shape[1:]) + [3]
    # use KMeans to find individual birds
    clusterer = cluster.KMeans(num_objects)
    # np.clip(vid, movement_threshold, 300, out=vid)
    positive = np.greater(vid, movement_threshold)
    frame_inds, frame_xs, frame_ys = np.where(positive)
    # make kalman filter to smooth and predict location of K-mean centers
    kalman_filter = Kalman_Filter(num_objects=num_objects)
    # get first measurement
    first_frame = positive[0]
    xs, ys = np.where(first_frame)
    arr = np.array([xs, ys]).T
    clusterer.fit(arr)
    measured_centers = clusterer.cluster_centers_
    kalman_filter.add_starting_points(measured_centers)
    # make new KMeans clusterer using points as seed
    clusterer = cluster.KMeans(num_objects, init=measured_centers, n_init=1)
    # for num, frame in enumerate(positive[1:]):
    for frame_num in range(num_frames):
        # predict object centers using kalman filter
        expected_centers = kalman_filter.get_prediction()
        clusterer.init = expected_centers
        # measure object centers using KMeans
        ind = np.where(frame_inds == frame_num)
        xs, ys = frame_xs[ind], frame_ys[ind]
        if len(xs) > num_objects:
            arr = np.array([xs, ys]).T
            clusterer.fit(arr)
            measured_centers = clusterer.cluster_centers_
        # add measurements to kalman filter for future prediction
        else:
            measured_centers.fill(np.nan)
        kalman_filter.add_measurement(measured_centers)
        print_progress(frame_num, num_frames)
    xs, ys = kalman_filter.Q_loc_estimateX, kalman_filter.Q_loc_estimateY
    coords = np.array([xs, ys]).T
    return coords


def make_video(video, coords, point_length=3, trail_length=30):
    '''
    Superimpose coordinates onto a video.

    Plot coordinates on top of video frames, leaving a fading trail behind it.

    Parameters
    ----------
    video : array_like
        Input video array. Should be a (num_frames, height, width, 3) matrix.
    coords : array_like
        Input coordinates array to be plotted. Should be a (num_frames, 
        num_objects, 2) matrix.
    point_length : int
        Diameter of the coordinate points.
    trail_length : int
        The time delay for the coordinate trail.

    Returns
    -------
    new_video : array_like
        New array of the video with superimposed points. Should have the shape
        (num_frames, height, width, 3).
    '''
    # video = np.copy(video)
    # make sure the video is of the right shape
    if video.ndim == 3 or (video.ndim == 4 and video.shape[-1] == 1):
        video = np.squeeze(video)
        num_frames, height, width = video.shape
        shape = (num_frames, height, width, 3)
        # video = np.repeat(video, 3).reshape(shape)
        video = np.repeat(video[..., np.newaxis], 3, axis=-1)
        # new_vid = np.zeros(shape, dtype='uint8')
    assert video.ndim == 4, print("input video array of wrong shape")
    num_frames, height, width, channels = video.shape
    assert video.shape[-1] == 3, print(
        "input video array should have 1 or 3 channels")
    assert coords.ndim == 3, print(
        "input coordinate array should have shape (num_frames, num_objects,\
        2)")
    num_frames, num_objects, ndim = coords.shape
    # get colors for plotting
    colors_arr = []
    for x in range(num_objects):
        colors_arr += [
            np.round(
                255 * np.array(plt.cm.colors.to_rgb(colors[x % len(colors)])))]
    colors_arr = np.array(colors_arr).astype('uint8')
    # make a window for dilating the image of points into circles
    if point_length % 2 == 0:
        point_length += 1
    window = np.zeros((point_length, point_length), dtype=bool)
    radius = (point_length - 1) / 2
    # make an image of the center points to update frame by frame
    weights = np.zeros((num_objects, height, width, 3), dtype=np.float32)
    for num, (frame, points) in enumerate(zip(video, coords)):
        # points = points.astype('uint16')
        overlays_color = np.zeros((num_objects, height, width, 3), dtype='uint8')
        for pnum, ((y, x), color) in enumerate(zip(points, colors_arr)):
            xmin, xmax = int(x - radius), int(x + radius)
            ymin, ymax = int(y - radius), int(y + radius)
            # weights[pnum, x, y] = 1
            weights[pnum, xmin:xmax, ymin:ymax] = 1
            # non_zero = weights[pnum] > 0
            # overlays_color[pnum, non_zero] = color * weights[pnum, non_zero]
            overlays_color[pnum] = color * weights[pnum]
        # overlays = color * weights
        overlay_weights = weights.max(0)
        overlay_inds = np.argmax(weights, axis=0)
        vid_weights = 1 - overlay_weights
        # frame[:] = frame * vid_weights
        frame[:] = overlay_weights * overlays_color.max(0) + vid_weights * frame
        weights -= trail_length ** -1
        weights[weights < 0] = 0
        print_progress(num, num_frames)
    return video


class VideoTracker():
    def __init__(self, num_objects=1, video_files=None,
                 tracks_folder='tracking_data', movement_threshold=90,
                 dt=30**-1):
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

    def track_files(self, save_video=False, point_length=7):
        for fn in self.video_files:
            # make a new filename for the tracking data
            print(f"Tracking file {fn}:")
            ftype = "." + fn.split(".")[-1]
            new_fn = os.path.basename(fn)
            new_fn = new_fn.replace(ftype, "_track_data.npy")
            self.new_fn = os.path.join(self.tracks_folder, new_fn)
            self.vid = np.squeeze(io.vread(fn, as_grey=True)).astype('int16')
            coords = track_video(
                self.vid, num_objects=self.num_objects,
                movement_threshold=self.movement_threshold)
            np.save(self.new_fn, coords)
            # xs, ys = kalman_filter(coords, self.dt)
            # self.coords = np.array([xs, ys]).transpose(1, 2, 0)
            coords = coords.transpose(1, 0, 2)
            coords = coords[..., [1, 0]]
            self.coords = coords
            if save_video:
                vid_fn = self.new_fn.replace("_data.npy", "_video.mp4")
                new_vid = make_video(
                    self.vid, self.coords[:, :5], point_length=point_length)
                io.vwrite(vid_fn, new_vid)
                print(f"Tracking video saved at {vid_fn}")
            return self.coords

# num_objects = 5
# save_video = True
# fn = "./Trial  1499.mpg"
# video_tracker = VideoTracker(
#     num_objects=num_objects, movement_threshold=50, video_files=[fn])
# video_tracker = VideoTracker(
#     num_objects=num_objects, movement_threshold=50)
# video_tracker.track_files(save_video=save_video)

# # video = video_tracker.vid
# # video = np.repeat(video[..., np.newaxis], 3, axis=-1)
# # coords = video_tracker.coords[:, :5]

# video = io.vread(fn, as_grey=True).astype('uint8')
# video = np.repeat(video, 3, axis=-1)
# coords = np.round(np.load("./tracking_data/Trial  1499_track_data.npy")).astype('uint16')
# coords = coords.transpose(1, 0, 2)
# coords = coords[..., [1, 0]]
# xs, ys = kalman_filter(coords)
# coords = np.array([xs[:, :5], ys[:, :5]]).transpose(1, 2, 0)
# new_vid = make_video(video, coords, point_length=7, trail_length=10)
# io.vwrite("./test.mp4", new_vid)

if __name__ == "__main__":
    num_objects = int(input("How many objects are moving, maximum? "))
    save_video = input(
        "Do you want to save a video of the tracking data? Type 1 for "
        "yes and 0 for no: ")
    while save_video not in ["0", "1"]:
        save_video = input(
            "The response must be a 0 or a 1")
    save_video = bool(int(save_video))
    video_tracker = VideoTracker(
        num_objects=num_objects, movement_threshold=90)
    video_tracker.track_files(save_video=save_video)
