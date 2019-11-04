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