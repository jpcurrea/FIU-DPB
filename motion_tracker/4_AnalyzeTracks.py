from motion_tracker import *
from scipy import signal, optimize, ndimage

fns = os.listdir("./")
fns = [fn for fn in fns if fn.endswith("track_data.npy")]
# 1. consolidate multiple tracks into 1 using Kalman Filter
# 2. for those with all nans, use GUI to select single location
for num, fn in enumerate(fns):
    tracks = np.load(fn)
    diff = tracks[1] - tracks[0]
    dists = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
    tracks = tracks.transpose((1, 0, 2))
    thresh = np.percentile(dists, 99)
    kalman_filter = Kalman_Filter(1, jerk_std=10)
    if dists[0] < thresh:
        center = tracks[0].mean(0)
        kalman_filter.add_starting_points(center[np.newaxis])
    for frame, dist in zip(tracks[1:], dists[1:]):
        prediction = kalman_filter.get_prediction()
        error = np.linalg.norm(prediction - frame, axis=-1)
        if sum(error < thresh) > 1:
            measurement = frame[error < thresh].mean(0)
        else:
            measurement = frame[np.argmin(error)]
        kalman_filter.add_measurement(measurement[np.newaxis])
    xs = np.squeeze(np.array(kalman_filter.Q_loc_estimateX))
    ys = np.squeeze(np.array(kalman_filter.Q_loc_estimateY))
    arr = np.array([xs, ys]).T
    np.save(fn.replace("track_data", "position_data"), arr)
    print_progress(num, len(fns))



if __name__ == "__main__":
    # num_objects = 3
    # save_video = True
    # # fn = "./Trial  1499.mpg"
    # fn = "/Volumes/Lab/Maternal Loco/vestibular_backup/Videos/small_inferior_vids/cropped/1111_cropped.mp4"
    # video_tracker = VideoTracker(
    #     num_objects=num_objects, movement_threshold=20, video_files=[fn])
    # # video_tracker = VideoTracker(
    # #     num_objects=num_objects, movement_threshold=50)
    # video_tracker.track_files(save_video=save_video, object_side_length=5)

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
    video_tracker = VideoTracker(
        num_objects=num_objects, movement_threshold=movement_threshold)
    video_tracker.track_files(save_video=save_video, object_side_length=object_side_length)
