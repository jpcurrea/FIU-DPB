from motion_tracker import *
from scipy import signal, optimize, ndimage
from shutil import copyfile


def fix_nans(fns, num_objects=1):
    nans_fns = []
    tracking_fns = []
    thumbnail_fns = []
    folder = os.path.dirname(fns[0])
    for fn in fns:
        base = os.path.basename(fn)
        tracking_fn = os.path.join(folder, "tracking_data", base.replace(".mpg", "_track_data.npy"))
        tracking_fns += [tracking_fn]
        thumbnail_fn = os.path.join(folder, "thumbnails", base.replace(".mpg", ".jpg"))
        thumbnail_fns += [thumbnail_fn]
    for num, (fn, tracking_fn, thumbnail_fn) in enumerate(zip(fns, tracking_fns, thumbnail_fns)):
        tracks = np.load(tracking_fn)
        if np.isnan(tracks).mean() > .9:
            nans_fns += [fn]
    # for those with all nans, use GUI to select single location
    if len(nans_fns) > 0:
        # make a folder for nans
        folder = os.path.dirname(fn)
        new_folder = os.path.join(folder,"nan_videos")
        if not os.path.isdir(new_folder):
            os.mkdir(new_folder)
        # get thumbnail file names to move
        thumbnail_fns = []
        tracking_fns = []
        for fn in nans_fns:
            base = os.path.basename(fn)
            tracking_fn = os.path.join(folder, "tracking_data", base.replace(".mpg", "_track_data.npy"))
            tracking_fns += [tracking_fn]
            thumbnail_fn = os.path.join(folder, "thumbnails", base.replace(".mpg", ".jpg"))
            thumbnail_fns += [thumbnail_fn]
        # make copies into new folder
        for thumbnail_fn, fn in zip(thumbnail_fns, fns):
            base = os.path.basename(thumbnail_fn)
            new_fn = os.path.join(new_folder, base)
            if not os.path.exists(new_fn):
                save_thumbnail(fn, new_fn)
        position_selector = tracker_window(dirname=new_folder, num_markers=num_objects,
                                           fn=os.path.join(new_folder, 'nan_data.npy'))
        plt.show()
        nan_data = np.load(os.path.join(new_folder, 'nan_data.npy'))
        for fn, point in zip(tracking_fns, nan_data[0]):
            track = np.load(fn)
            track[:] = np.array(point)[np.newaxis, np.newaxis, :]
            np.save(fn, track)

def consolidate_tracks(fns, num_objects=1):
    nans_fns = []
    tracking_fns = []
    thumbnail_fns = []
    folder = os.path.dirname(fns[0])
    for fn in fns:
        base = os.path.basename(fn)
        tracking_fn = os.path.join(folder, "tracking_data", base.replace(".mpg", "_track_data.npy"))
        tracking_fns += [tracking_fn]
        thumbnail_fn = os.path.join(folder, "thumbnails", base.replace(".mpg", ".jpg"))
        thumbnail_fns += [thumbnail_fn]
    for num, (fn, tracking_fn, thumbnail_fn) in enumerate(zip(fns, tracking_fns, thumbnail_fns)):
        tracks = np.load(tracking_fn)
        # consolidate multiple tracks into 1 using Kalman Filter
        diff = tracks[1] - tracks[0]
        dists = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)
        tracks = tracks.transpose((1, 0, 2))
        thresh = np.percentile(dists, 99)
        kalman_filter = Kalman_Filter(num_objects, jerk_std=10)
        if dists[0] < thresh:
            if num_objects == 1:
                center = tracks[0].mean(0)
                kalman_filter.add_starting_points(center[np.newaxis])
            else:
                kalman_filter.add_starting_points(tracks[0])
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
        np.save(tracking_fn.replace("track_data", "position_data"), arr)
        print_progress(num, len(fns))


if __name__ == "__main__":
    num_objects = int(input("How many objects are you actually interested? "))
    print("Select the video files you want to motion track:")
    file_UI = FileSelector()
    file_UI.close()
    fns = file_UI.files
    # 1. replace nans with GUI-selected points
    fix_nans(fns, num_objects=num_objects)
    # 2. consolidate tracks down to the desired number of points
    consolidate_tracks(fns, num_objects=num_objects)
