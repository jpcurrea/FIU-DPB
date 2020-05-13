from motion_tracker import *
import os
from scipy import signal, optimize, ndimage
from shutil import copyfile
from scipy import stats

def fix_nans(fns, num_objects=1, tracking_folder="tracking_data", thumbnail_folder="thumbnails"):
    nans_fns = []
    tracking_fns = []
    thumbnail_fns = []
    # nans = []
    folder = os.path.dirname(fns[0])
    thumbnail_folder = os.path.join(folder, thumbnail_folder)
    tracking_folder = os.path.join(folder, tracking_folder)
    for fn in fns:
        base = os.path.basename(fn)
        vid_ftype = base.split(".")[-1]
        tracking_fns.append(os.path.join(tracking_folder, base.replace(f".{vid_ftype}", "_track_data.npy")))
        thumbnail_fns.append(os.path.join(thumbnail_folder, base.replace(f".{vid_ftype}", ".jpg")))
    for num, (fn, tracking_fn, thumbnail_fn) in enumerate(zip(fns, tracking_fns, thumbnail_fns)):
        tracks = np.load(tracking_fn)
        nans = np.isnan(tracks)
        nan_measure = nans.mean()
        if nan_measure > .95:
            nans_fns.append(fn)
        elif nan_measure > 0:
            # partial nans will only have nans on the front end of the time series,
            # meaning that they were still for a portion of the video. Extend that first
            # frame to the beginning of the time series
            nans = nans.mean((0, -1))
            no_nans = nans == 0
            first_frame = np.where(no_nans)[0][0]
            tracks[:, :first_frame] = tracks[:, first_frame][:, np.newaxis, :]
            np.save(tracking_fn, tracks)
    # for those with all nans, use GUI to select single location
    if len(nans_fns) > 0:
        # make a folder for nans
        folder = os.path.dirname(fn)
        nan_folder = os.path.join(folder, "nan_videos")
        if not os.path.isdir(nan_folder):
            os.mkdir(nan_folder)
        # get thumbnail file names to move
        thumbnail_fns = []
        tracking_fns = []
        for fn in nans_fns:
            base = os.path.basename(fn)
            vid_ftype = base.split(".")[-1]
            tracking_fns.append(os.path.join(tracking_folder, base.replace(f".{vid_ftype}", "_track_data.npy")))
            thumbnail_fns.append(os.path.join(thumbnail_folder, base.replace(f".{vid_ftype}", ".jpg")))
        # make copies into new folder
        for thumbnail_fn, fn in zip(thumbnail_fns, nans_fns):
            base = os.path.basename(thumbnail_fn)
            new_fn = os.path.join(nan_folder, base)
            if not os.path.exists(new_fn):
                save_thumbnail(fn, new_fn)
        position_selector = tracker_window(dirname=nan_folder, num_markers=num_objects,
                                           fn=os.path.join(nan_folder, 'nan_data.npy'))
        plt.show()
        nan_data = np.load(os.path.join(nan_folder, 'nan_data.npy'))
        for fn, point in zip(tracking_fns, nan_data[0]):
            track = np.load(fn)
            track[:] = np.array(point)[np.newaxis, np.newaxis, :]
            np.save(fn, track)

def consolidate_tracks(fns, num_objects=1, thumbnail_folder="thumbnails",
                       position_folder="position_data", tracking_folder="tracking_data",
                       speed_limit=15, min_length=15, save_video=True):
    nans_fns = []
    tracking_fns = []
    thumbnail_fns = []
    new_position_fns = []
    folder = os.path.dirname(fns[0])
    thumbnail_folder = os.path.join(folder, thumbnail_folder)
    tracking_folder = os.path.join(folder, tracking_folder)
    position_folder = os.path.join(folder, position_folder)
    if not os.path.isdir(position_folder):
        os.mkdir(position_folder)
    for fn in fns:
        base = os.path.basename(fn)
        vid_ftype = base.split('.')[-1]
        tracking_fns.append(os.path.join(tracking_folder, base.replace(f".{vid_ftype}", "_track_data.npy")))
        thumbnail_fns.append(os.path.join(thumbnail_folder, base.replace(f".{vid_ftype}", ".jpg")))
        new_position_fns.append(os.path.join(position_folder, base.replace(f".{vid_ftype}", "_position_data.npy")))
    calib = np.load(os.path.join(thumbnail_folder, "calibration_lengths.npy"))
    calib_order = np.load(os.path.join(thumbnail_folder, "order.npy"))
    for num, (fn, tracking_fn, thumbnail_fn, new_fn) in enumerate(
            zip(fns, tracking_fns, thumbnail_fns, new_position_fns)):
        i = []
        base = os.path.basename(thumbnail_fn)
        for calib_fn in calib_order:
            i.append(base in calib_fn)
        if sum(i) > 0 and os.path.exists(new_fn) is False:
            tracks = np.load(tracking_fn)
            if tracks.ndim == 2:
                tracks = tracks[np.newaxis]
            num_points, dur, ndim = tracks.shape
            pixel_length = calib[i][0]
            if num_points > 1:
                # remove points with zero velocity: if object is truly still, Kalman will follow without problem
                velocity = np.linalg.norm(np.diff(tracks, axis=1), axis=-1)
                # tracks[:, 1:][velocity == 0] = np.nan
                # consolidate multiple tracks into 1 using Kalman Filter
                # problematic! only works if num_points = 2
                # to do this properly, we should look at all pairs of points per frame
                # look for when tracks are close enough, and average them together
                # find all points within min_length of each other
                within_dists = np.linalg.norm(tracks[np.newaxis, :] - tracks[:, np.newaxis], axis=-1) < min_length
                for frame in tracks[:, 1:].transpose((1, 0, 2)):
                    tree = spatial.KDTree(frame)
                    dists, inds = tree.query(frame, k=2)
                    # dists, inds = dists[:, 1], inds[:, 1]
                    too_close = dists[:, 1] <= min_length
                    while any(too_close):
                        mappings = inds[too_close]
                        mappings.sort(axis=1)
                        mappings = sorted(set([tuple(mapping) for mapping in mappings.tolist()]))
                        for (ma, mb) in mappings:
                            pa, pb = frame[ma], frame[mb]
                            frame[ma] = .5 * (pa + pb)
                            frame[mb] = np.nan
                        tree = spatial.KDTree(frame)
                        dists, inds = tree.query(frame, k=2)
                        too_close = dists[:, 1] <= min_length
                for track in tracks:
                    speed = np.linalg.norm(np.diff(track, axis=0), axis=-1)
                    too_fast = (speed > 25).astype(int)
                    changes = np.diff(too_fast)
                    starts = np.where(changes == 1)[0]
                    stops = np.where(changes == -1)[0]
                    if len(starts) > 0 or len(stops) > 0:
                        if len(stops) == 0:
                            stops = np.append(stops, -1)
                        elif len(starts) == 0:
                            starts = np.append(0, starts)
                        if stops.min() < starts.min() and stops.min() > 0:
                            starts = np.append(0, starts)
                        if starts.max() > stops.max():
                            stops = np.append(stops, -1)
                        for start, stop in zip(starts, stops):
                            if stop > 0:
                                stop += 1
                            if start > 0:
                                start -= 1
                            track[start:stop] = np.nan

                # nans = np.isnan(tracks)
                # nans_per_track = nans.mean((1, 2))
                # i = np.argsort(nans_per_track)
                # tracks = tracks[i[:5]]
                # dists = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
                close_enough = dists < 25
                # tracks[:, close_enough] = tracks[:, close_enough][np.newaxis]

                # when different, split into contiguous sequences
                # different = np.squeeze(close_enough == False).astype(int)
                # changes = np.diff(different)
                # starts = np.where(changes == 1)[0]
                # stops = np.where(changes == -1)[0]
                if len(starts) > 0 or len(stops) > 0:
                    if len(stops) == 0:
                        stops = np.append(stops, -1)
                    elif len(starts) == 0:
                        starts = np.append(0, starts)
                    if stops.min() < starts.min() and stops.min() > 0:
                        starts = np.append(0, starts)
                    if starts.max() > stops.max():
                        stops = np.append(stops, -1)
                    for start, stop in zip(starts, stops):
                        if stop > 0:
                            stop += 1
                        if start > 0:
                            start -= 1
                        segment = tracks[:, start:stop]
                        velocity = np.diff(segment, axis=1)
                        speed = np.linalg.norm(velocity, axis=-1)
                        # are there any points moving too quickly or too slowly?
                        try:
                            max_speeds = speed.max(1)
                        except:
                            import pdb; pdb.set_trace()
                        too_fast = max_speeds > speed_limit/3
                        mean_speeds = speed.mean(1)
                        ts, ps = stats.ttest_1samp(velocity, 0, axis=1)
                        too_slow = ps > .0001
                        too_slow = np.logical_and(too_slow[:, 0], too_slow[:, 1])
                        problematic = np.logical_or(too_fast, too_slow)
                        if any(problematic):
                            if all(problematic):
                                strikes = (speed > speed_limit).sum(1)
                                slow_enough = strikes == min(strikes)
                                too_fast = slow_enough == False
                                if any(too_fast):
                                    segment[too_fast] = segment[slow_enough].mean(0)[np.newaxis]
                                else:
                                    segment[:] = segment[slow_enough].mean(0)[np.newaxis]
                            else:
                                segment[problematic] = segment[problematic == False].mean(0)
            if num_points == 1:
                tracks = tracks.transpose((1, 0, 2))
                kalman_filter = Kalman_Filter(num_objects, jerk_std=25)
                kalman_filter.add_starting_points(tracks[0])
                for frame in tracks[1:]:
                    prediction = kalman_filter.get_prediction()
                    if np.isnan(frame).mean() < 1:
                        error = np.linalg.norm(prediction - frame, axis=-1)
                        if error < 50:
                            measurement = frame[0]
                        else:
                            measurement = np.empty(frame[0].shape)
                            measurement.fill(np.nan)
                    else:
                        measurement = np.empty(frame[0].shape)
                        measurement.fill(np.nan)
                    try:
                        kalman_filter.add_measurement(measurement[np.newaxis])
                    except:
                        import pdb; pdb.set_trace()
            else:
                dists = np.linalg.norm(np.diff(tracks, axis=1), axis=-1)[0]
                tracks = tracks.transpose((1, 0, 2))
                diff_no_nans = np.isnan(dists) == False
                thresh = np.percentile(dists[diff_no_nans], 99)
                kalman_filter = Kalman_Filter(num_objects, jerk_std=25)
                # if dists[0] <= thresh:
                if num_objects == 1:
                    center = tracks[0].mean(0)
                    kalman_filter.add_starting_points(center[np.newaxis])
                else:
                    kalman_filter.add_starting_points(tracks[0])
                for frame in tracks[1:]:
                    prediction = kalman_filter.get_prediction()
                    if np.isnan(frame).mean() < 1:
                        error = np.linalg.norm(prediction - frame, axis=-1)
                        if sum(error <= thresh) > 1:
                            measurement = frame[error <= thresh].mean(0)
                        else:
                            measurement = frame[np.argmin(error)]
                    else:
                        measurement = frame[0]
                    kalman_filter.add_measurement(measurement[np.newaxis])
                    # kalman_filter.add_measurement(frame)
            xs = np.squeeze(np.array(kalman_filter.Q_loc_estimateX))
            ys = np.squeeze(np.array(kalman_filter.Q_loc_estimateY))
            arr = np.array([xs, ys]).T
            nans = np.isnan(arr).mean()
            if nans > 0:
                breakpoint()
            np.save(new_fn, arr * pixel_length)
            if save_video:
                breakpoint()
                print(f"Saving video with tracks superimposed:\n")
                vid_ftype = fn.split(".")[-1]
                vid = np.squeeze(io.vread(fn, as_grey=True)).astype('int16')
                if arr.ndim > 2:
                    new_arr = arr.transpose((1, 0, 2))[..., [1, 0]]
                elif arr.ndim == 2:
                    new_arr = arr[:, np.newaxis]
                new_vid = make_video(vid, new_arr, point_length=7)
                io.vwrite(new_fn.replace(".npy", f".{vid_ftype}"), new_vid)
            print_progress(num, len(fns))


def get_ROI_data(fns, thumbnail_folder="thumbnails", position_folder="position_data"):
    folder = os.path.dirname(fns[0])
    position_folder = os.path.join(folder, position_folder)
    thumbnail_folder = os.path.join(folder, thumbnail_folder)
    thumbnail_order = np.load(os.path.join(thumbnail_folder, "order.npy"))
    roi_markers = np.load(os.path.join(thumbnail_folder, "ROI_markers.npy"))
    roi_radii = np.load(os.path.join(thumbnail_folder, "ROI_radii.npy"))
    roi_distances_folder = os.path.join(folder, "ROI_distances")
    calibration = np.load(os.path.join(thumbnail_folder, "calibration_lengths.npy"))
    if not os.path.isdir(roi_distances_folder):
        os.mkdir(roi_distances_folder)
    print("getting distances from ROIs")
    for num, fn in enumerate(fns):
        base = os.path.basename(fn)
        vid_ftype = base.split(".")[-1]
        thumbnail_fn = os.path.join(thumbnail_folder, base.replace(f".{vid_ftype}", ".jpg"))
        position_fn = os.path.join(position_folder, base.replace(f".{vid_ftype}", "_position_data.npy"))
        if thumbnail_fn in thumbnail_order:
            ind = np.where(thumbnail_order == thumbnail_fn)[0][0]
            rois = roi_markers[:, ind]
            # the x and y axes are swapped in the rois compared to position_data
            rois = rois[..., [1, 0]]
            pixel_length = calibration[ind]
            rois *= pixel_length    # convert to actual distances
            radius = roi_radii[ind]
            position_data = np.load(position_fn)
            diffs = position_data[np.newaxis] - rois[:, np.newaxis]
            dists = np.linalg.norm(diffs, axis=-1)
            new_fn = os.path.join(roi_distances_folder, base.replace(f".{vid_ftype}", "_roi_distances.npy"))
            np.save(new_fn, dists)
        print_progress(num + 1, len(fns))

if __name__ == "__main__":
    num_objects = int(input("How many objects are you actually interested? "))
    save_video = input(
        "Do you want to save a video of the final position data? Type 1 for "
        "yes and 0 for no: ")
    while save_video not in ["0", "1"]:
        save_video = input(
            "The response must be a 0 or a 1")
    save_video = bool(int(save_video))
    print("Select the video files you want to motion track:")
    file_UI = FileSelector()
    file_UI.close()
    fns = file_UI.files
    # os.chdir("/Volumes/Lab/av_isr_1/free_roam/")
    # fns = os.listdir()
    # fns = [os.path.abspath(fn) for fn in fns if fn.endswith(".mpg")]
    num_objects = 1
    save_video = False
    # 1. replace nans with GUI-selected points
    # fn = "Trial  1639.mpg"
    # fns = [fn]
    fix_nans(fns, num_objects=num_objects)
    # 2. consolidate tracks down to the desired number of points
    consolidate_tracks(fns, num_objects=num_objects, save_video=save_video)
    # 3. get time series of distances from ROIs
    get_ROI_data(fns)
