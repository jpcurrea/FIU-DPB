from ROI_GUI import *


class VideoCropper():
    """A GUI for selecting a bounding rectangle in video thumbnails and then 
    rotating the videos and cropping them around that frame using skvideo and 
    ffmpeg.
    """

    def __init__(self, thumbnail_folder="thumbnails",
                 frames=None, cropped_folder='cropped',
                 video_files=None):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        if fns is None:
            print("Select the video files you want to crop and rotate:")
            file_UI = fileSelector()
            file_UI.close()
            self.video_files = file_UI.files
        else:
            self.video_files = video_files
        # figure out the parent directory
        self.folder = os.path.dirname(self.video_files[0])
        os.chdir(self.folder)
        self.cropped_folder = os.path.join(self.folder, cropped_folder)
        if os.path.isdir(self.cropped_folder) is False:
            os.mkdir(self.cropped_folder)
        self.thumbnail_folder = os.path.join(self.folder, thumbnail_folder)
        # if folder of thumbnails does not exist, make a folder
        self.thumbnail_folder = os.path.join(self.folder, thumbnail_folder)
        if os.path.isdir(self.thumbnail_folder) is False:
            os.mkdir(self.thumbnail_folder)
        # make a thumbnail for each video if we haven't already
        self.thumbnail_files = os.listdir(self.thumbnail_folder)
        self.thumbnail_files = [
            fn for fn in self.thumbnail_files if fn.endswith(".jpg")]
        self.thumbnail_files = [
            os.path.join(self.thumbnail_folder, fn) for fn in self.thumbnail_files]
        for vid_file in self.video_files:
            new_fn = os.path.basename(vid_file)
            ftype = new_fn.split(".")[-1]
            new_fn = new_fn.replace(ftype, "jpg")
            new_fn = os.path.join(self.thumbnail_folder, new_fn)
            if new_fn not in self.thumbnail_files:
                save_thumbnail(vid_file, new_fn)
        # if frame points are not input, use rectangle_bounding_box_GUI
        frames_fn = os.path.join(self.thumbnail_folder, "frame_corners.npy")
        if os.path.exists(frames_fn):
            frames = np.load(frames_fn)
            self.frames_fn = frames_fn
        elif frames is None:
            self.gui = rectangle_bounding_box_GUI(self.thumbnail_folder)
            plt.show()
            frames = self.gui.frames
        self.frames = frames

    def crop_and_rotate(self):
        print("cropping and rotating:")
        for vid_fn, frame in zip(self.video_files, self.frames.transpose((1, 0, 2))):
            # grab video data
            base = os.path.basename(vid_fn)
            print(vid_fn)
            ftype = "." + base.split(".")[-1]
            new_fn = base.replace(ftype, f"_cropped{ftype}")
            new_fn = os.path.join(self.cropped_folder, new_fn)
            if os.path.exists(new_fn) is False:
                vid = io.vread(vid_fn)
                # first find lower and upper bounds for initial cropping
                lower = np.floor(frame.min(0)).astype(int)
                upper = np.ceil(frame.max(0)).astype(int)
                # initial cropping:
                vid = vid[:, lower[1]:upper[1], lower[0]:upper[0]]
                frame = frame - lower
                # find small edges to calculate angle of rotation
                tree = spatial.KDTree(frame)
                dists, indexes = tree.query(frame, k=3)
                small_inds = np.copy(indexes[:, :2])
                small_inds.sort(1)
                small_inds = np.unique(small_inds, axis=0)
                mid_points = []
                for ind in small_inds:
                    mid_points += [np.mean(frame[ind], axis=0)]
                mid_points = np.array(mid_points)
                # find the angle of the midline
                xs, ys = mid_points.T
                angle_rad = np.arctan2(np.diff(ys).mean(), np.diff(xs).mean())
                angle_deg = angle_rad * 180 / np.pi
                # rotate the video
                vid = ndimage.rotate(vid, angle_deg, axes=(1, 2))
                # and rotate the frame
                new_frame = rotate(frame - frame.mean(0), angle_rad)
                height, width = vid.shape[1:3]
                new_frame += np.array([width/2., height/2.])
                new_lower = np.floor(new_frame.min(0)).astype(int)
                new_upper = np.ceil(new_frame.max(0)).astype(int)
                # final crop of the video using the rotate frame
                vid = vid[:, new_lower[1]:new_upper[1],
                          new_lower[0]:new_upper[0]]
                # save video in cropped_folder
                io.vwrite(new_fn, vid)
                print(f"new video saved at {new_fn}")
                del vid


if __name__ == "__main__":
    crop = VideoCropper()
    crop.crop_and_rotate()
