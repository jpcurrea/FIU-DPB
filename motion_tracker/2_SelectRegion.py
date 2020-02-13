from __init__ import *


class ROISelector():
    def __init__(self, thumbnail_folder='thumbnails',
                 radius=10, video_files=None,
                 roi_markers=None, roi_radii=None,
                 num_markers=5):
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication([])
        # grab the video files
        if video_files is None:
            print("Select the video files you want to measure:")
            file_UI = FileSelector()
            file_UI.close()
            self.video_files = file_UI.files
        else:
            self.video_files = video_files
        # find the parent directory
        assert len(self.video_files) > 0, "No files found!"
        self.folder = os.path.dirname(self.video_files[0])
        os.chdir(self.folder)
        # find/make thumbnail folder
        self.thumbnail_folder = os.path.join(self.folder, thumbnail_folder)
        if os.path.isdir(self.thumbnail_folder) is False:
            os.mkdir(self.thumbnail_folder)
        # get pixel conversion, if it exists
        calibration_fn = os.path.join(self.thumbnail_folder, "calibration_lengths.npy")
        if os.path.exists(calibration_fn):
            self.pixel_length = np.load(calibration_fn)
        else:
            self.pixel_length = 1
        # run through video files
        self.thumbnail_files = []
        for fn in self.video_files:
            base = os.path.basename(fn)
            ftype = base.split(".")[-1]
            base = base.replace(ftype, "jpg")
            new_fn = os.path.join(self.thumbnail_folder, base)
            # if thumbnail doesn't exist, make it
            if os.path.exists(new_fn) is False:
                save_thumbnail(fn, new_fn)
            self.thumbnail_files += [new_fn]
        self.num_markers = num_markers
        self.roi_markers = roi_markers
        self.roi_radii = roi_radii
        self.radius = radius

    def get_ROIs(self):
        if self.roi_markers is None:
            self.gui = ROI_GUI(
                num_markers=self.num_markers,
                dirname=self.thumbnail_folder,
                radius=self.radius,
                pixel_length=self.pixel_length)
            plt.show()
            self.roi_markers = self.gui.markers
            self.roi_radii = self.gui.radii


if __name__ == "__main__":
    num_markers = int(input("How many regions do you want to keep track of? "))
    # scale in centimeters
    roi_gui = ROISelector(radius=10, num_markers=num_markers)
    roi_gui.get_ROIs()
