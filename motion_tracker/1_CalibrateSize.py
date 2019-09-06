from __init__ import *


class VideoCalibrator():
    def __init__(self, thumbnail_folder='thumbnails',
                 scale=10, video_files=None):
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

    def get_lengths(self):
        self.gui = distance_calibration_GUI(
            dirname=self.thumbnail_folder,
            scale=10)
        plt.show()
        self.lengths = self.gui.lengths


if __name__ == "__main__":
    video_calibrator = VideoCalibrator(scale=10)  # scale in centimeters
    video_calibrator.get_lengths()
