from . import *


# folder = "./test/"
# 0. make thumbnail for each file in a set of videos

# app = QApplication([])
# file_UI = fileSelector()
# file_UI.close()
# fns = file_UI.files

# parent_folder = os.path.dirname(fns[0])
# folder = os.path.join(parent_folder, "thumbnails")
# if os.path.isdir(folder) is False:
#     os.mkdir(folder)

# files = os.listdir(folder)
# files = [os.path.join(folder, fn) for fn in files]


# for fn in fns:
#     l = fn.split(".")
#     l[-1] = "jpg"
#     new_fn = ".".join(l)
#     new_fn = os.path.join(folder, new_fn)
#     base = os.path.basename(new_fn)
#     new_fn = os.path.join(folder, base)
#     if new_fn not in files:
#         print(new_fn)
#         cmd = [
#             "ffmpeg", "-y", "-i", fn,
#             "-vf", "select=gte(n\,100)",
#             "-vframes", "1", new_fn
#         ]
#         subprocess.call(cmd)

# 0. crop the files using a minimum bounding box around user-chosen points


# 2. offer GUI for getting a size reference from each thumbnail
# # (optional: if they don't do this, output will be in terms of pixels)
# calib = distance_calibration_GUI(dirname=folder, scale=121.92)
# plt.show()

# lengths = calib.lengths


# lengths = np.load(os.path.join(folder, "calibration_lengths.npy"))
# 3. offer GUI for selecting ROI's with optional hidden zones
# (optional: if they don't do this, output will be just x and y coordinates)
# roi = ROI_GUI(folder, num_points=2, pixel_length=lengths)
# plt.show()
# # 4. ask "how many moving objects are you actually interested in?"

# 5. consolidate the multiple trajectories using the following logic:
# # - label outlier speeds, which should come in pairs, and move points within those bounds to a new list, called aberations
# # - if len(hidden_zones) > 0, remove any lines in aberations that start and end in the hidden zones
# # - all trajectories should now correspond to the subject(s) of interest. If not, warn the user and offer a GUI to remove the faulty trajectories
# 6. save the x and y coordinates as one csv and npy file per video
# 7. if there are ROI's, calculate the distance of the subject from each ROI
