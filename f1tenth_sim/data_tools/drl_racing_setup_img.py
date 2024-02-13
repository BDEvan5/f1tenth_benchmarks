import pandas as pd 
import numpy as np
import os
import torch
from f1tenth_sim.run_scripts.run_functions import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from f1tenth_sim.utils.MapData import MapData
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.track_utils import CentreLine
from f1tenth_sim.localmap_racing.LocalMap import LocalMap
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def ensure_path_exists(path):
    if not os.path.exists(path): 
        os.mkdir(path)

save_path = 'Data/MiscImages/'
ensure_path_exists(save_path)

def rotate_bound(image, angle):
    import cv2
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), borderValue=(1, 1, 1))


def plot_scan(scan_pts, x, y):
    dists = np.linalg.norm(np.diff(scan_pts, axis=0), axis=1)
    idx = np.argmax(dists)
    ind_arr = np.linspace(0, len(scan_pts)-1, 50, dtype=int)
    ind_arr = np.sort(np.append(ind_arr, [idx, idx+1]))

    poly_pts = np.array([scan_pts[ind] for ind in ind_arr])
    poly_pts = np.insert(poly_pts, 0, [x, y], axis=0)
    poly = Polygon(poly_pts, color=free_speech, alpha=0.3)
    plt.gca().add_patch(poly)

    # plt.plot(scan_pts[idx, 0], scan_pts[idx, 1], color='red', marker='o', markersize=20)
    # plt.plot(scan_pts[idx+1, 0], scan_pts[idx+1, 1], color='red', marker='o', markersize=20)

def set_axis_limits(scan_pts):
    plt.gca().axis('off')
    xa = 3
    yb = 10
    x_min = np.min(scan_pts[:, 0]) - xa
    x_max = np.max(scan_pts[:, 0]) + xa
    y_min = np.min(scan_pts[:, 1]) - yb
    y_max = np.max(scan_pts[:, 1]) + yb
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])


def add_car_picture(x, y):
    img = plt.imread("f1tenth_sim/data_tools/RacingCar.png", format='png')
    img = rotate_bound(img, -14)
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (x+0.5, y-3.8), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

# inds = [317, 317+8]
inds = [309, 317]
planner_name = f"LocalMapPP"
test_id = "mu60"
map_name = "aut"

root = f"Logs/{planner_name}/"
localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
history = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")

map_data = MapData(map_name)
origin = map_data.map_origin[:2]

fig = plt.figure(1, figsize=(7, 2))
n_graphs = 2
ax1 = plt.subplot(1, n_graphs, 1)
ax2 = plt.subplot(1, n_graphs, 2)

axs = [ax1, ax2]

for z in range(2):
    plt.sca(axs[z])
    i = inds[z]
    scan = scans[i]

    position = history[i, 0:2]
    heading = history[i, 4]
    map_data.plot_map_img()
    x, y = map_data.xy2rc(position[0], position[1])

    angles = np.linspace(-4.7/2, 4.7/2, 1080)
    sines = np.sin(angles)
    cosines = np.cos(angles)
    xs, ys = cosines * scan, sines * scan
    scan_pts = np.column_stack((xs, ys))

    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])

    scan_pts = np.matmul(rotation, scan_pts.T).T
    scan_pts = scan_pts + position
    scan_pts = (scan_pts - origin) / map_data.map_resolution

    xs, ys = map_data.pts2rc(history[:i, 0:2])
    plt.plot(xs, ys, color=free_speech, linewidth=2)

    normalised_scan = np.array(scan / np.max(scan) * 100, dtype=int)
    plt.scatter(scan_pts[:, 0], scan_pts[:, 1], c=normalised_scan, cmap=cm.get_cmap("gist_rainbow"))
    plot_scan(scan_pts, x, y)
    add_car_picture(x, y)

    set_axis_limits(scan_pts) # change to use same points
    


plt.tight_layout()
plt.rcParams['pdf.use14corefonts'] = True

plt.savefig(save_path + f"drl_racing_setup.svg", bbox_inches='tight', pad_inches=0.01)
# plt.savefig(save_path + f"drl_racing_setup_{i}.pdf", bbox_inches='tight', pad_inches=0.01)


