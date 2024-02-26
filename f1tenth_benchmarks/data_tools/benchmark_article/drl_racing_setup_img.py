import pandas as pd 
import numpy as np
import os
import torch
from f1tenth_benchmarks.run_scripts.run_functions import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from f1tenth_benchmarks.utils.MapData import MapData
from f1tenth_benchmarks.data_tools.plotting_utils import *
from f1tenth_benchmarks.utils.track_utils import CentreLine
from f1tenth_benchmarks.localmap_racing.LocalMap import LocalMap
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.cm as cm


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
    xa = 15
    yb = 8
    x_min = np.min(scan_pts[:, 0]) - xa + 5
    x_max = np.max(scan_pts[:, 0]) + xa
    y_min = np.min(scan_pts[:, 1]) - yb
    y_max = np.max(scan_pts[:, 1]) + yb
    plt.ylim([y_min, y_max])
    plt.xlim([x_min, x_max])


def add_car_picture(x, y, heading_angle):
    img = plt.imread("f1tenth_benchmarks/data_tools/RacingCar.png", format='png')
    img = rotate_bound(img, heading_angle)
    # img = rotate_bound(img, 74)
    oi = OffsetImage(img, zoom=0.3)
    ab = AnnotationBbox(oi, (x+15, y), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)


start = 27
space = 10
inds = [start + space, start]
planner_name = f"LocalMapPP"
test_id = "mu60"
map_name = "aut"

root = f"Logs/{planner_name}/"
localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
history = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")

map_data = MapData(map_name)
origin = map_data.map_origin[:2]

fig = plt.figure(1, figsize=(7, 2), clip_on=True, clip_box='round')
n_graphs = 2
ax1 = plt.subplot(1, n_graphs, 1)
ax2 = plt.subplot(1, n_graphs, 2)

axs = [ax1, ax2]

scan_pt_list = []

for z in range(2):
    plt.sca(axs[z])
    i = inds[z]
    scan = scans[i]

    position = history[i, 0:2]
    heading = history[i, 4]
    map_data.plot_map_img_left_right_flip()
    x, y = map_data.xy2rc(position[0], position[1])
    x = map_data.map_width - x
    # plt.plot(x, y, color='red', marker='o', markersize=10)

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

    scan_pts[:, 0] = map_data.map_width - scan_pts[:, 0]

    xs, ys = map_data.pts2rc(history[:i, 0:2])
    xs = map_data.map_width - xs
    plt.plot(xs, ys, color=free_speech, linewidth=2)

    normalised_scan = np.array(scan / np.max(scan) * 100, dtype=int)
    plt.scatter(scan_pts[:, 0], scan_pts[:, 1], c=normalised_scan, cmap=cm.get_cmap("gist_rainbow"))
    car_pos = np.array([x, y])
    plot_scan(scan_pts, car_pos[0], car_pos[1])
    car_heading = np.rad2deg(heading-np.pi/2)
    add_car_picture(car_pos[0], car_pos[1], car_heading)

    if z == 0:
        arrow_size = 22
        plt.arrow(car_pos[0], car_pos[1], arrow_size*np.cos(-heading-np.pi), arrow_size*np.sin(-heading-np.pi), width=2, head_width=8, head_length=6, fc='r', ec='r', zorder=10)

    scan_pt_list.append(scan_pts)

scan_pts = np.vstack(scan_pt_list)
set_axis_limits(scan_pts) # change to use same points
plt.sca(axs[0])
set_axis_limits(scan_pts) # change to use same points


x_txt = car_pos[0] - 80
y_txt = car_pos[1] - 45
ax1.text(x=x_txt, y=y_txt, s="Current timestep", fontsize=11, color='black', fontdict={'weight': 'bold'}, backgroundcolor='white', bbox={'facecolor': 'white', 'alpha': 0.9, "boxstyle": "round", "edgecolor": "white"})
ax2.text(x=x_txt, y=y_txt, s="Previous timestep", fontsize=11, color='black', fontdict={'weight': 'bold'}, backgroundcolor='white', bbox={'facecolor': 'white', 'alpha': 0.9, "boxstyle": "round", "edgecolor": "white"})
ax1.text(x=car_pos[0] -95, y=car_pos[1] - 7, s="Vehicle\nspeed", fontsize=9, color='black', backgroundcolor='white', bbox={'facecolor': 'white', 'alpha': 0.5, "boxstyle": "round", "edgecolor": "white"}, va='center', ha='center')


plt.tight_layout()
plt.rcParams['pdf.use14corefonts'] = True

plt.savefig(save_path + f"drl_racing_setup.svg", bbox_inches='tight', pad_inches=0.01)
plt.savefig(save_path + f"drl_racing_setup.pdf", bbox_inches='tight', pad_inches=0.01)


