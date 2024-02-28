import numpy as np
import os
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
    poly = Polygon(poly_pts, color=free_speech, alpha=0.15)
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
    oi = OffsetImage(img, zoom=0.4)
    ab = AnnotationBbox(oi, (x, y), xycoords='data', frameon=False)
    plt.gca().add_artist(ab)


planner_name = f"LocalMapPP"
test_id = "mu60"
# map_name = "aut"
map_name = "esp"
# map_name = "gbr"
# map_name = "mco"

root = f"Logs/{planner_name}/"
localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
history = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")

print(len(history))

map_data = MapData(map_name)
origin = map_data.map_origin[:2]

fig = plt.figure(1, figsize=(3, 4))
n_graphs = 1
ax1 = plt.subplot(1, n_graphs, 1)

scan_pt_list = []

# i=78
# i=469
i=635 # good for ESP
# i = 840
# i = 878
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

car_pos = np.array([x, y])
plot_scan(scan_pts, car_pos[0], car_pos[1])
normalised_scan = np.array(scan / np.max(scan) * 100, dtype=int)
plt.scatter(scan_pts[:, 0], scan_pts[:, 1], c=normalised_scan, cmap=cm.get_cmap("gist_rainbow"))
car_heading = np.rad2deg(heading-np.pi/2)
print(car_heading)
add_car_picture(car_pos[0], car_pos[1], -car_heading)


set_axis_limits(scan_pts) # change to use same points

x_txt = car_pos[0] + 10
y_txt = car_pos[1] + 50
# plt.text(x=x_txt, y=y_txt, s="Racing car\non a track", fontsize=12, color='black', fontdict={'weight': 'bold'}, backgroundcolor='white', bbox={'facecolor': 'white', 'alpha': 0.9, "boxstyle": "round", "edgecolor": "white"})

# plt.gca().set_aspect('auto')

plt.tight_layout()
plt.rcParams['pdf.use14corefonts'] = True

plt.savefig(save_path + f"front_img2.svg", bbox_inches='tight', pad_inches=0.01)
plt.savefig(save_path + f"front_img2.pdf", bbox_inches='tight', pad_inches=0.01)


