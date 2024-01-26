import numpy as np 
from matplotlib import pyplot as plt

from f1tenth_sim.localmap_racing.LocalMap import LocalMap
from f1tenth_sim.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_sim.data_tools.plotting_utils import *
from f1tenth_sim.utils.MapData import MapData


from matplotlib.patches import RegularPolygon
from matplotlib.patches import Polygon, Circle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.collections import LineCollection
from matplotlib.transforms import Affine2D

text_size = 9
text_size2 = 14



def make_pipeline(planner_name, test_id, i, map_name):
    root = f"Logs/{planner_name}/"
    localmap_data_path = root + f"RawData_{test_id}/LocalMapData_{test_id}/"
    history = np.load(root + f"RawData_{test_id}/SimLog_{map_name}_0.npy")
    scans = np.load(root + f"RawData_{test_id}/ScanLog_{map_name}_0.npy")
    save_path = root + f"LocalMapGeneration_{test_id}/"
    img_path = root

    states = history[:, 0:7]

    position = states[i, 0:2]
    heading = states[i, 4]

    map_data = MapData(map_name)
    origin = map_data.map_origin[:2]
    local_track = np.load(localmap_data_path + f"local_map_{i}.npy")
    line_1 = np.load(localmap_data_path + f"line1_{i}.npy")
    line_2 = np.load(localmap_data_path + f"line2_{i}.npy")
    boundaries = np.load(localmap_data_path + f"boundaries_{i}.npy")
    boundary_extension= np.load(localmap_data_path + f"boundExtension_{i}.npy") 

    scan = scans[i]

    plt.figure(1, figsize=(10, 2.5))
    # plt.figure(1, figsize=(6, 4.5))
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(1, 4, wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    # ax1 = plt.subplot(1, 4, 1)
    # ax2 = plt.subplot(1, 4, 2)
    # ax3 = plt.subplot(1, 4, 3)
    # ax4 = plt.subplot(1, 4, 4)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[0, 2])
    ax4 = plt.subplot(gs[0, 3])
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.figure(1, figsize=(6, 4.5))
    # # plt.figure(1, figsize=(6, 4.5))
    # ax1 = plt.subplot(2, 2, 1)
    # ax2 = plt.subplot(2, 2, 2)
    # ax3 = plt.subplot(2, 2, 3)
    # ax4 = plt.subplot(2, 2, 4)

    plt.sca(ax1)
    map_data.plot_map_img_T()
    x, y = map_data.xy2rc(states[i, 0], states[i, 1])

    def add_car_picture():
        img = plt.imread("f1tenth_sim/data_tools/RacingCar.png", format='png')
        img = rotate_bound(img, 0)
        oi = OffsetImage(img, zoom=0.35)
        ab = AnnotationBbox(oi, (y-0.8, x-8.2), xycoords='data', frameon=False)
        # ab = AnnotationBbox(oi, (x-12, y+4), xycoords='data', frameon=False)
        plt.gca().add_artist(ab)

    def add_circle(number):
        circle = Circle((y-16, x+46), 8, color=lighter_purple, alpha=0.5)
        plt.gca().add_patch(circle)
        plt.text(y-16, x+46, f"{number}", fontsize=text_size2, color='k', horizontalalignment='center', verticalalignment='center')

    angles = np.linspace(-4.7/2, 4.7/2, 1080)
    sines = np.sin(angles)
    cosines = np.cos(angles)
    xs, ys = cosines * scan, sines * scan
    scan_pts = np.column_stack((xs, ys))

    def plot_scan():
        ind_arr = [0, 100, 300, 452, 500, 600, 650, 700, 760, 783, 784, 800, 900, 1079]
        poly_pts = np.array([scan_pts[ind] for ind in ind_arr])
        poly_pts = np.insert(poly_pts, 0, [x, y], axis=0)
        poly_pts = np.flip(poly_pts, axis=1)
        # poly = Polygon(poly_pts, color=free_speech, alpha=0.3)
        poly = Polygon(poly_pts, color=free_speech, alpha=0.2)
        plt.gca().add_patch(poly)

        # ind = 783
        # plt.plot(scan_pts[ind, 1], scan_pts[ind, 0], color='red', marker='o', markersize=20)

    def set_axis_limits(scan_pts):
        plt.gca().axis('off')
        b = 10
        x_min = np.min(scan_pts[:, 0]) - 6
        x_max = np.max(scan_pts[:, 0]) + 6
        y_min = np.min(scan_pts[:, 1]) - b
        y_max = np.max(scan_pts[:, 1]) + b
        plt.xlim([y_min, y_max])
        plt.ylim([x_min, x_max])

    rotation = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])


    scan_pts = np.matmul(rotation, scan_pts.T).T
    scan_pts = scan_pts + position
    scan_pts = (scan_pts - origin) / map_data.map_resolution

    x, y = map_data.xy2rc(states[i, 0], states[i, 1])
    plt.plot(y, x, 'x', color='red')

    plt.plot(scan_pts[:, 1], scan_pts[:, 0], '.', color=free_speech, label="LiDAR Scan", markersize=3)
    # plt.plot(scan_pts[:, 0], scan_pts[:, 1], '.', color=color_pallette[0], label="LiDAR Scan", markersize=3)
    plot_scan()
    add_car_picture()

    x_len = 25
    # angle = 70 * np.pi/180
    angle = np.pi/2
    xs = [y-np.sin(angle)*x_len, y+np.sin(angle)*x_len]
    ys = [x-np.cos(angle)*x_len, x+np.cos(angle)*x_len]
    plt.plot(xs, ys, color=nartjie, linewidth=2)
    y_len = 50
    # angle = -20 * np.pi/180
    angle = 0
    xs = [y-np.sin(angle)*y_len, y+np.sin(angle)*y_len]
    ys = [x-np.cos(angle)*y_len, x+np.cos(angle)*y_len]
    plt.plot(xs, ys, color=nartjie, linewidth=2)

    leg = plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.73, 0.1), fancybox=True, shadow=True)
    # leg = plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True)
    leg.legendHandles[0]._markersize = 12
    # leg.legend_handles[0]._markersize = 12
    set_axis_limits(scan_pts)
    # plt.title("1. Receive LiDAR scan")
    add_circle(1)

    plt.sca(ax2)

    # lm_generator = LocalMapGenerator("Data/LocalCenter_1/", test_id, save_data=False)

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

    map_data.plot_map_img_T()
    long_side = np.copy(line_1)[::2]
    short_side = np.copy(line_2)[::2]

    long_side = (np.matmul(rotation, long_side.T).T + position - origin ) / map_data.map_resolution
    short_side = (np.matmul(rotation, short_side.T).T + position - origin ) / map_data.map_resolution

    # long_side = long_side1[::2]
    # short_side = short_side1[::2]

    # plt.plot(scan_pts[:, 1], scan_pts[:, 0], 'o', color=color_pallette[0], alpha=0.9)
    boundary_color = high_pink
    plt.plot(long_side[:, 1], long_side[:, 0], '-o', markersize=6, color=boundary_color, linewidth=1, label="Long boundary")
    plt.plot(short_side[:, 1], short_side[:, 0], '-*', markersize=8, color=boundary_color, label="Short boundary", linewidth=1)

    add_car_picture()
    plot_scan()

    set_axis_limits(scan_pts)

    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.72, 0.32))
    # plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, -0.05))
    # plt.title("2. Extract Boundaries")
    add_circle(2)


    plt.sca(ax3)

    l1 = boundaries[:, :2][::2]
    l2 = boundaries[:, 2:][::2]
    l1 = (np.matmul(rotation, l1.T).T + position - origin ) / map_data.map_resolution
    l2 = (np.matmul(rotation, l2.T).T + position - origin ) / map_data.map_resolution


    be1 = boundary_extension[:, :2][2::2]
    be2 = boundary_extension[:, 2:][2::2]
    be1 = (np.matmul(rotation, be1.T).T + position - origin ) / map_data.map_resolution
    be2 = (np.matmul(rotation, be2.T).T + position - origin ) / map_data.map_resolution

    map_data = MapData(map_name)
    map_data.plot_map_img_light_T()

    boundary_color = 'black'
    # boundary_color = color_pallette[2]
    plt.plot(l1[:, 1], l1[:, 0], '-', color=boundary_color, linewidth=2, alpha=1)
    plt.plot(l2[:, 1], l2[:, 0], '-', color=boundary_color, linewidth=2, alpha=1)
    # plt.plot(short_side[:, 0], short_side[:, 1], '-', color=boundary_color, label="Edges", linewidth=2, alpha=0.7)

    plt.plot([l1[-1, 1], be1[0, 1]], [l1[-1, 0], be1[0, 0]], '-', color=boundary_color, linewidth=2)
    # plt.plot([l2[-1, 1], be2[0, 1]], [l2[-1, 0], be2[0, 0]], '-', color=london_square, linewidth=2)



    add_car_picture()

    for z in range(1, len(l1)):
        n_xs = [l1[z, 0], l2[z, 0]]
        n_ys = [l1[z, 1], l2[z, 1]]
        if z == 1:
            plt.plot(n_ys, n_xs, '-', color=sweedish_green, linewidth=2, label="Calculated segments")
        else:
            plt.plot(n_ys, n_xs, '-', color=sweedish_green, linewidth=2)
    

    for z in range(0, len(be1)):
        n_xs = [be1[z, 0], be2[z, 0]]
        n_ys = [be1[z, 1], be2[z, 1]]
        if z == 0: 
            plt.plot(n_ys, n_xs, '-', color=jade_dust, linewidth=2, label="Projected segments")
        else:
            plt.plot(n_ys, n_xs, '-', color=jade_dust, linewidth=2)
    plt.plot(be1[:, 1], be1[:, 0], '-', color=boundary_color, linewidth=2)
    plt.plot(be2[:, 1], be2[:, 0], 'o', color=jade_dust, linewidth=2, markersize=8)
    # plt.plot(be2[:, 1], be2[:, 0], 'o', color=london_square, linewidth=2, markersize=8)
    # plt.plot(be2[:, 1], be2[:, 0], 'o', color=london_square, linewidth=2, markersize=8, label="Estimated boundary")
    # plt.text(225, 378, "Estimated\nboundary\npoints")

    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.66, 0.12))
    # plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, -0.08))
    set_axis_limits(scan_pts)
    # plt.title("3. Extract track segments")
    add_circle(3)


    plt.sca(ax4)


    ws = local_track[:, 2:] * 20
    st = (np.matmul(rotation, local_track[:, :2].T).T + position - origin) / map_data.map_resolution
    st_in = np.concatenate((st, ws), axis=1)
    st_lm = LocalMap(st_in)

    map_data.plot_map_img_light_T()

    # for i in range(len(smooth_track)):
    pts = st_lm.track[:, :2]
    l1 = pts + st_lm.nvecs * st_lm.track[:, 2][:, None]
    l2 = pts - st_lm.nvecs * st_lm.track[:, 3][:, None]
    
    good_grey = "#485460"
    for z in range(1, len(l1)):
        n_xs = [l1[z, 0], l2[z, 0]]
        n_ys = [l1[z, 1], l2[z, 1]]
        if z == 1:
            plt.plot(n_ys, n_xs, '-', color=good_grey, linewidth=2, label="Normal vectors")
        else:
            plt.plot(n_ys, n_xs, '-', color=good_grey, linewidth=2)


    plt.plot(st[1:, 1], st[1:, 0], '-', color=red_orange, label="Centre line", linewidth=4)
    # plt.plot(st[1:, 1], st[1:, 0], '-', color=sweedish_green, label="Centre line", linewidth=3)
    add_car_picture()

    set_axis_limits(scan_pts)
    plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.72, 0.12))
    # plt.legend(loc='center', fontsize=text_size, bbox_to_anchor=(0.5, -0.08))
    # plt.title("5. Regularise local track")
    add_circle(4)

    plt.tight_layout()
    plt.rcParams['pdf.use14corefonts'] = True

    plt.savefig(img_path + f"MapExtractionPipeline_{i}.svg", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(img_path + f"MapExtractionPipeline_{i}.pdf", bbox_inches='tight', pad_inches=0.01)
    plt.savefig(img_path + f"MapExtractionPipeline_{i}.jpeg", bbox_inches='tight', pad_inches=0.01, dpi=300)



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


map_name = "aut"
# n = 197
n = 496
# n = 501
make_pipeline("LocalMapPP", "c1", n, map_name)
# make_lidar_scan_img("LocalCenter_1", 91, map_name)
# make_boundary_img("LocalCenter_1", 91, map_name)

# plt.show()




