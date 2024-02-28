import yaml 
import numpy as np 
import matplotlib.pyplot as plt
import csv
from scipy import ndimage 
from scipy import interpolate
import cv2 as cv
import trajectory_planning_helpers as tph
import os
from PIL import Image 


save_path = f"Data/CentreLineExtraction/"

if not os.path.exists(save_path):
    os.makedirs(save_path)


def extract_centre_line(map_name):
    file_name = 'maps/' + map_name + '.yaml'
    with open(file_name) as file:
        documents = yaml.full_load(file)

    yaml_file = dict(documents.items())
    resolution = yaml_file['resolution']
    origin = yaml_file['origin']

    image = cv.imread('maps/' + yaml_file['image'])
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    c, h = cv.findContours(gray ,cv.RETR_TREE , cv.CHAIN_APPROX_NONE)

    contours = []
    for raw_contour in c:
        contour = raw_contour[:, 0, :]
        contour[:,] = np.ones_like(contour) * np.array([0, image.shape[0]]) - contour
        contour = np.abs(contour)
        contour = contour * resolution + origin[:2]
        contours.append(contour)

    contour1 = TrackContour(contours[0])
    contour2 = TrackContour(contours[1])

    centre_line = caluclate_centre_line(contour1.path, contour2.path)

    flipped_map_img = np.array(Image.open('maps/' + yaml_file['image']).transpose(Image.FLIP_TOP_BOTTOM))
    dt = ndimage.distance_transform_edt(flipped_map_img)
    np.array(dt *resolution)
    widths = np.zeros_like(centre_line)
    for i in range(centre_line.shape[0]):
        widths[i, :] = get_dt_value(centre_line[i], origin, resolution, dt)

    track =  np.concatenate([centre_line, widths], axis=-1)     
    map_c_name = f"maps/{map_name}_centerline.csv"
    with open(map_c_name, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(track)

    print(f"Centerline saved in: {map_c_name}")
    
    plt.figure(1) 
    plt.clf()
    plt.plot(contours[0][:, 0], contours[0][:, 1], 'r')
    plt.plot(contours[1][:, 0], contours[1][:, 1], 'b')    
    
    plt.plot(centre_line[:, 0], centre_line[:, 1], 'k')
    plt.plot(centre_line[0, 0], centre_line[0, 1], 'ro', markersize=10)
    plt.plot(centre_line[20, 0], centre_line[20, 1], 'go', markersize=10)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path + f"centre_line_extraction_{map_name}.svg")


def get_dt_value(pt_xy, origin, resolution, dt):
    c = int((pt_xy[0] - origin[0]) / resolution)
    r = int((pt_xy[1] - origin[1]) / resolution)

    if c >= dt.shape[1]:
        c = dt.shape[1] - 1
    if r >= dt.shape[0]:
        r = dt.shape[0] - 1

    distance = dt[r, c] * resolution

    return distance


def caluclate_centre_line(b1, b2):
    centre_line = []
    for i in range(len(b1)):
        dists = np.linalg.norm(b1[i, :] - b2, axis=1)
        min_dist_segment = np.argmin(dists)
        pt = (b1[i, :] + b2[min_dist_segment, :]) / 2
        centre_line.append(pt)


    centre_line = np.array(centre_line)[:-1] # trim the last point
    CENTRE_SEP_DISTANCE = 0.2
    line_length = np.sum(np.linalg.norm(np.diff(centre_line, axis=0), axis=1))
    n_pts = int(line_length / CENTRE_SEP_DISTANCE)

    tck = interpolate.splprep([centre_line[:, 0], centre_line[:, 1]], k=3, s=0, per=True)[0]
    centre_line = np.array(interpolate.splev(np.linspace(0, 1, n_pts), tck)).T[:-1] # remove duplicate point
    
    start_ind = np.argmin(np.linalg.norm(centre_line, axis=1))
    centre_line = np.roll(centre_line, -start_ind, axis=0)

    if centre_line[5, 0] < centre_line[0, 0]:
        centre_line = centre_line[::-1, :] # reverse it so that x is increasing

    return centre_line

class TrackContour:
    def __init__(self, points, n_pts=2000, smoothing_s=0.5):
        if points[5, 0] < points[0, 0]:
            points = points[::-1, :] # reverse it so that x is increasing

        points = interpolate_track(points, n_pts, smoothing_s)
        self.path = interpolate_track(points, n_pts, 0)

        self.el_lengths = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.s_path = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.path, self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

    def get_projection(self, width,  sign=1, n_pts=500):
        projection =  self.path + self.nvecs * width * sign
        projection = interpolate_track(projection, n_pts, 0)

        return projection


def interpolate_track(points, n_points=None, s=0):
    tck = interpolate.splprep([points[:, 0], points[:, 1]], k=3, s=s)[0]
    if n_points is None: n_points = len(points)
    track = np.array(interpolate.splev(np.linspace(0, 1, n_points), tck)).T

    return track



if __name__ == '__main__':
    extract_centre_line('aut')
    extract_centre_line('esp')
    extract_centre_line('gbr')
    extract_centre_line('mco')

