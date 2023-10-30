import numpy as np
from matplotlib import pyplot as plt
import csv, yaml
from PIL import Image
from matplotlib.collections import LineCollection

class MapData:
    def __init__(self, map_name):
        self.path = "maps/"
        # self.path = "map_data/"
        self.map_name = map_name

        self.xs, self.ys = None, None
        self.t_ss, self.t_xs, self.t_ys, self.t_ths, self.t_ks, self.t_vs, self.t_accs = None, None, None, None, None, None, None

        self.N = 0
        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.load_map_img()
        self.load_centerline()
        try:
            self.load_raceline()
        except: pass

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def load_centerline(self):
        xs, ys = [], []
        # with open(self.path + self.map_name + "_std.csv", 'r') as file:
        with open(self.path + self.map_name + "_centerline.csv", 'r') as file:
            csvFile = csv.reader(file)

            for i, lines in enumerate(csvFile):
                if i ==0:
                    continue
                xs.append(float(lines[0]))
                ys.append(float(lines[1]))

        self.xs = np.array(xs)
        self.ys = np.array(ys)

        self.N = len(xs)

    def load_raceline(self):
        filename = f"f1tenth_sim/racing_methods/planning/pp_traj_following/" + self.map_name + "_raceline.csv"
        racetrack = np.loadtxt(filename, delimiter=',', skiprows=1)
        self.t_xs = racetrack[:, 1]
        self.t_ys = racetrack[:, 2]

    def xy2rc(self, xs, ys):
        xs = (xs - self.map_origin[0]) / self.map_resolution
        ys = (ys - self.map_origin[1]) /self.map_resolution
        return xs, ys

    def pts2rc(self, pts):
        return self.xy2rc(pts[:,0], pts[:,1])
    
    def plot_centre_line(self):
        xs, ys = self.xy2rc(self.xs, self.ys)
        plt.plot(xs, ys, '--', color='black', linewidth=1)

    def plot_raceline(self):
        xs, ys = self.xy2rc(self.t_xs, self.t_ys)

        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(self.t_vs)
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

    def plot_map_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

    def plot_map_img_T(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.transpose(self.map_img)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def get_formatted_img(self):
        self.map_img[self.map_img == 1] = 180
        self.map_img[self.map_img == 0 ] = 230
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0

        return self.map_img

    def plot_map_img_light_T(self):
        self.map_img[self.map_img == 1] = 220
        self.map_img[self.map_img == 0 ] = 255
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        map_img = np.transpose(self.map_img)
        plt.imshow(map_img, origin='lower', cmap='gray')

    def plot_map_img_light(self):
        self.map_img[self.map_img == 1] = 220
        self.map_img[self.map_img == 0 ] = 255
        self.map_img[0, 1] = 255
        self.map_img[0, 0] = 0
        plt.imshow(self.map_img, origin='lower', cmap='gray')

    def plot_map_trajectory_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.t_vs)
        
        plt.xlabel("Track point")
        plt.ylabel("Speed (m/s)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("map_data/" + self.map_name + "_speeds.svg")

    def plot_map_trajectory_data(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.t_ks)
        
        plt.xlabel("Track point")
        plt.ylabel("Speed (m/s)")
        plt.grid()
        plt.tight_layout()
        plt.savefig("map_data/" + self.map_name + "_curvatures.svg")

    def plot_map_data(self):
        self.plot_map_img()

        self.plot_centre_line()
        
        self.plot_raceline()

        plt.savefig("map_data/" + self.map_name + "_map.svg")
        # plt.show()

        self.plot_map_trajectory_data()



def main():
    map_name = "esp"

    map_data = MapData(map_name)
    map_data.plot_map_data()

if __name__ == '__main__':

    main()