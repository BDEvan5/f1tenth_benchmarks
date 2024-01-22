import numpy as np
from matplotlib import pyplot as plt
import csv, yaml
from PIL import Image
from matplotlib.collections import LineCollection

class MapData:
    def __init__(self, map_name):
        self.map_name = map_name

        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        try:
            self.path = "maps/"
            self.load_map_img()
        except:
            self.path = "../maps/"
            self.load_map_img()

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        if len(self.map_img.shape) > 2:
            self.map_img = self.map_img[:, :, 0]

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
    def xy2rc(self, xs, ys):
        xs = (xs - self.map_origin[0]) / self.map_resolution
        ys = (ys - self.map_origin[1]) /self.map_resolution
        return xs, ys

    def pts2rc(self, pts):
        return self.xy2rc(pts[:,0], pts[:,1])
    
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

    

def main():
    map_name = "esp"

    map_data = MapData(map_name)
    map_data.plot_map_img()

    plt.show()

if __name__ == '__main__':

    main()