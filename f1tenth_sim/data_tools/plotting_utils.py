import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import glob

red_orange = "#ff3f34"
jade_dust = "#00d8d6"
minty_green = "#0be881"
chrome_yellow = "#ffa801"
disco_ball = "#0fbcf9"
free_speech = "#3c40c6"
high_pink = "#ef5777"
vibe_yellow = "#ffd32a"
periwinkle = "#575fcf"
sunset_orange = "#ff5e57"
sweedish_green = "#05c46b"
london_square = "#808e9b"
fresh_t = "#34e7e4"
nartjie = "#ffc048"
sizzling_red = "#f53b57"
megaman = "#4bcffa"
yerieal_yellow = "#ffdd59"
fresh_turquoise = "#34e7e4"
lighter_purple = "#a55eea"
good_grey = "#485460"

def std_img_saving(name):

    plt.rcParams['pdf.use14corefonts'] = True

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)


def ensure_path_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def true_moving_average(data, period):
    if len(data) < period:
        return np.zeros_like(data)
    ret = np.convolve(data, np.ones(period), 'same') / period
    for i in range(period): # start
        t = np.convolve(data, np.ones(i+2), 'valid') / (i+2)
        ret[i] = t[0]
    for i in range(period):
        length = int(round((i + period)/2))
        t = np.convolve(data, np.ones(length), 'valid') / length
        ret[-i-1] = t[-1]
    return ret



def convert_to_min_max_avg(step_list, progress_list, xs):
    """Returns the 3 lines 
        - Minimum line
        - maximum line 
        - average line 
    """ 
    n = len(step_list)

    ys = np.zeros((n, len(xs)))
    for i in range(n):
        ys[i] = np.interp(xs, step_list[i], progress_list[i])

    min_line = np.min(ys, axis=0)
    max_line = np.max(ys, axis=0)
    avg_line = np.mean(ys, axis=0)

    return min_line, max_line, avg_line

def smooth_line(steps, progresses, length_xs=300):
    xs = np.linspace(steps[0], steps[-1], length_xs)
    smooth_line = np.interp(xs, steps, progresses)

    return xs, smooth_line



