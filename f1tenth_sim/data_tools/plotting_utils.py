import matplotlib.pyplot as plt
import numpy as np
import csv
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

def std_img_saving(name):

    plt.rcParams['pdf.use14corefonts'] = True

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.savefig(name + ".svg", bbox_inches='tight', pad_inches=0)



def load_time_data(folder, map_name=""):
    files = glob.glob(folder + f"Results_*{map_name}*.txt")
    files.sort()
    print(files)
    keys = ["time", "success", "progress"]
    mins, maxes, means = {}, {}, {}
    for key in keys:
        mins[key] = []
        maxes[key] = []
        means[key] = []
    
    for i in range(len(files)):
        with open(files[i], 'r') as file:
            lines = file.readlines()
            for j in range(len(keys)):
                mins[keys[j]].append(float(lines[3].split(",")[1+j]))
                maxes[keys[j]].append(float(lines[4].split(",")[1+j]))
                means[keys[j]].append(float(lines[1].split(",")[1+j]))

    return mins, maxes, means


def load_csv_data(path):
    """loads data from a csv training file

    Args:   
        path (file_path): path to the agent

    Returns:
        rewards: ndarray of rewards
        lengths: ndarray of episode lengths
        progresses: ndarray of track progresses
        laptimes: ndarray of laptimes
    """
    rewards, lengths, progresses, laptimes = [], [], [], []
    with open(f"{path}training_data_episodes.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if float(row[2]) > 0:
                rewards.append(float(row[1]))
                lengths.append(float(row[2]))
                progresses.append(float(row[3]))
                laptimes.append(float(row[4]))

    rewards = np.array(rewards)[:-1]
    lengths = np.array(lengths)[:-1]
    progresses = np.array(progresses)[:-1]
    laptimes = np.array(laptimes)[:-1]
    
    return rewards, lengths, progresses, laptimes

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



pp_light = ["#EC7063", "#5499C7", "#58D68D", "#F4D03F", "#AF7AC5", "#F5B041", "#EB984E"]    
pp = ["#CB4335", "#2874A6", "#229954", "#D4AC0D", "#884EA0", "#BA4A00", "#17A589"]
pp_dark = ["#943126", "#1A5276", "#1D8348", "#9A7D0A", "#633974", "#9C640C", "#7E5109"]
pp_darkest = ["#78281F", "#154360", "#186A3B", "#7D6608", "#512E5F", "#7E5109"]

light_blue = "#5DADE2"
dark_blue = "#154360"
light_red = "#EC7063"
dark_red = "#78281F"
light_green = "#58D68D"
dark_green = "#186A3B"

light_purple = "#AF7AC5"
light_yellow = "#F7DC6F"

plot_green = "#2ECC71"
plot_red = "#E74C3C"
plot_blue = "#3498DB"

google = ["#008744", "#0057e7", "#d62d20", "#ffa700"]
science_pallet = ['#0C5DA5', '#FF2C00', '#00B945', '#FF9500', '#845B97', '#474747', '#9e9e9e']
science_bright = ['EE7733', '0077BB', '33BBEE', 'EE3377', 'CC3311', '009988', 'BBBBBB']
science_bright = [f"#{c}" for c in science_bright]

science_high_vis = ["#0d49fb", "#fec32d", "#e6091c", "#26eb47", "#8936df", "#25d7fd"]
color_pallette = science_high_vis

def plot_error_bars(x_base, mins, maxes, dark_color, w):
    for i in range(len(x_base)):
        xs = [x_base[i], x_base[i]]
        ys = [mins[i], maxes[i]]
        plt.plot(xs, ys, color=dark_color, linewidth=2)
        xs = [x_base[i]-w, x_base[i]+w]
        y1 = [mins[i], mins[i]]
        y2 = [maxes[i], maxes[i]]
        plt.plot(xs, y1, color=dark_color, linewidth=2)
        plt.plot(xs, y2, color=dark_color, linewidth=2)


