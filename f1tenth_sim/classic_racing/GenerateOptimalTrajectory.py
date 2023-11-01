import numpy as np 
import matplotlib.pyplot as plt
import csv
import trajectory_planning_helpers as tph
from matplotlib.collections import LineCollection
np.printoptions(precision=3, suppress=True)

MAX_KAPPA = 0.99
VEHICLE_WIDTH = 1.1
RACELINE_STEP = 0.2
MU = 0.75
V_MAX = 8
VEHICLE_MASS = 3.4
ax_max_machine = np.array([[0, 8.5],[8, 8.5]])
ggv = np.array([[0, 8.5, 8.5], [8, 8.5, 8.5]])


class OptimiseMap:
    def __init__(self, map_name) -> None:
        self.map_name = map_name
        self.track = np.loadtxt('maps/' + self.map_name + "_centerline.csv", delimiter=',')

        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track[:, :2], self.el_lengths, False)
        self.nvecs = tph.calc_normal_vectors.calc_normal_vectors(self.psi)

        crossing = tph.check_normals_crossing.check_normals_crossing(self.track, self.nvecs)
        if crossing: print(f"Major problem: nvecs are crossing. Result will be incorrect. Fix the center line file.")

        self.min_curve_path = None
        self.vs = None
        self.s_raceline = None
        self.psi_r = None
        self.kappa_r = None

        self.generate_minimum_curvature_path()
        self.plot_minimum_curvature_path()

        self.generate_velocity_profile()
        self.plot_raceline_trajectory()


    def generate_minimum_curvature_path(self):
        coeffs_x, coeffs_y, A, normvec_normalized = tph.calc_splines.calc_splines(self.track[:, 0:2], self.el_lengths, psi_s=self.psi[0], psi_e=self.psi[-1])

        alpha, error = tph.opt_min_curv.opt_min_curv(self.track, self.nvecs, A, MAX_KAPPA, VEHICLE_WIDTH, print_debug=True, closed=False, fix_s=True, psi_s=self.psi[0], psi_e=self.psi[-1], fix_e=True)

        self.min_curve_path, A_raceline, coeffs_x_raceline, coeffs_y_raceline, spline_inds_raceline_interp, t_values_raceline_interp, self.s_raceline, spline_lengths_raceline, el_lengths_raceline_interp_cl = tph.create_raceline.create_raceline(self.track[:, 0:2], self.nvecs, alpha, RACELINE_STEP) 
        self.psi_r, self.kappa_r = tph.calc_head_curv_num.calc_head_curv_num(self.min_curve_path, el_lengths_raceline_interp_cl, True)

    def generate_velocity_profile(self):
        mu = MU * np.ones(len(self.kappa_r))
        el_lengths = np.linalg.norm(np.diff(self.min_curve_path, axis=0), axis=1)

        self.vs = tph.calc_vel_profile.calc_vel_profile(ax_max_machine, self.kappa_r, el_lengths, False, 0, VEHICLE_MASS, ggv=ggv, mu=mu, v_max=V_MAX, v_start=V_MAX)

        ts = tph.calc_t_profile.calc_t_profile(self.vs, el_lengths, 0)
        print(f"Planned Lap Time: {ts[-1]}")
        acc = tph.calc_ax_profile.calc_ax_profile(self.vs, el_lengths, True)

        raceline = np.concatenate([self.s_raceline[:, None], self.min_curve_path, self.psi_r[:, None], self.kappa_r[:, None], self.vs[:, None], acc[:, None]], axis=1)
        np.savetxt("racelines/"+ self.map_name+ '_raceline.csv', raceline, delimiter=',')

    def plot_minimum_curvature_path(self):
        plt.figure(3)
        plt.clf()
        plt.plot(self.track[:, 0], self.track[:, 1], '-', linewidth=2, color='blue', label="Track")
        plt.plot(self.min_curve_path[:, 0], self.min_curve_path[:, 1], '-', linewidth=2, color='red', label="Raceline")

        l_line = self.track[:, 0:2] + self.nvecs * self.track[:, 2][:, None]
        r_line = self.track[:, 0:2] - self.nvecs * self.track[:, 3][:, None]
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green', label="Boundaries")
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')
        
        plt.legend()
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"racelines/minimum_curvature_path_{self.map_name}.svg", pad_inches=0)

    def plot_raceline_trajectory(self):
        plt.figure(1)
        plt.clf()

        l_line = self.track[:, 0:2] + self.nvecs * self.track[:, 2][:, None]
        r_line = self.track[:, 0:2] - self.nvecs * self.track[:, 3][:, None]
        plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1, color='green', label="Boundaries")
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1, color='green')

        vs = self.vs
        points = self.min_curve_path.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(2, 8)
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(vs)
        lc.set_linewidth(5)
        line = plt.gca().add_collection(lc)
        plt.colorbar(line)
        plt.xlim(self.min_curve_path[:, 0].min()-1, self.min_curve_path[:, 0].max()+1)
        plt.ylim(self.min_curve_path[:, 1].min()-1, self.min_curve_path[:, 1].max()+1)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"racelines/raceline_speeds_{self.map_name}.svg", pad_inches=0)


def run_profiling(function, name):
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    function()
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    with open(f"Data/profile_{name}.txt", "w") as f:
        ps.print_stats()
        f.write(s.getvalue())


def profile_aut():
    OptimiseMap('aut', 30)


if __name__ == "__main__":
    map_list = ['aut', 'esp', 'gbr', 'mco']
    for map_name in map_list: OptimiseMap(map_name)

    # run_profiling(profile_aut, 'GenerateAUT')