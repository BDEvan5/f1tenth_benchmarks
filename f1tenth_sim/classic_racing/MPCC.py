import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca
import os
from argparse import Namespace
from matplotlib.collections import LineCollection

from f1tenth_sim.classic_racing.planner_utils import RaceTrack
from f1tenth_sim.classic_racing.ReferencePath import ReferencePath

VERBOSE = False
VERBOSE = True

WAIT_FOR_USER = False
# WAIT_FOR_USER = True

L = 0.33
GRAVITY = 9.81 
MASS = 3.71
MU = 0.8
F_MAX = 1 * GRAVITY * MASS * MU
MAX_ACCELERATION = 8


WEIGHT_PROGRESS = 0
WEIGHT_LAG = 20
WEIGHT_CONTOUR = 20
WEIGHT_STEER = 500
WEIGHT_STEER_CHANGE = 1000
WEIGHT_SPEED_CHANGE = 10

np.printoptions(precision=2, suppress=True)


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi

NX = 4
NU = 3

p = {
    "position_min": -100,
    "position_max": 100,
    "heading_max": 10,
    "delta_max": 0.4,
    "local_path_max_length": 200,
    "max_speed": 8,
    "min_speed": 2,
    "local_path_speed_min": 2,
    "local_path_speed_max": 10,
    "max_v_dot": 0.04*8
}
p = Namespace(**p)

class MPCC:
    def __init__(self):
        self.name = "MPCC"
        self.rp = None
        self.rt = None
        self.dt = 0.1
        self.N = 15 # number of steps to predict
        self.p_initial = 5
        self.g, self.obj = None, None

        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time

        self.init_optimisation()
        self.init_constraints()
    
    def set_map(self, map_name):
        self.rp = ReferencePath(map_name)
        self.rt = RaceTrack(map_name)

        self.init_objective()
        self.init_bounds()
        self.init_solver()

    def init_optimisation(self):
        states = ca.MX.sym('states', NX) # [x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, v, p]

        rhs = ca.vertcat(states[3] * ca.cos(states[2]), 
                         states[3] * ca.sin(states[2]), 
                         (states[3] / L) * ca.tan(controls[0]), 
                         controls[1])  # dynamic equations of the states

        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX + 2 * self.N + 1) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        lbx = [[p.position_min], [p.position_min], [-p.heading_max], [0]] * (self.N + 1) + [[-p.delta_max], [p.min_speed], [p.local_path_speed_min]] * self.N
        self.lbx = np.array(lbx)
        ubx = [[p.position_max], [p.position_max], [p.heading_max], [p.local_path_max_length]] * (self.N + 1) + [[p.delta_max], [p.max_speed], [p.local_path_speed_max]] * self.N
        self.ubx = np.array(ubx)

    def init_objective(self):
        self.obj = 0  # Objective function

        for k in range(self.N):
            st_next = self.X[:, k + 1]
            con = self.U[:, k]
            
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            #Contouring error
            e_c = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            #Lag error
            e_l = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + e_c **2 * WEIGHT_CONTOUR  
            self.obj = self.obj + e_l **2 * WEIGHT_LAG
            self.obj = self.obj - con[2] * WEIGHT_PROGRESS 
            self.obj = self.obj + (con[0]) ** 2 * WEIGHT_STEER  # minimize the use of steering input

            if k != 0: 
                self.obj = self.obj + (con[0] - self.U[0, k - 1]) ** 2 * WEIGHT_STEER_CHANGE  # minimize the change of steering input
                self.obj = self.obj + (con[1] - self.U[1, k - 1]) ** 2 * WEIGHT_SPEED_CHANGE  # 
            
        
    def init_bounds(self):
        self.g = []  # constraints vector
        self.g = ca.vertcat(self.g, self.X[:, 0] - self.P[:NX])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]

            # Vehicle dynamics constraints
            k1 = self.f(st, con)
            st_next_euler = st + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # compute constraints

            # path boundary constraints
            self.g = ca.vertcat(self.g, self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  --represents half space planes

            # frictional constraint
            force_lateral = con[1] **2 / L * ca.tan(ca.fabs(con[0])) *  MASS
            self.g = ca.vertcat(self.g, force_lateral)

            if k == 0: 
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.P[-1]))
            else:
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.U[1, k - 1]))  # prevent velocity change from being too abrupt


    def init_solver(self):
        opts = {}
        opts["ipopt"] = {}
        opts["ipopt"]["max_iter"] = 1000
        opts["ipopt"]["acceptable_tol"] = 1e-8
        opts["ipopt"]["acceptable_obj_change_tol"] = 1e-6
        opts["ipopt"]["fixed_variable_treatment"] = "make_parameter"
        opts["ipopt"]["print_level"] = 0
        opts["print_time"] = 0
        
        OPT_variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': OPT_variables, 'g': self.g, 'p': self.P}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def prepare_input(self, obs):
        x0 = obs["pose"]
        x0[2] = normalise_psi(x0[2]) 

        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))
        x0[3] = self.filter_estimate(x0[3])
        vehicle_speed = obs["vehicle_speed"]

        return x0, vehicle_speed

    def plan(self, obs):
        x0, vehicle_speed = self.prepare_input(obs)

        p = self.generate_constraints_and_parameters(x0, vehicle_speed)
        states, controls, solved_status = self.solve(p)
        if not solved_status:
            self.warm_start = True
            p = self.generate_constraints_and_parameters(x0, vehicle_speed)
            states, controls, solved_status = self.solve(p)
            if VERBOSE:
                print(f"Solve failed: ReWarm Start: New outcome: {solved_status}")

        s = states[:, 3]
        s = [s[k] if s[k] < self.rp.track_length else s[k] - self.rp.track_length for k in range(self.N+1)]
        c_pts = [[self.rp.center_lut_x(states[k, 3]).full()[0, 0], self.rp.center_lut_y(states[k, 3]).full()[0, 0]] for k in range(self.N + 1)]

        first_control = controls[0, :]
        action = first_control[0:2]
        if VERBOSE:
            print(f"S:{x0[3]:2f} --> Action: {action}")
            if not solved_status:
                plt.show()

        if VERBOSE:
            self.rp.plot_path()
            plt.figure(2)
            points = states[:, 0:2].reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 8)
            lc = LineCollection(segments, cmap='jet', norm=norm)
            lc.set_array(controls[:, 1])
            lc.set_linewidth(4)
            line = plt.gca().add_collection(lc)
            plt.colorbar(line, fraction=0.046, pad=0.04, shrink=0.99)

            # plt.plot(states[:, 0], states[:, 1], '-o', color='red')
            for i in range(self.N + 1):
                xs = [states[i, 0], c_pts[i][0]]
                ys = [states[i, 1], c_pts[i][1]]
                plt.plot(xs, ys, '--', color='orange')

            size = 12
            plt.xlim([x0[0] - size, x0[0] + size])
            plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)

            # if not solved_status:
            #     plt.show()

        if VERBOSE:
            fig, axs = plt.subplots(5, 1, num=3, clear=True, figsize=(8, 15))
            axs[0].plot(controls[:, 0], '-o', color='red')
            axs[0].set_ylabel('Steering Angle')
            axs[0].set_ylim([-0.5, 0.5])
            axs[0].grid(True)
            # axs[0].set_title(f"Speed: {fs[3]:2f}")

            axs[1].plot(controls[:, 1], '-o', color='red')
            # vs = self.rt.get_interpolated_vs(states[:, 3])
            # axs[1].plot(vs, '-o', color='blue')
            axs[1].set_ylabel('Speed')
            axs[1].set_ylim([0, 9])
            axs[1].grid(True)

            forces = [controls[k, 1] **2 / L * ca.tan(ca.fabs(controls[k, 0])) *  MASS for k in range(self.N)]
            axs[2].plot(forces, '-o', color='red')
            axs[2].set_ylabel('Lateral Force')
            axs[2].set_ylim([0, 40])
            axs[2].grid(True)


            axs[3].plot(controls[:, 2], '-o', color='red')
            axs[3].set_ylabel('Centerline \nSpeed')
            axs[3].set_ylim([0, 9])
            axs[3].grid(True)

            dv = np.diff(controls[:, 1])
            # dv = np.insert(dv, 0, controls[0, 1]- fs[3])
            # dv = np.insert(0, fs[3] - controls[0, 1])

            axs[4].plot(dv, '-o', color='red')
            axs[4].set_ylabel('Acceleration')
            # axs[4].set_ylim([-2, 2])
            axs[4].grid(True)

            plt.pause(0.001)
            # plt.pause(0.1)

            # if action[1] < 4:
            #     plt.show()

        if VERBOSE and False:
            plt.figure(4)
            plt.clf()
            plt.plot(self.rp.center_lut_x(self.rp.s_track), self.rp.center_lut_y(self.rp.s_track), label="center", color='green', alpha=0.7)
            plt.plot(self.rp.left_lut_x(self.rp.s_track), self.rp.left_lut_y(self.rp.s_track), label="left", color='green', alpha=0.7)
            plt.plot(self.rp.right_lut_x(self.rp.s_track), self.rp.right_lut_y(self.rp.s_track), label="right", color='green', alpha=0.7)

            plt.plot(states[:, 0], states[:, 1], '-o', color='red')
            # plt.plot(c_pts[:, 0], c_pts[:, 1], '-o', color='blue')
            # plt.plot(x0[0], x0[1], 'x', color='green')
            plt.plot(self.rt.wpts[:, 0], self.rt.wpts[:, 1], color='blue')

            size = 12
            plt.xlim([x0[0] - size, x0[0] + size])
            plt.ylim([x0[1] - size, x0[1] + size])
            plt.pause(0.001)


        if VERBOSE and WAIT_FOR_USER:
            plt.show()


        return action # return the first control action

    def generate_constraints_and_parameters(self, x0_in, x0_speed):
        self.lbg, self.ubg = np.zeros((self.g.shape[0], 1)), np.zeros((self.g.shape[0], 1))
        if self.warm_start:
            if VERBOSE:
                print(f"Warm starting with condition: {x0_in}")
            self.construct_warm_start_soln(x0_in) 

        pp = np.zeros(NX + 2 * self.N + 1)
        pp[:NX] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s = self.X0[k, 3]
            if s > self.rp.track_length:
                s = s - self.rp.track_length
            right_point = [self.rp.right_lut_x(s).full()[0, 0], self.rp.right_lut_y(s).full()[0, 0]]
            left_point = [self.rp.left_lut_x(s).full()[0, 0], self.rp.left_lut_y(s).full()[0, 0]]

            delta_x_path = right_point[0] - left_point[0]
            delta_y_path = right_point[1] - left_point[1]
            pp[NX + 2 * k:NX + 2 * k + 2] = [-delta_x_path, delta_y_path]

            up_bound = max(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                           -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            low_bound = min(-delta_x_path * right_point[0] - delta_y_path * right_point[1],
                            -delta_x_path * left_point[0] - delta_y_path * left_point[1])
            self.lbg[NX - 3 + (NX + 3) * (k + 1), 0] = low_bound # check this, there could be an error
            self.ubg[NX - 3 + (NX + 3) * (k + 1), 0] = up_bound
            self.lbg[NX - 2 + (NX + 3) * (k + 1), 0] = - F_MAX
            self.ubg[NX - 2 + (NX + 3) * (k + 1) , 0] = F_MAX
            #Adjust indicies.
            self.lbg[NX -1 + (NX + 3) * (k + 1), 0] = - p.max_v_dot
            self.ubg[NX -1 + (NX + 3) * (k + 1) , 0] = ca.inf # do not limit speeding up
            # self.ubg[NX -1 + (NX + 3) * (k + 1) , 0] = self.max_v_dot


        # the optimizer cannot control the init state.
        self.lbg[NX *2, 0] = - ca.inf
        self.ubg[NX *2, 0] = ca.inf

        pp[-1] = max(x0_speed, 1) # prevent constraint violation

        return pp

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        # Get state and control solution
        self.X0 = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T  # get soln trajectory
        u = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T  # get controls solution

        trajectory = self.X0.full()  # size is (N+1,n_states)
        inputs = u.full()
        stats = self.solver.stats()
        solved_status = True
        if stats['return_status'] == 'Infeasible_Problem_Detected':
        # if stats['return_status'] != 'Solve_Succeeded':
            if VERBOSE:
                print(stats['return_status'])
            solved_status = False

        # Shift trajectory and control solution to initialize the next step
        self.X0 = ca.vertcat(self.X0[1:, :], self.X0[self.X0.size1() - 1, :])
        self.u0 = ca.vertcat(u[1:, :], u[u.size1() - 1, :])

        return trajectory, inputs, solved_status
        
    def filter_estimate(self, initial_arc_pos):
        if (self.X0[0, 3] >= self.rp.track_length) and (
                (initial_arc_pos >= self.rp.track_length) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.rp.track_length
        if initial_arc_pos >= self.rp.track_length:
            initial_arc_pos -= self.rp.track_length
        return initial_arc_pos

    def construct_warm_start_soln(self, initial_state):
        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.p_initial * self.dt
            if s_next > self.rp.track_length:
                s_next = s_next - self.rp.track_length

            psi_next = self.rp.angle_lut_t(s_next).full()[0, 0]
            x_next, y_next = self.rp.center_lut_x(s_next), self.rp.center_lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2

            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

        self.u0 = np.zeros((self.N, NU))
        self.u0[:, 1] = self.p_initial
        self.u0[:, 2] = self.p_initial

        self.warm_start = False

