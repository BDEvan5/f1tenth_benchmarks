import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca
from matplotlib.collections import LineCollection

from f1tenth_benchmarks.localmap_racing.LocalMapGenerator import LocalMapGenerator
from f1tenth_benchmarks.localmap_racing.LocalReference import LocalReference
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner, ensure_path_exists
from f1tenth_benchmarks.localmap_racing.LocalMap import *


NX = 4
NU = 3


class LocalMPCC(BasePlanner):
    def __init__(self, test_id, save_data=False, surpress_output=False):
        super().__init__("LocalMPCC", test_id)
        self.surpress_output = surpress_output
        self.rp = None
        self.dt = self.planner_params.dt
        self.N = self.planner_params.N
        self.g, self.obj = None, None
        self.mpcc_data_path = self.data_root_path + f"MPCCData_{test_id}/"
        ensure_path_exists(self.mpcc_data_path)
        self.local_map_generator = LocalMapGenerator(self.data_root_path, test_id, save_data)

        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time
        self.f_max = self.vehicle_params.gravity * self.vehicle_params.vehicle_mass * self.planner_params.friction_mu

        self.init_optimisation()
        self.init_constraints()
    
    def init_optimisation(self):
        states = ca.MX.sym('states', NX) # [x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, v, p]

        rhs = ca.vertcat(controls[1] * ca.cos(states[2]), 
                         controls[1] * ca.sin(states[2]), 
                         (controls[1] / self.vehicle_params.wheelbase) * ca.tan(controls[0]), 
                         controls[2])  # dynamic equations of the states

        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX + 2 * self.N + 1) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        p = self.planner_params
        lbx = [[p.position_min], [p.position_min], [-p.heading_max], [0]] * (self.N + 1) + [[-p.delta_max], [p.min_speed], [p.local_path_speed_min]] * self.N
        self.lbx = np.array(lbx)
        ubx = [[p.position_max], [p.position_max], [p.heading_max], [p.local_path_max_length]] * (self.N + 1) + [[p.delta_max], [p.max_speed], [p.local_path_speed_max]] * self.N
        self.ubx = np.array(ubx)

    def init_objective(self):
        self.obj = 0  # Objective function

        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.rp.angle_lut_t(st_next[3])
            delta_x = st_next[0] - self.rp.center_lut_x(st_next[3])
            delta_y = st_next[1] - self.rp.center_lut_y(st_next[3])
            
            contouring_error = ca.sin(t_angle) * delta_x - ca.cos(t_angle) * delta_y 
            lag_error = -ca.cos(t_angle) * delta_x - ca.sin(t_angle) * delta_y 

            self.obj = self.obj + contouring_error **2 * self.planner_params.weight_contour  
            self.obj = self.obj + lag_error **2 * self.planner_params.weight_lag
            self.obj = self.obj - self.U[2, k] * self.planner_params.weight_progress # maximise progress speed
            self.obj = self.obj + (self.U[0, k]) ** 2 * self.planner_params.weight_steer  # minimize steering input

    def init_bounds(self):
        self.g = []  # constraints vector
        self.g = ca.vertcat(self.g, self.X[:, 0] - self.P[:NX])  # initial condition constraints
        for k in range(self.N):
            st = self.X[:, k]
            st_next = self.X[:, k + 1]
            con = self.U[:, k]

            st_next_euler = st + (self.dt * self.f(st, con)) # Vehicle dynamics constraints
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  

            self.g = ca.vertcat(self.g, self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1])  # path boundary constraints
            
            force_lateral = con[1] **2 / self.vehicle_params.wheelbase * ca.tan(ca.fabs(con[0])) *  self.vehicle_params.vehicle_mass
            self.g = ca.vertcat(self.g, force_lateral) # frictional constraint

            if k == 0: 
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.P[-1])) # ensure initial speed matches current speed
            else:
                self.g = ca.vertcat(self.g, ca.fabs(con[1] - self.U[1, k - 1]))  # limit decceleration

    def init_solver(self):
        variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))
        nlp_prob = {'f': self.obj, 'x': variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 1000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def plan(self, obs):
        self.step_counter += 1
        local_track = self.local_map_generator.generate_line_local_map(obs['scan'])
        if len(local_track) <= 2:
            return np.array([0, 1])

        self.local_map = LocalMap(local_track)
        self.rp = LocalReference(self.local_map)

        x0 = np.zeros(3)
        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))
        vehicle_speed = obs["vehicle_speed"]

        self.init_objective()
        self.init_bounds()
        self.init_solver()

        p = self.generate_constraints_and_parameters(x0, vehicle_speed)
        states, controls, solved_status = self.solve(p)
        if not solved_status:
            self.warm_start = True
            p = self.generate_constraints_and_parameters(x0, vehicle_speed)
            states, controls, solved_status = self.solve(p)
            if not solved_status:
                if self.surpress_output:
                    print(f"Solve failed: ReWarm Start: New outcome: {solved_status}")
                    print(f"S:{x0[3]:2f} --> Action: {controls[0, 0:2]}")
                return np.array([0, 1])
            
        action = controls[0, 0:2]
        # print(f"{self.step_counter} -- S: {100*obs['progress']:.2f} --> Pose: {obs['pose']} --> Action: {action}")
        np.save(self.mpcc_data_path + f"States_{self.step_counter}.npy", states)
        np.save(self.mpcc_data_path + f"Controls_{self.step_counter}.npy", controls)
        

        return action 

    def generate_constraints_and_parameters(self, x0_in, x0_speed):
        self.lbg, self.ubg = np.zeros((self.g.shape[0], 1)), np.zeros((self.g.shape[0], 1))
        if self.warm_start:
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

            right_bound = -delta_x_path * right_point[0] - delta_y_path * right_point[1]
            left_bound = -delta_x_path * left_point[0] - delta_y_path * left_point[1]

            self.lbg[NX - 3 + (NX + 3) * (k + 1), 0] = min(left_bound, right_bound)
            self.ubg[NX - 3 + (NX + 3) * (k + 1), 0] = max(left_bound, right_bound)
            self.lbg[NX - 2 + (NX + 3) * (k + 1), 0] = - self.f_max
            self.ubg[NX - 2 + (NX + 3) * (k + 1) , 0] = self.f_max
            #Adjust indicies.
            self.lbg[NX -1 + (NX + 3) * (k + 1), 0] = - self.planner_params.max_decceleration * self.dt
            self.ubg[NX -1 + (NX + 3) * (k + 1) , 0] = ca.inf # do not limit speeding up


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
            s_next = self.X0[k - 1, 3] + self.planner_params.p_initial * self.dt
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
        self.u0[:, 1] = self.planner_params.p_initial
        self.u0[:, 2] = self.planner_params.p_initial

        self.warm_start = False

    def plot_vehicle_controls(self, p, controls):
        fig, axs = plt.subplots(5, 1, num=3, clear=True, figsize=(8, 15))
        axs[0].plot(controls[:, 0], '-o', color='red')
        axs[0].set_ylabel('Steering Angle')
        axs[0].set_ylim([-0.5, 0.5])
        axs[0].grid(True)

        axs[1].plot(controls[:, 1], '-o', color='red')
        axs[1].set_ylabel('Speed')
        axs[1].set_ylim([0, 9])
        axs[1].grid(True)

        forces = [controls[k, 1] **2 / p.wheelbase * ca.tan(ca.fabs(controls[k, 0])) *  p.vehicle_mass for k in range(self.N)]
        axs[2].plot(forces, '-o', color='red')
        axs[2].set_ylabel('Lateral Force')
        axs[2].set_ylim([0, 40])
        axs[2].grid(True)

        axs[3].plot(controls[:, 2], '-o', color='red')
        axs[3].set_ylabel('Centerline \nSpeed')
        axs[3].set_ylim([0, 9])
        axs[3].grid(True)

        dv = np.diff(controls[:, 1])
        axs[4].plot(dv, '-o', color='red')
        axs[4].set_ylabel('Acceleration')
        axs[4].grid(True)

        plt.pause(0.001)

    def plot_vehicle_position(self, x0, states, controls):
        c_pts = [[self.rp.center_lut_x(states[k, 3]).full()[0, 0], self.rp.center_lut_y(states[k, 3]).full()[0, 0]] for k in range(self.N + 1)]

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

        for i in range(self.N + 1):
            xs = [states[i, 0], c_pts[i][0]]
            ys = [states[i, 1], c_pts[i][1]]
            plt.plot(xs, ys, '--', color='orange')

        size = 12
        plt.xlim([x0[0] - size, x0[0] + size])
        plt.ylim([x0[1] - size, x0[1] + size])
        plt.pause(0.001)# return the first control action

def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi