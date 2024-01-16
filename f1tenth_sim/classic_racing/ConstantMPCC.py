import numpy as np 
import matplotlib.pyplot as plt
import casadi as ca

from f1tenth_sim.classic_racing.ReferencePath import ReferencePath
from f1tenth_sim.classic_racing.mpcc_utils import *
from f1tenth_sim.utils.BasePlanner import BasePlanner
from f1tenth_sim.utils.track_utils import CentreLine, TrackLine


NX = 4
NU = 2

class ConstantMPCC(BasePlanner):
    def __init__(self, test_id, save_data=False, planner_name="ConstantMPCC"):
        super().__init__(planner_name, test_id, params_name="GlobalMPCC")
        self.save_data = save_data
        if self.save_data:
            self.mpcc_data_path = self.data_root_path + f"MPCCData_{test_id}/"
            self.create_clean_path(self.mpcc_data_path)

        self.N = self.planner_params.N
        self.centre_line = None
        self.track_length = None
        self.centre_interpolant = None
        self.left_interpolant = None
        self.right_interpolant = None

        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time

        self.init_optimisation()
    
    def set_map(self, map_name):
        self.centre_line = CentreLine(map_name)
        self.track_length = self.centre_line.s_path[-1]
        self.centre_interpolant, self.left_interpolant, self.right_interpolant = init_track_interpolants(self.centre_line, self.planner_params.exclusion_width)

        self.init_constraints()
        self.init_bounds()
        self.init_objective()
        self.init_solver()
       
    def init_optimisation(self):
        states = ca.MX.sym('states', NX) #[x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, p]

        rhs = ca.vertcat(self.planner_params.vehicle_speed * ca.cos(states[2]), self.planner_params.vehicle_speed * ca.sin(states[2]), (self.planner_params.vehicle_speed / self.vehicle_params.wheelbase) * ca.tan(controls[0]), controls[1])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX + 2 * self.N) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        self.lbg = np.zeros((NX * (self.N + 1) + self.N, 1))
        self.ubg = np.zeros((NX * (self.N + 1) + self.N, 1))
        self.lbx = np.zeros((NX + (NX + NU) * self.N, 1))
        self.ubx = np.zeros((NX + (NX + NU) * self.N, 1))
                
        x_min, y_min = np.min(self.centre_line.path, axis=0) - 2
        x_max, y_max = np.max(self.centre_line.path, axis=0) + 2
        s_max = self.centre_line.s_path[-1] *1.5
        lbx = np.array([[x_min, y_min, self.planner_params.psi_min, 0]])
        ubx = np.array([[x_max, y_max, self.planner_params.psi_max, s_max]])
        for k in range(self.N + 1):
            self.lbx[NX * k:NX * (k + 1), 0] = lbx
            self.ubx[NX * k:NX * (k + 1), 0] = ubx

        state_count = NX * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + NU, 0] = np.array([[-self.planner_params.max_steer, self.planner_params.p_min]]) 
            self.ubx[state_count:state_count + NU, 0] = np.array([[self.planner_params.max_steer, self.planner_params.p_max]])  
            state_count += NU

    def init_bounds(self):
        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:NX]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.planner_params.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            self.g = ca.vertcat(self.g, self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1])  # LB<=ax-by<=UB  :represents path boundary constraints

    def init_objective(self):
        self.obj = 0  # Objective function
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.centre_interpolant.lut_angle(st_next[3])
            ref_x, ref_y = self.centre_interpolant.lut_x(st_next[3]), self.centre_interpolant.lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + countour_error **2 * self.planner_params.weight_contour  
            self.obj = self.obj + lag_error **2 * self.planner_params.weight_lag
            self.obj = self.obj - self.U[1, k] * self.planner_params.weight_progress 
            self.obj = self.obj + (self.U[0, k]) ** 2 * self.planner_params.weight_steer 
            
    def init_solver(self):
        optimisation_variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': optimisation_variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 2000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def plan(self, obs):
        self.step_counter += 1
        x0 = self.build_initial_state(obs)
        self.construct_warm_start_soln(x0) 
        self.set_up_constraints()
        p = self.generate_parameters(x0)
        states, controls = self.solve(p)

        np.save(self.mpcc_data_path + f"States_{self.step_counter}.npy", states)
        np.save(self.mpcc_data_path + f"Controls_{self.step_counter}.npy", controls)

        action = np.array([controls[0, 0], self.planner_params.vehicle_speed])

        return action 

    def build_initial_state(self, obs):
        x0 = obs["pose"]
        x0[2] = normalise_psi(x0[2]) 
        x0 = np.append(x0, self.centre_line.calculate_progress_m(x0[0:2]))
        # x0 = np.append(x0, self.centre_line.calculate_progress_m(x0[0:2]))

        return x0

    def generate_parameters(self, x0_in):
        p = np.zeros(NX + 2 * self.N)
        p[:NX] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            right_x = self.right_interpolant.lut_x(s_progress).full()[0, 0]
            right_y = self.right_interpolant.lut_y(s_progress).full()[0, 0]
            left_x = self.left_interpolant.lut_x(s_progress).full()[0, 0]
            left_y = self.left_interpolant.lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y
            p[NX + 2 * k:NX + 2 * k + 2] = [-delta_x, delta_y]

        return p
    
    def set_up_constraints(self):
        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s_progress = self.X0[k, 3]
            right_x = self.right_interpolant.lut_x(s_progress).full()[0, 0]
            right_y = self.right_interpolant.lut_y(s_progress).full()[0, 0]
            left_x = self.left_interpolant.lut_x(s_progress).full()[0, 0]
            left_y = self.left_interpolant.lut_y(s_progress).full()[0, 0]

            delta_x = right_x - left_x
            delta_y = right_y - left_y

            self.lbg[NX - 1 + (NX + 1) * (k + 1), 0] = min(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y) 
            self.ubg[NX - 1 + (NX + 1) * (k + 1), 0] = max(-delta_x * right_x - delta_y * right_y,
                                    -delta_x * left_x - delta_y * left_y)

        self.lbg[NX *2, 0] = - ca.inf
        self.ubg[NX *2, 0] = ca.inf

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.X0 = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T
        controls = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T
        states = self.X0# [:, 0:NX].T
        print(states)

        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print("Solve failed!!!!!")

        return states.full(), controls.full()
        
    def construct_warm_start_soln(self, initial_state):
        if not self.warm_start: return
        # self.warm_start = False

        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.planner_params.p_initial * self.planner_params.dt

            psi_next = self.centre_interpolant.lut_angle(s_next).full()[0, 0]
            x_next, y_next = self.centre_interpolant.lut_x(s_next), self.centre_interpolant.lut_y(s_next)

            # adjusts the centerline angle to be continuous
            psi_diff = self.X0[k-1, 2] - psi_next
            psi_mul = self.X0[k-1, 2] * psi_next
            if (abs(psi_diff) > np.pi and psi_mul < 0) or abs(psi_diff) > np.pi*1.5:
                if psi_diff > 0:
                    psi_next += np.pi * 2
                else:
                    psi_next -= np.pi * 2
            self.X0[k, :] = np.array([x_next.full()[0, 0], y_next.full()[0, 0], psi_next, s_next])

        print(self.X0)


def init_track_interpolants(centre_line, exclusion_width):
    widths = np.row_stack((centre_line.widths, centre_line.widths[1:int(centre_line.widths.shape[0] / 2), :]))
    path = np.row_stack((centre_line.path, centre_line.path[1:int(centre_line.path.shape[0] / 2), :]))
    extended_track = TrackLine(path)
    extended_track.init_path()
    extended_track.init_track()

    centre_interpolant = LineInterpolant(extended_track.path, extended_track.s_path, extended_track.psi)

    left_path = extended_track.path - extended_track.nvecs * np.clip((widths[:, 0][:, None]  - exclusion_width), 0, np.inf)
    left_interpolant = LineInterpolant(left_path, extended_track.s_path)
    right_path = extended_track.path + extended_track.nvecs * np.clip((widths[:, 1][:, None] - exclusion_width), 0, np.inf)
    right_interpolant = LineInterpolant(right_path, extended_track.s_path)

    return centre_interpolant, left_interpolant, right_interpolant

class LineInterpolant:
    def __init__(self, path, s_path, angles=None):
        self.lut_x = ca.interpolant('lut_x', 'bspline', [s_path], path[:, 0])
        self.lut_y = ca.interpolant('lut_y', 'bspline', [s_path], path[:, 1])
        if angles is not None:
            self.lut_angle = ca.interpolant('lut_angle', 'bspline', [s_path], angles)

    def get_point(self, s):
        return np.array([self.lut_x(s).full()[0, 0], self.lut_y(s).full()[0, 0]])


