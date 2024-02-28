import numpy as np 
import casadi as ca

from f1tenth_benchmarks.classic_racing.mpcc_utils import *
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner
from f1tenth_benchmarks.utils.track_utils import CentreLine, TrackLine

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
        self.optimisation_parameters = np.zeros(NX + 2 * self.N)

        self.init_optimisation()
        self.init_constraints()
    
    def set_map(self, map_name):
        self.centre_line = CentreLine(map_name)
        self.track_length = self.centre_line.s_path[-1]
        self.centre_interpolant, self.left_interpolant, self.right_interpolant = init_track_interpolants(self.centre_line, self.planner_params.exclusion_width)

        self.init_objective()
        self.init_bounds()
        self.init_bound_limits()
        self.init_solver()
       
    def init_optimisation(self):
        states = ca.MX.sym('states', NX) #[x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, p]

        rhs = ca.vertcat(self.planner_params.vehicle_speed * ca.cos(states[2]), 
                         self.planner_params.vehicle_speed * ca.sin(states[2]), 
                         (self.planner_params.vehicle_speed / self.vehicle_params.wheelbase) * ca.tan(controls[0]), 
                         controls[1])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX + 2 * self.N) # init state and boundaries of the reference path

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        p = self.planner_params
        lbx = [[p.position_min], [p.position_min], [-p.heading_max], [0]] * (self.N + 1)  + [[-p.max_steer], [p.local_path_speed_min]] * self.N
        ubx = [[p.position_max], [p.position_max], [p.heading_max], [p.local_path_max_length]] * (self.N + 1) + [[p.max_steer], [p.local_path_speed_max]] * self.N
        
        self.lbx = np.array(lbx)
        self.ubx = np.array(ubx)

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

    def init_bounds(self):
        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:NX]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.planner_params.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

            # LB<=ax-by<=UB  :represents path boundary constraints
            constraint = self.P[NX + 2 * k] * st_next[0] - self.P[NX + 2 * k + 1] * st_next[1]
            self.g = ca.vertcat(self.g, constraint)  
    
    def init_bound_limits(self):
        self.lbg = np.zeros((self.g.shape[0], 1))
        self.ubg = np.zeros((self.g.shape[0], 1))        

    def init_solver(self):
        optimisation_variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': optimisation_variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 2000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def prepare_input(self, obs):
        x0 = np.append(obs["pose"], self.centre_line.calculate_progress_m(obs["pose"][0:2]))
        x0[2] = normalise_psi(x0[2]) 
        self.optimisation_parameters[:NX] = x0

        return x0

    def plan(self, obs):
        self.step_counter += 1
        x0 = self.prepare_input(obs)
        self.set_path_constraints()
        states, controls, solved_status = self.solve()
        if not solved_status:
            self.construct_warm_start_soln(x0) 
            self.set_path_constraints()
            states, controls, solved_status = self.solve()
            if not solved_status:
                print(f"{self.step_counter} --> Optimisation has not been solved!!!!!!!!")
                return np.array([0, 1])

        if self.save_data:
            np.save(self.mpcc_data_path + f"States_{self.step_counter}.npy", states)
            np.save(self.mpcc_data_path + f"Controls_{self.step_counter}.npy", controls)

        action = np.array([controls[0, 0], self.planner_params.vehicle_speed])

        return action 

    def set_path_constraints(self):
        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            right_point = self.right_interpolant.get_point(self.X0[k, 3])
            left_point = self.left_interpolant.get_point(self.X0[k, 3])
            delta_point = right_point - left_point
            delta_point[0] = -delta_point[0]

            self.optimisation_parameters[NX + 2 * k:NX + 2 * k + 2] = delta_point

            right_bound = delta_point[0] * right_point[0] - delta_point[1] * right_point[1]
            left_bound = delta_point[0] * left_point[0] - delta_point[1] * left_point[1]

            self.lbg[NX - 1 + (NX + 1) * (k + 1), 0] = min(left_bound, right_bound)
            self.ubg[NX - 1 + (NX + 1) * (k + 1), 0] = max(left_bound, right_bound)

    def solve(self):
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=self.optimisation_parameters)

        self.X0 = states = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T
        controls = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T

        solved_status = True
        if self.solver.stats()['return_status'] == 'Infeasible_Problem_Detected':
            solved_status = False

        return states.full(), controls.full(), solved_status
        
    def construct_warm_start_soln(self, initial_state):
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

        if self.save_data:
            np.save(self.mpcc_data_path + f"x0_{self.step_counter}.npy", self.X0)


