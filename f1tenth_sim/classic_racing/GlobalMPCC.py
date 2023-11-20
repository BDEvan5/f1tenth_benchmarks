import numpy as np 
import casadi as ca

from f1tenth_sim.utils.BasePlanner import BasePlanner
from f1tenth_sim.utils.track_utils import CentreLine, TrackLine


NX = 4
NU = 3


class GlobalMPCC(BasePlanner):
    def __init__(self, test_id, save_data=False, planner_name="GlobalMPCC"):
        super().__init__(planner_name, test_id, params_name="GlobalMPCC")
        self.save_data = save_data
        if self.save_data: 
            self.mpcc_data_path = self.data_root_path + "mpcc_data/"
            self.create_clean_path(self.mpcc_data_path)
        self.track_width = None
        self.centre_interpolant = None
        self.left_interpolant, self.right_interpolant = None, None
        self.g, self.obj = None, None
        self.f_max = self.planner_params.friction_mu * self.vehicle_params.vehicle_mass * self.vehicle_params.gravity

        self.dt = self.planner_params.dt
        self.N = self.planner_params.N
        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time

        self.init_optimisation()
        self.init_constraints()
    
    def set_map(self, map_name):
        self.centre_line = CentreLine(map_name)
        widths = np.row_stack((self.centre_line.widths, self.centre_line.widths[1:int(self.centre_line.widths.shape[0] / 2), :]))
        path = np.row_stack((self.centre_line.path, self.centre_line.path[1:int(self.centre_line.path.shape[0] / 2), :]))
        extended_track = TrackLine(path)
        extended_track.init_path()
        extended_track.init_track()

        self.track_length = self.centre_line.s_path[-1]
        self.centre_interpolant = LineInterpolant(extended_track.path, extended_track.s_path, extended_track.psi)

        left_path = extended_track.path - extended_track.nvecs * np.clip((widths[:, 0][:, None]  - self.planner_params.exclusion_width), 0, np.inf)
        self.left_interpolant = LineInterpolant(left_path, extended_track.s_path)
        right_path = extended_track.path + extended_track.nvecs * np.clip((widths[:, 1][:, None] - self.planner_params.exclusion_width), 0, np.inf)
        self.right_interpolant = LineInterpolant(right_path, extended_track.s_path)

        self.init_objective()
        self.init_bounds()
        self.init_solver()

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
        lbx = [[p.position_min], [p.position_min], [-p.heading_max], [0]] * (self.N + 1) + [[-p.max_steer], [p.min_speed], [p.local_path_speed_min]] * self.N
        self.lbx = np.array(lbx)
        ubx = [[p.position_max], [p.position_max], [p.heading_max], [p.local_path_max_length]] * (self.N + 1) + [[p.max_steer], [p.max_speed], [p.local_path_speed_max]] * self.N
        self.ubx = np.array(ubx)

    def init_objective(self):
        self.obj = 0  # Objective function

        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.centre_interpolant.lut_angle(st_next[3])
            delta_x = st_next[0] - self.centre_interpolant.lut_x(st_next[3])
            delta_y = st_next[1] - self.centre_interpolant.lut_y(st_next[3])
            
            contouring_error = ca.sin(t_angle) * delta_x - ca.cos(t_angle) * delta_y 
            lag_error = -ca.cos(t_angle) * delta_x - ca.sin(t_angle) * delta_y 

            self.obj = self.obj + contouring_error **2 * self.planner_params.weight_contour  
            self.obj = self.obj + lag_error **2 * self.planner_params.weight_lag
            self.obj = self.obj - self.U[2, k] * self.planner_params.weight_progress
            self.obj = self.obj + (self.U[0, k]) ** 2 * self.planner_params.weight_steer

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

    def prepare_input(self, obs):
        x0 = np.append(obs["pose"], self.centre_line.calculate_progress_m(obs["pose"][0:2]))
        x0[2] = normalise_psi(x0[2]) 

        x0[3] = self.filter_estimate(x0[3]) # I think that I can remove this???

        return x0, obs["vehicle_speed"]

    def plan(self, obs):
        self.step_counter += 1
        x0, vehicle_speed = self.prepare_input(obs)

        p = self.generate_constraints_and_parameters(x0, vehicle_speed)
        states, controls, solved_status = self.solve(p)
        if not solved_status:
            self.warm_start = True
            p = self.generate_constraints_and_parameters(x0, vehicle_speed)
            states, controls, solved_status = self.solve(p)
            if not solved_status:
                print(f"Solve failed: ReWarm Start: New outcome: {solved_status}")
                print(f"S:{x0[3]:2f} --> Action: {controls[0, 0:2]}")
                return np.array([0, 1])

        if self.save_data:
            np.save(self.mpcc_data_path + f"States_{self.step_counter}.npy", states)
            np.save(self.mpcc_data_path + f"Controls_{self.step_counter}.npy", controls)

        action = controls[0, 0:2]

        return action 

    def generate_constraints_and_parameters(self, x0_in, x0_speed):
        self.lbg, self.ubg = np.zeros((self.g.shape[0], 1)), np.zeros((self.g.shape[0], 1))
        if self.warm_start:
            self.construct_warm_start_soln(x0_in) 

        pp = np.zeros(NX + 2 * self.N + 1)
        pp[:NX] = x0_in

        for k in range(self.N):  # set the reference controls and path boundary conditions to track
            s = self.X0[k, 3]
            if s > self.track_length:
                s = s - self.track_length
            right_point = [self.right_interpolant.lut_x(s).full()[0, 0], self.right_interpolant.lut_y(s).full()[0, 0]]
            left_point = [self.left_interpolant.lut_x(s).full()[0, 0], self.left_interpolant.lut_y(s).full()[0, 0]]

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
            # self.ubg[NX -1 + (NX + 3) * (k + 1) , 0] = self.max_v_dot

        # the optimizer cannot control the init state.
        # self.lbg[NX *2, 0] = - ca.inf #! I think this is wrong....
        # self.ubg[NX *2, 0] = ca.inf

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
        if (self.X0[0, 3] >= self.track_length) and (
                (initial_arc_pos >= self.track_length) or (initial_arc_pos <= 5)):
            self.X0[:, 3] = self.X0[:, 3] - self.track_length
        if initial_arc_pos >= self.track_length:
            initial_arc_pos -= self.track_length
        return initial_arc_pos

    def construct_warm_start_soln(self, initial_state):
        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + self.planner_params.p_initial * self.dt
            if s_next > self.track_length:
                s_next = s_next - self.track_length

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

        self.u0 = np.zeros((self.N, NU))
        self.u0[:, 1] = self.planner_params.p_initial
        self.u0[:, 2] = self.planner_params.p_initial

        self.warm_start = False


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi



class LineInterpolant:
    def __init__(self, path, s_path, angles=None):
        self.lut_x = ca.interpolant('lut_x', 'bspline', [s_path], path[:, 0])
        self.lut_y = ca.interpolant('lut_y', 'bspline', [s_path], path[:, 1])
        if angles is not None:
            self.lut_angle = ca.interpolant('lut_angle', 'bspline', [s_path], angles)

