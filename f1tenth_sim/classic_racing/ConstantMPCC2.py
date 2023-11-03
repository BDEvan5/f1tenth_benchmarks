import matplotlib.pyplot as plt
import numpy as np 
import casadi as ca
from argparse import Namespace

from f1tenth_sim.classic_racing.planner_utils import RaceTrack
from f1tenth_sim.classic_racing.ReferencePath import ReferencePath

VERBOSE = False
# VERBOSE = True


def normalise_psi(psi):
    while psi > np.pi:
        psi -= 2*np.pi
    while psi < -np.pi:
        psi += 2*np.pi
    return psi

NX = 4
NU = 2

p = {
    "position_min": -100,
    "position_max": 100,
    "heading_max": 10,
    "delta_max": 0.4,
    "local_path_max_length": 200,
    "max_speed": 8,
    "min_speed": 1,
    "local_path_speed_min": 2,
    "local_path_speed_max": 10,
    "max_v_dot": 0.04*8,
    "wheelbase": 0.33,
    "vehicle_speed": 2,
    "weight_progress": 0,
    "weight_lag": 1,
    "weight_contour": 10,
    "weight_steer": 1,
    "p_initial": 2,
}
p = Namespace(**p)


class ConstantMPCC2:
    def __init__(self):
        self.name = "ConstantMPCC"
        self.N = 20
        self.dt = 0.04
        self.rp = None

        self.u0 = np.zeros((self.N, NU))
        self.X0 = np.zeros((self.N + 1, NX))
        self.warm_start = True # warm start every time

        self.init_optimisation()

    def set_map(self, map_name):
        self.rp = ReferencePath(map_name, 0.25)
        self.init_constraints()
        self.init_bounds()
        self.init_objective()
        self.init_solver()

       
    def init_optimisation(self):
        states = ca.MX.sym('states', NX) #[x, y, psi, s]
        controls = ca.MX.sym('controls', NU) # [delta, p]

        rhs = ca.vertcat(p.vehicle_speed * ca.cos(states[2]), p.vehicle_speed * ca.sin(states[2]), (p.vehicle_speed / p.wheelbase) * ca.tan(controls[0]), controls[1])  # dynamic equations of the states
        self.f = ca.Function('f', [states, controls], [rhs])  # nonlinear mapping function f(x,u)
        
        self.U = ca.MX.sym('U', NU, self.N)
        self.X = ca.MX.sym('X', NX, (self.N + 1))
        self.P = ca.MX.sym('P', NX) # init state 

    def init_constraints(self):
        '''Initialize upper and lower bounds for state and control variables'''
        self.lbg = np.zeros((NX * (self.N + 1), 1))
        self.ubg = np.zeros((NX * (self.N + 1), 1))
        self.lbx = np.zeros((NX + (NX + NU) * self.N, 1))
        self.ubx = np.zeros((NX + (NX + NU) * self.N, 1))
                
        x_min, y_min = np.min(self.rp.path, axis=0) - 2
        x_max, y_max = np.max(self.rp.path, axis=0) + 2
        s_max = self.rp.s_track[-1] *1.5
        lbx = np.array([[x_min, y_min, -p.heading_max, 0]])
        ubx = np.array([[x_max, y_max, p.heading_max, s_max]])
        for k in range(self.N + 1):
            self.lbx[NX * k:NX * (k + 1), 0] = lbx
            self.ubx[NX * k:NX * (k + 1), 0] = ubx

        state_count = NX * (self.N + 1)
        for k in range(self.N):
            self.lbx[state_count:state_count + NU, 0] = np.array([[-p.delta_max, p.local_path_speed_min]]) 
            self.ubx[state_count:state_count + NU, 0] = np.array([[p.delta_max, p.local_path_speed_max]])  
            state_count += NU

    def init_bounds(self):
        """Initialise the bounds (g) on the dynamics and track boundaries"""
        self.g = self.X[:, 0] - self.P[:NX]  # initial condition constraints
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            k1 = self.f(self.X[:, k], self.U[:, k])
            st_next_euler = self.X[:, k] + (self.dt * k1)
            self.g = ca.vertcat(self.g, st_next - st_next_euler)  # add dynamics constraint

    def init_objective(self):
        self.obj = 0  # Objective function
        for k in range(self.N):
            st_next = self.X[:, k + 1]
            t_angle = self.rp.angle_lut_t(st_next[3])
            ref_x, ref_y = self.rp.center_lut_x(st_next[3]), self.rp.center_lut_y(st_next[3])
            countour_error = ca.sin(t_angle) * (st_next[0] - ref_x) - ca.cos(t_angle) * (st_next[1] - ref_y)
            lag_error = -ca.cos(t_angle) * (st_next[0] - ref_x) - ca.sin(t_angle) * (st_next[1] - ref_y)

            self.obj = self.obj + countour_error **2 * p.weight_contour  
            self.obj = self.obj + lag_error **2 * p.weight_lag
            self.obj = self.obj + (self.U[0, k]) ** 2 * p.weight_steer 
            
    def init_solver(self):
        optimisation_variables = ca.vertcat(ca.reshape(self.X, NX * (self.N + 1), 1),
                                ca.reshape(self.U, NU * self.N, 1))

        nlp_prob = {'f': self.obj, 'x': optimisation_variables, 'g': self.g, 'p': self.P}
        opts = {"ipopt": {"max_iter": 2000, "print_level": 0}, "print_time": 0}
        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

    def plan(self, obs):
        x0 = self.build_initial_state(obs)
        self.construct_warm_start_soln(x0) 
        controls = self.solve(x0)

        action = np.array([controls[0, 0], p.vehicle_speed])

        return action 

    def build_initial_state(self, obs):
        x0 = obs["vehicle_state"][np.array([0, 1, 4])]
        x0[2] = normalise_psi(x0[2]) 
        x0 = np.append(x0, self.rp.calculate_s(x0[0:2]))

        return x0

    def solve(self, p):
        x_init = ca.vertcat(ca.reshape(self.X0.T, NX * (self.N + 1), 1),
                         ca.reshape(self.u0.T, NU * self.N, 1))

        sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)

        self.X0 = ca.reshape(sol['x'][0:NX * (self.N + 1)], NX, self.N + 1).T
        controls = ca.reshape(sol['x'][NX * (self.N + 1):], NU, self.N).T

        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print("Solve failed!!!!!")

        return controls.full()
        
    def construct_warm_start_soln(self, initial_state):
        if not self.warm_start: return
        # self.warm_start = False

        self.X0 = np.zeros((self.N + 1, NX))
        self.X0[0, :] = initial_state
        for k in range(1, self.N + 1):
            s_next = self.X0[k - 1, 3] + p.p_initial * self.dt

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



