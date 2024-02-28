
import numpy as np
from numba import njit
from f1tenth_benchmarks.utils.track_utils import RaceTrack, CentreLine
from f1tenth_benchmarks.utils.BasePlanner import BasePlanner


class GlobalPurePursuit(BasePlanner):
    def __init__(self, test_id, use_centre_line=False, planner_name="GlobalPurePursuit", init_folder=True, extra_params={}):
        self.use_centre_line = use_centre_line
        super().__init__(planner_name, test_id, params_name="GlobalPurePursuit", init_folder=init_folder, extra_params=extra_params)
        self.racetrack = None
        self.use_centre_line = use_centre_line

        if self.planner_params.training:
            self.constant_lookahead = self.planner_params.constant_lookahead
            self.variable_lookahead = self.planner_params.variable_lookahead

        else:
            self.constant_lookahead = self.planner_params.tal_constant_lookahead
            self.variable_lookahead = self.planner_params.tal_variable_lookahead

    def set_map(self, map_name):
        if self.use_centre_line:
            self.racetrack = CentreLine(map_name)
        else:
            self.racetrack = RaceTrack(map_name, self.planner_params.racetrack_set)

    def plan(self, obs):
        self.step_counter += 1
        pose = obs["pose"]
        vehicle_speed = obs["vehicle_speed"]

        lookahead_distance = self.constant_lookahead + (vehicle_speed/self.vehicle_params.max_speed) * (self.variable_lookahead)
        lookahead_point, i = self.get_lookahead_point(pose[:2], lookahead_distance)

        if vehicle_speed < 1:
            return np.array([0.0, 4])

        true_lookahead_distance = np.linalg.norm(lookahead_point[:2] - pose[:2])
        steering_angle = get_actuation(pose[2], lookahead_point, pose[:2], true_lookahead_distance, self.vehicle_params.wheelbase)
        if self.planner_params.training:
            steering_angle = get_actuation(pose[2], lookahead_point, pose[:2], self.planner_params.tal_actuation_distance, self.vehicle_params.wheelbase)
        steering_angle = np.clip(steering_angle, -self.planner_params.max_steer, self.planner_params.max_steer)
            
        if self.use_centre_line:
            speed = self.planner_params.constant_speed
        else:
            speed = min(self.racetrack.speeds[i], self.vehicle_params.max_speed)
            speed = min(speed, calculate_speed_limit(steering_angle, self.planner_params.friction_limit))
        action = np.array([steering_angle, speed])

        return action

    def get_lookahead_point(self, position, lookahead_distance):
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory_py2(position, self.racetrack.path, self.racetrack.l2s, self.racetrack.diffs)

        lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, self.racetrack.path, i+t, wrap=True)
        if i2 == None: # this happens when the circle does not intersect the trajectory.
            i2 = i + int(lookahead_distance / np.sqrt(self.racetrack.l2s[i]))
            # return None
        lookahead_point = np.empty((3, ))
        if i2 >= len(self.racetrack.path) - 1: i2 = len(self.racetrack.path) - 2
        lookahead_point[0:2] = self.racetrack.path[i2, :]
        
        return lookahead_point, i


GRAVITY = 9.81
MAX_SPEED = 8
WHEELBASE = 0.33
@njit(cache=True)
def calculate_speed_limit(delta, friction_limit=1.2):
    if abs(delta) < 0.03:
        return MAX_SPEED

    V = np.sqrt(friction_limit*GRAVITY*WHEELBASE/np.tan(abs(delta)))
    V = min(V, MAX_SPEED)

    return V

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    if np.abs(waypoint_y) < 1e-6:
        return 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return steering_angle


@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory_py2(point, trajectory, l2s, diffs):
    '''
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    '''
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    ''' starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    '''
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

