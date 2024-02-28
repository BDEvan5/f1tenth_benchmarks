
import json
from time import time_ns

from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import numpy as np
import yaml
from PIL import Image

import base64

vehicle_name = "MPCC"
# map_name = "esp"
map_name = "aut"
file_name = f"Logs/{vehicle_name}/test_{map_name}.mcap"
schema_path = "f1tenth_benchmarks/data_tools/mcap/schemas/"


class MapData:
    def __init__(self, map_name):
        self.path = "maps/"
        self.map_name = map_name

        self.xs, self.ys = None, None

        self.N = 0
        self.map_resolution = None
        self.map_origin = None
        self.map_img = None
        self.map_height = None
        self.map_width = None

        self.load_map_img()

    def load_map_img(self):
        with open(self.path + self.map_name + ".yaml", 'r') as file:
            map_yaml_data = yaml.safe_load(file)
            self.map_resolution = map_yaml_data["resolution"]
            self.map_origin = map_yaml_data["origin"]
            map_img_name = map_yaml_data["image"]

        self.map_img = np.array(Image.open(self.path + map_img_name).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 1.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]
        
import os
import trajectory_planning_helpers as tph
class Trajectory:
    def __init__(self, map_name) -> None:
        filename = f"racelines/" + map_name + "_raceline.csv"
        self.track = np.loadtxt(filename, delimiter=',', skiprows=1)

class TrackPath:
    def __init__(self, map_name) -> None:
        self.track = np.loadtxt("maps/" + map_name + "_centerline.csv", delimiter=',', skiprows=1)

        self.el_lengths = np.linalg.norm(np.diff(self.track[:, :2], axis=0), axis=1)
        self.s_track = np.insert(np.cumsum(self.el_lengths), 0, 0)
        self.psi, self.kappa = tph.calc_head_curv_num.calc_head_curv_num(self.track, self.el_lengths, False)

        self.nvecs = tph.calc_normal_vectors_ahead.calc_normal_vectors_ahead(self.psi)

        self.left_path = self.track[:, :2] - self.nvecs * (self.track[:, 2][:, None])
        self.right_path = self.track[:, :2] + self.nvecs * (self.track[:, 3][:, None])



def register_schema(writer, schema_name, schema_path):
    with open(schema_path, "rb") as f:
        schema = f.read()

    schema_id = writer.register_schema(
        name=schema_name,
        encoding=SchemaEncoding.JSONSchema,
        data=schema,
    )

    return schema_id

def register_channel(writer, channel_name, schema_id):
    channel_id = writer.register_channel(
        topic=channel_name,
        message_encoding=MessageEncoding.JSON,
        schema_id=schema_id,
    )

    return channel_id

def publish_message(writer, channel_id, msg, time):
    writer.add_message(
        channel_id=channel_id,
        log_time=time,
        data=json.dumps(msg).encode("utf-8"),
        publish_time=time,
    )



def load_agent_test_data(file_name):
    data = np.load(file_name)
    return data[:, :7], data[:, 7:]

states, actions = load_agent_test_data(f"Logs/{vehicle_name}/RawData/SimLog_{map_name}_0.npy")
try:
    scans = np.load(f"Logs/{vehicle_name}/ScanLog_{map_name}_0.npy")
except:
    scans = None
map_data = MapData(map_name)
traj_data = Trajectory(map_name)
track_data = TrackPath(map_name)

def build_pose_list_msgs(data, start_time):
    trajectory = {"frame_id": "map"}
    trajectory["timestamp"] = {
            "sec": int(start_time * 1e-9),
            "nsec": int(start_time - int(start_time * 1e-9))}
    trajectory["poses"] = []
    for i in range(len(data)):
        pose = {"position": {"x": data[i, 0], "y": data[i, 1], "z": 0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 0}}
        trajectory["poses"].append(pose)
    return trajectory

with open(file_name, "wb") as stream:
    writer = Writer(stream)
    writer.start('x-jsonschema')

    schema_id = register_schema(writer, "foxglove.PoseInFrame", f"{schema_path}PoseInFrame.json")
    pose_channel_id = register_channel(writer, "vehicle_pose", schema_id)
    schema_id = register_schema(writer, "foxglove.Grid", f"{schema_path}Grid.json")
    grid_channel_id = register_channel(writer, "grid", schema_id)
    schema_id = register_schema(writer, "foxglove.Vector1", f"{schema_path}Vector1.json")
    speed_channel_id = register_channel(writer, "speed_states", schema_id)
    steering_channel_id = register_channel(writer, "steering_states", schema_id)
    speed_actions_channel_id = register_channel(writer, "speed_actions", schema_id)
    steering_actions_channel_id = register_channel(writer, "steering_actions", schema_id)
    yaw_rate_channel_id = register_channel(writer, "yaw_rate", schema_id)
    slip_angle_channel_id = register_channel(writer, "slip_angle", schema_id)
    lap_time_channel_id = register_channel(writer, "laptime", schema_id)

    schema_id = register_schema(writer, "foxglove.PosesInFrame", f"{schema_path}PosesInFrame.json")
    trajectory_channel_id = register_channel(writer, "trajectory", schema_id)
    l_boudnary_channel_id = register_channel(writer, "l_boundaries", schema_id)
    r_boudnary_channel_id = register_channel(writer, "r_boundaries", schema_id)
    schema_id = register_schema(writer, "foxglove.FrameTransform", f"{schema_path}FrameTransform.json")
    tf_channel_id = register_channel(writer, "tf", schema_id)

    schema_id = register_schema(writer, "foxglove.LaserScan", f"{schema_path}LaserScan.json")
    scan_channel_id = register_channel(writer, "scan", schema_id)

    start_time = time_ns()

    grid = {"frame_id": "map"}
    grid["timestamp"] = {
            "sec": int(start_time * 1e-9),
            "nsec": int(start_time - int(start_time * 1e-9))}
    grid["pose"] = {
            "position": {"x": map_data.map_origin[0], "y": map_data.map_origin[1], "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 0} }
    grid["column_count"] = map_data.map_img.shape[1]
    grid["cell_size"] = {"x": 0.05, "y": 0.05}
    grid["row_stride"] =  map_data.map_img.shape[1] * 4
    grid["cell_stride"] = 4
    grid["fields"] = [
        {"name": "red", "offset": 0, "type": 7},
    ]
    img = map_data.map_img.astype(np.float32) * 255
    grid["data"] = base64.b64encode(img).decode("utf-8")
    publish_message(writer, grid_channel_id, grid, start_time)

    trajectory = build_pose_list_msgs(traj_data.track[:, 1:3], start_time)
    publish_message(writer, trajectory_channel_id, trajectory, start_time)

    left_bound = build_pose_list_msgs(track_data.left_path, start_time)
    publish_message(writer, l_boudnary_channel_id, left_bound, start_time)
    right_bound = build_pose_list_msgs(track_data.right_path, start_time)
    publish_message(writer, r_boudnary_channel_id, right_bound, start_time)

    timestep = 0.05
    for i in range(len(states)):
        time = int(start_time + i * 1e9 * timestep)
        time_in_s = int(time * 1e-9)
        time_in_ns = int(time - time_in_s)

        pose = {"frame_id": "map"}
        pose["pose"] = {
            "position": {"x": states[i, 0], "y": states[i, 1], "z": 0},
            "orientation": {"x": 0, "y": 0, "z": np.sin(states[i, 4]/2), "w": np.cos(states[i, 4]/2)}}
        pose["timestamp"] = {"sec": time_in_s, "nsec": time_in_ns}

        publish_message(writer, pose_channel_id, pose, time)
        publish_message(writer, speed_channel_id, {"data": states[i, 3]}, time)
        publish_message(writer, steering_channel_id, {"data": states[i, 2]}, time)
        publish_message(writer, steering_actions_channel_id, {"data": actions[i, 0]}, time)
        publish_message(writer, speed_actions_channel_id, {"data": actions[i, 1]}, time)
        publish_message(writer, yaw_rate_channel_id, {"data": states[i, 5]}, time)
        publish_message(writer, slip_angle_channel_id, {"data": states[i, 6]}, time)
        publish_message(writer, lap_time_channel_id, {"data": round(i*timestep+0.001, 3)}, time)

        # if i > 5:
        tf = {"parent_frame_id": "map"}
        tf["child_frame_id"] = f"vehicle"
        # tf["child_frame_id"] = f"vehicle_{i}"
        tf["timestamp"] = {"sec": time_in_s, "nsec": time_in_ns}
        tf["translation"] = {"x": states[i, 0], "y": states[i, 1], "z": 0}
        tf["rotation"] = {"x": 0, "y": 0, "z": np.sin(states[i, 4]/2), "w": np.cos(states[i, 4]/2)}
        publish_message(writer, tf_channel_id, tf, time)

        if scans is not None:
            scan_msg = {"frame_id": "map"}
            scan_msg["timestamp"] = {"sec": time_in_s, "nsec": time_in_ns}
            scan_msg["start_angle"] = -2.35
            scan_msg["end_angle"] = 2.35
            # scan_msg["ranges"] = base64.b64encode(scans[i]).decode("utf-8")
            scan_msg["ranges"] = scans[i].tolist()
            scan_msg["pose"] = {"position": {"x": states[i, 0], "y": states[i, 1], "z": 0},
                "orientation": {"x": 0, "y": 0, "z": np.sin(states[i, 4]/2), "w": np.cos(states[i, 4]/2)}}
            publish_message(writer, scan_channel_id, scan_msg, time)

    writer.finish()

    # print(start_time)