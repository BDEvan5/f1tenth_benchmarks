
import json
import sys
from time import time_ns
from foxglove_schemas_protobuf import Pose_pb2

from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import numpy as np
import yaml, csv
from PIL import Image

import struct 
import base64

vehicle_name = "pp_traj_following"
map_name = "aut"
file_name = f"Logs/{vehicle_name}/test_{map_name}.mcap"
schema_path = "f1tenth_sim/data_tools/schemas/"


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
        self.load_centerline()

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
        
    def load_centerline(self):
        track = np.loadtxt(self.path + self.map_name + "_centerline.csv", delimiter=',', skiprows=1)
        self.wpts = track[:, :2]
        self.N = len(self.wpts)
        

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

states, actions = load_agent_test_data(f"Logs/{vehicle_name}/SimLog_{map_name}_0.npy")
map_data = MapData(map_name)

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

    start_time = time_ns()

    grid = {"frame_id": "map"}
    grid["timestamp"] = {
            "sec": int(start_time * 1e-9),
            "nsec": int(start_time - int(start_time * 1e-9))}
    grid["pose"] = {
            "position": {"x": map_data.map_origin[0], "y": map_data.map_origin[1], "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 0} }
    grid["column_count"] = map_data.map_img.shape[0]
    grid["cell_size"] = {"x": 0.05, "y": 0.05}
    grid["row_stride"] =  map_data.map_img.shape[1] * 4
    grid["cell_stride"] = 4
    grid["fields"] = [
        {"name": "red", "offset": 0, "type": 7},
    ]
    img = map_data.map_img.astype(np.float32) * 255
    grid["data"] = base64.b64encode(img).decode("utf-8")
    publish_message(writer, grid_channel_id, grid, start_time)

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

    writer.finish()

    # print(start_time)