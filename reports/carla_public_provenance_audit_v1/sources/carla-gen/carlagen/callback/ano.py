"""
Callbacks related to causing anomalous behavior
"""

import math
import os
import shutil
import random
from os.path import join
from typing import List, Dict, Any
from typing import TYPE_CHECKING

import carla
import numpy as np
import pandas as pd
from PIL import Image
import cv2


import logging


from .callback import Callback
from .saving import SaveCameraCallback
from .utils import (
    check_if_is_in_front_of_ego_vehicle,
    find_next_tf,
    check_if_is_in_front_of,
    angle_between_vectors_2d,
)
from carlagen.utils.carla import set_weather
from carlagen.utils.carla import get_vehicle_type
from carlagen.utils.carla import is_far_away, is_too_cloose

if TYPE_CHECKING:
    from carlagen import Simulation


class EventScheduler:
    """
    Helper class used to schedule anomalous events

    Frequency: How often in the clip should something be scheduled

    duration: lower and upper limit for event duration
    """
    def __init__(self, mode: str, seed=1234, interval=5, delay=5, duration=[20,50], frequency=3, cool_down=[2], time_between_events=10, **kwargs):

        assert mode in ["static", "dynamic", "frequency"]

        self.mode = mode
        self.log = logging.getLogger(type(self).__name__)

        self.interval = interval
        self.duration = duration
        self.cool_down = cool_down
        self.frequency = frequency
        self.time_between_events = time_between_events
        self.delay = delay

        self.ticks_total = None

        self.rng = np.random.default_rng(seed)

        self.log.info(f"Initializing event scheduler with {self.mode=} {self.frequency=} {self.duration=} {self.cool_down=}")

    def set_ticks(self, ticks_total):
        self.ticks_total = ticks_total
        self._create_plan()

    def get_duration(self):
        if len(self.duration) == 1:
            return self.duration[0]
        else:
            return int(self.rng.uniform(self.duration[0], self.duration[1]))

    def get_cool_down(self):
        if len(self.cool_down) == 1:
            return self.cool_down[0]
        else:
            return int(self.rng.uniform(self.cool_down[0], self.cool_down[1]))

    def _generate_schedule(self) -> List[bool]:
        result = [False] * self.ticks_total
        possible_starts = list(
            range(self.delay, self.ticks_total - self.duration[-1] - 10)
        )

        for n in range(self.frequency):
            if not possible_starts:
                break  # Falls keine Startpunkte mehr verfügbar sind

            start = self.rng.choice(possible_starts)
            tmp_duration = self.get_duration()
            self.log.info(f"Setting event sequence {n} to start at {start} for {tmp_duration}")

            for i in range(start, start + tmp_duration):
                result[i] = True

            # remove all possible starts that are too close to the current event
            possible_starts = [
                x
                for x in possible_starts
                if x < start - (self.time_between_events + self.duration[-1])
                or x >= start + tmp_duration + self.time_between_events
            ]

        return result

    def _create_plan(self):
        self.existing_frames = []
        self.not_existing_frames = []
        if self.mode == "dynamic":
            dynamic_information = []
            while len(dynamic_information) < self.ticks_total:
                cool_down = self.get_cool_down()
                for i in range(0, cool_down):
                    dynamic_information.append(False)
                duration = self.get_duration()
                for i in range(0, duration):
                    dynamic_information.append(True)

        if self.mode == "frequency":
            dynamic_information = self._generate_schedule()

        for frame in range(0, self.ticks_total):
            if self.delay is not None:
                if frame < self.delay:
                    self.not_existing_frames.append(frame)
                    continue

            if self.mode == "static_interval":
                # existing frames --> muss eins früher gespawnt werden
                modulo_classes = [self.interval - 1]
                ## append in range of duration
                for i in range(0, self.get_duration() - 1):
                    modulo_classes.append(self.interval + i)
                if frame % self.interval in modulo_classes:
                    self.existing_frames.append(frame)
                else:
                    self.not_existing_frames.append(frame)
            elif self.mode in ["dynamic_interval", "frequency"]:
                if dynamic_information[frame]:
                    self.existing_frames.append(frame)
                else:
                    self.not_existing_frames.append(frame)
            else:
                raise NotImplementedError(f"Mode {self.mode} not implemented")

    def __call__(self, frame: int):
        return self.is_scheduled_for(frame)

    def set_not_possible(self, frame: int):
        """
        Notify scheduler that execution at given frame was not possible
        """
        # existing at frame is not possible
        # remove frame from existing frames
        if frame in self.existing_frames:
            self.existing_frames.remove(frame)
        # add frame to not existing frames
        self.not_existing_frames.append(frame)

        switchframe = None
        # add the next not existing frame to existing frames
        self.not_existing_frames.sort()
        self.existing_frames.sort()
        for not_existing_frame in self.not_existing_frames:
            if not_existing_frame > frame:
                self.existing_frames.append(not_existing_frame)
                self.not_existing_frames.remove(not_existing_frame)
                switchframe = not_existing_frame
                break
        self.log.info(f"Could not exec scheduled event at {frame}, rescheduling to {switchframe}")
        return switchframe

    def is_scheduled_for(self, frame: int):
        if frame in self.existing_frames:
            return True
        else:
            return False

    def is_not_scheduled_for(self, frame: int):
        if frame in self.not_existing_frames:
            return True
        else:
            return False

    def is_not_scheduled_anymore(self, frame: int):
        # true if frame is not existing and the frame before is existing
        if frame in self.not_existing_frames:
            if frame - 1 in self.existing_frames:
                return True
        return False

    def get_last_existing_before_frame(self, current_frame: int):
        for frame in reversed(self.existing_frames):
            if frame < current_frame:
                return frame
        return -1

    def get_time_since_last_existing(self, current_frame: int):
        last_existing = self.get_last_existing_before_frame(current_frame)
        if last_existing == -1:
            return -1
        else:
            return current_frame - last_existing

    def is_first_scheduled_for(self, frame: int):
        """
        Return true if the frame is the first for which an event is scheduled
        """
        if frame in self.existing_frames:
            if frame - 1 in self.not_existing_frames:
                return True
        return False

    def get_current_event_duration(self, frame: int):
        if frame in self.existing_frames:
            duration = 0
            for i in range(frame, self.ticks_total):
                if i in self.existing_frames:
                    duration += 1
                else:
                    return duration
        return -1

    def get_first_frame_of_current_event(self, frame: int):
        while frame in self.existing_frames:
            frame -= 1
        return frame + 1


class AnomalousCallback(Callback):
    """
    Base class for anomaly inducing callbacks
    """
    def __init__(
        self,
        seed=1234,
        scheduler_mode = "frequency",
        scheduler_seed = 1234,
        **kwargs,
    ):
        """
        """
        super().__init__(**kwargs)
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.scheduler = EventScheduler(mode=scheduler_mode, **kwargs, seed=scheduler_seed)

    def on_simulation_start(self, sim, **kwargs):
        self.scheduler.set_ticks(sim.end_tick)
        self.log.info(f"Scheduler Mode: {self.scheduler.mode}")
        self.log.info(f"Scheduler existing_frames: {self.scheduler.existing_frames}")


class ObservationLevelAnomalousCallback(AnomalousCallback):
    """
    For anomalies on a "per observation" level.

    Will write an "anomaly-observation.feather" file that indicates if an observation (at a timestep) is anomalous.
    """

    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.observation_lvl_data = []


    def register_observation_anomaly_tick(self, sim, anomaly: bool, meta: dict = None):
        """
        Add info for a single observation during sim
        """
        entry = {"anomaly": anomaly, "tick": sim.tick_count}

        if meta:
            entry.update(meta)

        self.observation_lvl_data.append(entry)

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        """
        Write resulting file
        """
        for _ in self.observation_lvl_data:
            # TODO: some postprocessing, maybe?
            pass

        data = pd.DataFrame(self.observation_lvl_data)
        data.to_feather(f"{sim.root}/anomaly-observation.feather")



class SampleLevelAnomalousCallback(ObservationLevelAnomalousCallback):
    """
    For anomalies on the pixel or point level

    During simulation, we trigger certain anomalies. These will cause objects
    with certain IDs or certain classes to be considered anomalous.

    We mark these objects as anomalies and write output masks (pointclouds or semantic segmentation
    or whatever).

    Furthermore, if there is some anomaly, we will also write info into the
    observation level data.
    """

    def __init__(
            self,
            sensors: List[Dict],
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.anomalous_actor_ids = []
        self.anomalous_class_ids = []
        self.sensors = sensors

        self.sensor_summary = {s["name"]: [] for s in sensors}

    def on_simulation_tick_ends(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        # get all sensors and send to the correct processors
        for sensor in self.sensors:
            sensor_name = sensor["name"]
            sensor_dir = sensor["directory"]

            data = measurements[sensor_name]

            contains_anomaly = False

            # self.log.info(f"Sensor {sensor} has {data.__class__} observations")
            if isinstance(data, carla.libcarla.Image):
                mask = self.process_camera_data(
                    data,
                    anomaly_obj_ids=self.anomalous_actor_ids,
                    anomaly_class_ids=self.anomalous_class_ids
                )

                contains_anomaly = mask.any()
                out_dir = join(sim.root, sensor_dir)
                os.makedirs(out_dir, exist_ok=True)

                out_path = join(out_dir, f"{sim.tick_count:06d}.png")
                Image.fromarray(mask, mode="L").save(out_path, format="PNG", compress_level=9)

            elif isinstance(data, carla.libcarla.SemanticLidarMeasurement):
                mask = self.process_lidar_data(
                    data,
                    anomaly_obj_ids=self.anomalous_actor_ids,
                    anomaly_class_ids=self.anomalous_class_ids
                )

                contains_anomaly = mask.any()

                out_dir = join(sim.root, sensor_dir)
                os.makedirs(out_dir, exist_ok=True)

                out_path = join(out_dir, f"{sim.tick_count:06d}.feather")
                pd.DataFrame(np.array(mask, dtype=bool), columns=["anomaly"]).to_feather(out_path, compression="zstd")

                self.log.info(f"Ano points: {mask.sum()}")
            else:
                self.log.warn(f"Unsupported type {type(data)}")

            self.sensor_summary[sensor_name].append(contains_anomaly)

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        for sensor in self.sensors:
            sensor_name = sensor["name"]
            sensor_dir = sensor["directory"]
            data = self.sensor_summary[sensor_name]
            out_path = join(sim.root, sensor_dir, f"sensor.feather")
            self.log.info(f"Writing sensor info to {out_path}")
            pd.DataFrame(data, columns=["anomaly"]).to_feather(out_path, compression="zstd")

        super().on_simulation_end(sim, **kwargs)

    def register_anomaly_tick(self, sim,
                              anomaly_obj_ids: List[int] = None,
                              anomaly_class_ids: List[int] = None,
                              meta: Dict = None):
        """

        """
        self.anomalous_actor_ids = anomaly_obj_ids or []
        self.anomalous_class_ids = anomaly_class_ids or []

        if not anomaly_obj_ids and not anomaly_class_ids:
            # no anomalies in this timestep.
            self.register_observation_anomaly_tick(sim, False, meta={
                "anomaly_obj_ids": None,
                "anomaly_class_ids": None,
                "meta": {"type": None}
            })
        else:
            self.register_observation_anomaly_tick(sim, True, meta={
                "anomaly_obj_ids": anomaly_obj_ids,
                "anomaly_class_ids": anomaly_class_ids,
                "meta": {"type": self.__class__.__name__}
            })

    def process_camera_data(self, img, anomaly_obj_ids, anomaly_class_ids):
        # 1. Convert raw buffer into array
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            (img.height, img.width, 4)
        )

        # 2. Reorder BGRA → RGBA
        rgba = arr[:, :, [2, 1, 0, 3]]

        # 3. Create PIL image
        img = Image.fromarray(rgba, mode="RGBA").convert("RGB")

        id_fields = np.array(img)[:, :, 1:3]

        instance_ids = np.zeros(shape=(id_fields.shape[0], id_fields.shape[1]), dtype=np.int32)
        instance_ids += id_fields[:, :, 0].astype(np.int32)
        instance_ids += id_fields[:, :, 1].astype(np.int32) << 8

        # per-pixel classes are encoded in the R channel
        segmentation_mask = np.array(img)[:, :, 0]

        mask = np.zeros(shape=(rgba.shape[0], rgba.shape[1]), dtype=np.uint8)

        if anomaly_obj_ids:
            self.log.info(f"Instances: {np.unique(instance_ids)}")

        for id in anomaly_obj_ids:
            id = id % 2**16 # the ids are reduced to 16bit precision
            self.log.info(f"CAM: Processing anomaly instance {id} {(instance_ids == id).sum()}")
            mask[instance_ids == id] = 255

        for id in anomaly_class_ids:
            self.log.info(f"CAM: Processing anomaly class {id} {(segmentation_mask == id).sum()}")
            mask[segmentation_mask == id] = 255

        return mask


    def process_lidar_data(self, lidar, anomaly_obj_ids, anomaly_class_ids):
        # TODO: add metadata
        segmentation_mask = np.array([ detection.object_tag for detection in lidar])
        instance_ids = np.array([ detection.object_idx for detection in lidar])
        mask = np.zeros(len(segmentation_mask), dtype=np.uint8)

        for id in anomaly_obj_ids:
            self.log.info(f"Processing anomaly instance {id} {(instance_ids == id).sum()}")
            mask[instance_ids == id] = 255

        for id in anomaly_class_ids:
            self.log.info(f"Processing anomaly class {id} {(segmentation_mask == id).sum()}")
            mask[segmentation_mask == id] = 255

        return mask


class SpawnMiscCallback(SampleLevelAnomalousCallback):
    """
    Spawn Misc Props
    """

    # TODO: make configurable
    spawnable_props = [
        "static.prop.barrel",
        "static.prop.bin",
        "static.prop.clothcontainer",
        "static.prop.container",
        "static.prop.glasscontainer",
        "static.prop.box01",
        "static.prop.box02",
        "static.prop.box03",
        "static.prop.trashbag",
        "static.prop.trashcan01",
        "static.prop.trashcan02",
        "static.prop.trashcan03",
        "static.prop.trashcan04",
        "static.prop.trashcan05",
        "static.prop.bench01",
        "static.prop.bench02",
        "static.prop.bench03",
        "static.prop.gardenlamp",
        "static.prop.plasticchair",
        "static.prop.barbeque",
        "static.prop.doghouse",
        "static.prop.gnome",
        "static.prop.haybale",
        "static.prop.haybalelb",
        "static.prop.plantpot01",
        "static.prop.plantpot02",
        "static.prop.plantpot03",
        "static.prop.plantpot05",
        "static.prop.plantpot06",
        "static.prop.plantpot07",
        "static.prop.plantpot08",
        # 'static.prop.shoppingcart', # TODO add spawn offset
        # 'static.prop.shoppingtrolley',
        "static.prop.briefcase",
        "static.prop.guitarcase",
        # 'static.prop.travelcase', # offset in Z
        "static.prop.purse",
        "static.prop.maptable",
        "static.prop.advertisement",
        "static.prop.streetsign",
        "static.prop.streetsign01",
        "static.prop.streetsign04",
        "static.prop.atm",
        "static.prop.mailbox",
        "static.prop.vendingmachine",
    ]

    def __init__(
        self, sensors, min_distance=6, max_distance=10, lr_offset=[-1, +1], **kwargs
    ):
        """
        :param delay: delay in ticks to first spawn
        :param filename: filename to store data
        :param existing_duration: how long the object should exist
        :param cool_down: how long to wait before spawning again
        :param distance: range if 2 values, fixed if one value
        :param lr_offset: left right offset, range if 2 values, fixed if one value
        """
        super().__init__(sensors=sensors, **kwargs)

        self.min_distance = min_distance
        self.max_distance = max_distance
        self.lr_offset = lr_offset

        self.is_spawned = False
        self.static_actor = None
        self.data = []

    def _new_location(self, loc, ori, sim):
        distance = self.rng.uniform(self.min_distance, self.max_distance)

        # get speed of ego vehicle
        speed = sim.ego_vehicle.get_velocity().length()
        # speed in is in m/s
        # add speed to distance
        distance += speed / 2

        # get lr_offset
        if len(self.lr_offset) == 1:
            lr_offset = self.lr_offset[0]
        else:
            # use self.rng for random lr_offset
            lr_offset = self.rng.uniform(self.lr_offset[0], self.lr_offset[1])

        x, y, z = loc.x, loc.y, loc.z

        # Convert yaw to radians
        yaw_rad = math.radians(ori.yaw)

        # Calculate forward vector
        forward_x = math.cos(yaw_rad)
        forward_y = math.sin(yaw_rad)

        # lr_offset
        right_x = math.cos(yaw_rad + math.pi / 2)
        right_y = math.sin(yaw_rad + math.pi / 2)

        # Calculate offset (4 meters in front)
        offset_x = forward_x * distance + right_x * lr_offset
        offset_y = forward_y * distance + right_y * lr_offset

        # Calculate new spawn point
        new_x = x + offset_x
        new_y = y + offset_y
        new_z = z  # Assuming the height remains the same

        new_location = carla.Location(x=new_x, y=new_y, z=new_z)

        return new_location

    def _check_location_is_street(self, location, sim):
        # check if location is on street
        # get all waypoints
        map = sim.world.get_map()
        waypoints = map.get_waypoint(location, project_to_road=False)
        if waypoints is None:
            return False
        return True

    def _spawn(self, sim):
        # get location of ego vehicle
        loc = sim.ego_vehicle.get_location()

        # get orientation of ego vehicle
        ori = sim.ego_vehicle.get_transform().rotation

        # set spawnpoint n front of ego vehicle
        new_location = self._new_location(loc, ori, sim)
        # check if loc is on street
        if not self._check_location_is_street(new_location, sim):
            self.log.info(f"Tick: {sim.tick_count} |Location is not on street")
            return

        # create spawnpoint
        spawn = carla.Transform(new_location, ori)

        # filter blueprints for static objects
        static_bps = sim.world.get_blueprint_library().filter("static")
        # filter only ids in self.get_good_props()
        static_bps = [bp for bp in static_bps if bp.id in SpawnMiscCallback.spawnable_props]

        # choose random static object
        # self.log.info(f"Len of available static props: {len(static_bps)}")
        static_bp = self.rng.choice(static_bps)
        # static_bp=static_bps[0]

        # spawn static object
        self.static_actor = sim.world.try_spawn_actor(static_bp, spawn)
        # check if actor is spawned#
        if self.static_actor is not None:
            # write in log
            self.log.info(f"Spawned actor {self.static_actor.id} with blueprint {static_bp.id} at {spawn}")
            self.is_spawned = True
            self.start_tick = sim.tick_count
            # sim.anomaly_data.append({"id": self.static_actor.id})

        else:
            self.log.info("Could not spawn actor")

    def on_simulation_tick(self, sim, **kwargs):
        if self.is_spawned:
            self.register_anomaly_tick(sim, anomaly_obj_ids=[self.static_actor.id])

            # check if duration is over
            if self.scheduler.is_not_scheduled_for(sim.tick_count):
                # destroy actor
                success_full = self.static_actor.destroy()
                self.log.info(
                    f"Tick: {sim.tick_count} Destroying actor {self.static_actor.id} | {success_full=}"
                )
                self.is_spawned = False

        else:
            self.register_anomaly_tick(sim, anomaly_obj_ids=None, anomaly_class_ids=None)

            # check if existing frames are set
            if self.scheduler.is_scheduled_for(sim.tick_count):
                self.log.info(f"Tick: {sim.tick_count} Try to Spawn a Prop")
                self._spawn(sim)





class RunningPedestrianCallback(SampleLevelAnomalousCallback):
    """
    Drastically increases movement speed for some pedestrian
    """

    def __init__(self,  sensors=["Instance Segmentation"], speed=[5, 7], max_d: int = 50, **kwargs):
        super().__init__(sensors)
        self.speed = speed
        self.max_d = max_d
        self.running_actors = []

    def make_walker_run(self, sim, speed):
        """
        Select a random pedestrian and set its speed
        """
        possible_runners = []

        walkers = sim.world.get_actors().filter("controller.ai.walker")
        for walker in walkers:
            # get only walker near the ego vehicle
            if walker.get_location().distance(sim.ego_vehicle.get_location()) < self.max_d:
                # get walkers in front of ego vehicle
                if check_if_is_in_front_of_ego_vehicle(sim, walker):
                    possible_runners.append(walker)

        if len(possible_runners) == 0:
            self.log.info(f"No walker in front of ego vehicle")
            return None

        speed = self.rng.uniform(speed[0], speed[1])
        walker = self.rng.choice(possible_runners)
        self.log.info(f"Creating running walker {walker.id} with speed {speed}")
        walker.set_max_speed(speed)

        # remove walker from possible runners
        possible_runners.remove(walker)
        self.running_actors.append(walker)

        # if more than 3 walkers are in front of the ego vehicle
        # if len(possible_runners) > 2:
        #     if self.rng.random() < 0.5:
        #         walker = self.rng.choice(possible_runners)
        #         walker.set_max_speed(self.rng.uniform(speed[0], speed[-1]))
        #        self.running_actors.append(walker)

        return walker

    def reset_walker_speed(self, sim):
        walkers = sim.world.get_actors().filter("controller.ai.walker")
        for walker in walkers:
            speed = 1.4 + self.rng.uniform(-0.4, 0.4)
            walker.set_max_speed(speed)

    def on_simulation_tick(self, sim, measurements, **kwargs):
        # the "parent" here is important to select the pedestrian instead of the walker controler
        anomaly_obj_ids = [actor.parent.id for actor in self.running_actors]
        self.register_anomaly_tick(sim, anomaly_obj_ids=anomaly_obj_ids)

        # check if an actor is running
        if len(self.running_actors) > 0:
            # check if duration is over
            if self.scheduler.is_not_scheduled_for(sim.tick_count):
                self.reset_walker_speed(sim)
                self.log.info(f"Tick: {sim.tick_count} Stop Running Pedestrian")
                self.running_actors = []
        else: # no running walker
            if self.scheduler(sim.tick_count):
                # create a running pedestrian
                walker = self.make_walker_run(sim, self.speed)
                if walker:
                    self.log.warning(f"Tick: {sim.tick_count} start running pedestrian with {walker.id} [{walker.id % 2**16}]")
                else:
                    self.log.warning(f"Tick: {sim.tick_count} could not start running pedestrian")
                    self.scheduler.set_not_possible(sim.tick_count)
            else:
                self.reset_walker_speed(sim)


class BackwardsDriverCallback(AnomalousCallback):
    """

    """

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs
        )

        self.target_actor = None

    def on_simulation_tick(
            self, sim: "Simulation", measurements, **kwargs
    ):
        if not self.target_actor:
            # auto update all lights of vehicles
            vehicles_list = sim.world.get_actors().filter("*vehicle*")

            possible_targets = []
            for v in vehicles_list:
                # get only walker near the ego vehicle
                if v.get_location().distance(sim.ego_vehicle.get_location()) < 10:
                    # get walkers in front of ego vehicle
                    if check_if_is_in_front_of_ego_vehicle(sim, v):
                        possible_targets.append(v)

            if len(possible_targets) == 0:
                self.log.info(f"No possible target in front of ego vehicle")
                return False
            # one random walker run
            self.target_actor = self.rng.choice(possible_targets)

        control = carla.VehicleControl(
            1.0, # throttle
            0.0, # steer
            0.0, # brake
            False, # hand_brake
            True, # reverse
            False, # manual_gear_shift
            0 # gear
        )

        self.target_actor.apply_control(control)
        self.log.info(f"Tick: {sim.tick_count} Setting Backwards Driver Control")
        self.add_to_file(sim, actor=self.target_actor)


class UnknownObjectsCallback(SampleLevelAnomalousCallback):
    """
    Treat objects of certain classes as unknown
    """
    def __init__(self, unknown_objects_classes, **kwargs):
        super().__init__(**kwargs)
        self.unknown_objects_classes = unknown_objects_classes


    def on_simulation_tick(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        # TODO: we have to check if these classes are present in the image
        self.register_anomaly_tick(
            sim,
            anomaly_obj_ids=[],
            anomaly_class_ids=self.unknown_objects_classes
        )



class SteerDriverCallback(SampleLevelAnomalousCallback):
    """

    """
    def __init__(self, sensors, direction = None, **kwargs):
        super().__init__(sensors=sensors, **kwargs)

        self.target_actor = None
        self.direction = direction

    def apply_control(self, actor):
        direction = self.direction or np.random.choice(["left", "right"])

        if direction == "left":
            direction = 0.1
        else:
            direction = -0.1

        control = carla.VehicleControl(
            0.8,  # throttle
            direction,  # steer
            0.0,  # brake
            False,  # hand_brake
            False,  # reverse
            False,  # manual_gear_shift
            0  # gear
        )

        actor.set_autopilot(enabled=False)
        actor.apply_control(control)


    def select_random_driver(self, sim):
        # auto update all lights of vehicles
        vehicles_list = sim.world.get_actors().filter("*vehicle*")

        possible_targets = []
        for v in vehicles_list:
            # get only walker near the ego vehicle
            if v.get_location().distance(sim.ego_vehicle.get_location()) < 10:
                # get walkers in front of ego vehicle
                if check_if_is_in_front_of_ego_vehicle(sim, v):
                    possible_targets.append(v)

        if len(possible_targets) == 0:
            self.log.info(f"No possible target in front of ego vehicle")
            return None

        target_actor = self.rng.choice(possible_targets)

        return target_actor

    def on_simulation_tick(
            self, sim: "Simulation", measurements, **kwargs
    ):

        anomaly_obj_ids = [self.target_actor.id] if self.target_actor else None
        self.register_anomaly_tick(sim, anomaly_obj_ids=anomaly_obj_ids)

        if self.scheduler.is_scheduled_for(sim.tick_count):
            if not self.target_actor:
                self.target_actor = self.select_random_driver(sim)

                if self.target_actor:
                    self.apply_control(self.target_actor)

            if self.target_actor:
                self.log.info(f"[{self.target_actor.id}] {self.target_actor.get_control()}")

        else:
            if self.scheduler.is_not_scheduled_for(sim.tick_count):
                if self.target_actor:
                    self.target_actor.set_autopilot(enabled=True)
                    self.target_actor = None



class FlickerStreetLightCallback(SampleLevelAnomalousCallback):
    """
    Randomly flicker streetlights during anomaly,
    then restore daytime-based default afterward.

    We assume that this will only be used during nighttime when "on" lights are normal.
    """
    def __init__(self, seed=0, flicker_p=0.33, **kwargs):
        super().__init__(cb_seed=seed, **kwargs)
        self.previous_state_is_on = None
        self.changed_last_tick = False
        self.rng = random.Random(seed)
        self.flicker_p = flicker_p

        self.ano_state_buffer = [False, False]

    def _default_from_weather(self, sim):
        weather = sim.world.get_weather()
        return weather.sun_altitude_angle < 0.0  # night if sun below horizon


    def on_simulation_tick(self, sim: "Simulation", **kwargs):
        """
        during the scheduled intervals, we switch on/off the lights with a certain probability.
        this change, however, will only register 2 (!!!) timesteps later, so the current
        timestep will be an anomaly iff, two timesteps before, we changed the lights to off.
        """
        self.register_observation_anomaly_tick(
            sim,
            anomaly=self.ano_state_buffer[sim.tick_count],
            meta={"type": self.__class__.__name__})

        if self.changed_last_tick:
            self.changed_last_tick = False

        lmanager = sim.world.get_lightmanager()
        my_lights = lmanager.get_all_lights(light_group=carla.LightGroup.Street)
        if not my_lights:
            self.log.error("No lights found")
            return

        if self.scheduler.is_scheduled_for(sim.tick_count):
            # 50% chance to toggle this tick
            if self.rng.random() < self.flicker_p:
                if self.previous_state_is_on:
                    self.log.info(f"Tick: {sim.tick_count} Turning lights off (scheduled)")
                    lmanager.turn_off(my_lights)
                    self.previous_state_is_on = False

                    # lights turned off, this is an anomaly, assuming that this will only be
                    # used during nights, when lights should be on
                    self.ano_state_buffer.append(True)
                else:
                    self.log.info(f"Tick: {sim.tick_count} Turning lights on (scheduled)")
                    lmanager.turn_on(my_lights)
                    self.previous_state_is_on = True

                    # lights are on, this is normal
                    self.ano_state_buffer.append(False)

                self.changed_last_tick = True
            else:
                self.log.info(f"Tick: {sim.tick_count} not flipping (scheduled), previous is {self.previous_state_is_on}")
                self.ano_state_buffer.append(not self.previous_state_is_on)
        else:
            # restore to weather default
            desired_on = self._default_from_weather(sim)
            if desired_on and not self.previous_state_is_on:
                lmanager.turn_on(my_lights)
                self.previous_state_is_on = True
                self.changed_last_tick = True
                self.log.info(f"Tick: {sim.tick_count} schedule ended, restoring to on")
                self.ano_state_buffer.append(False)
            elif (not desired_on) and self.previous_state_is_on:
                lmanager.turn_off(my_lights)
                self.previous_state_is_on = False
                self.changed_last_tick = True
                self.log.info(f"Tick: {sim.tick_count} schedule ended, restoring to off")
                self.ano_state_buffer.append(True)
            else:
                # should be in desired state
                self.log.info(f"Tick: {sim.tick_count} not scheduled, state should be fine")
                self.ano_state_buffer.append(False)

        self.log.info(f"[{sim.tick_count}] -> {len(self.ano_state_buffer)}")
        assert len(self.ano_state_buffer) == sim.tick_count + 3


class FlickerTrafficLightCallback(SampleLevelAnomalousCallback):
    """
    Cause unusual traffic light behavior
    """

    def __init__(
        self,
        sensors,
        **kwargs,
    ):
        super().__init__(sensors, **kwargs)
        self.previous_state_is_on = None
        self.changed_last_tick = False
        self.tf = None

    def switching_traffic_light(self, sim, closest_tl):

        # state = self.rng.choice([carla.TrafficLightState.Green, carla.TrafficLightState.Red, carla.TrafficLightState.Yellow])
        # closest_tl.set_state(state)

        if closest_tl.get_state() == carla.TrafficLightState.Green:
            self.log.info(f"{sim.tick_count} Switching traffic light state")
            closest_tl.set_state(carla.TrafficLightState.Red)
            self.changed_last_tick = True
        elif closest_tl.get_state() == carla.TrafficLightState.Red:
            self.log.info(f"{sim.tick_count} Switching traffic light state")
            closest_tl.set_state(carla.TrafficLightState.Green)
            self.changed_last_tick = True
        else:
            self.log.info(f"Traffic light is not red or green")

    def on_simulation_tick(self, sim: "Simulation", **kwargs):

        anomaly_obj_ids = [self.tf.id] if self.tf is not None else []
        self.register_anomaly_tick(sim, anomaly_obj_ids=anomaly_obj_ids)

        self.changed_last_tick = False

        if self.scheduler.is_scheduled_for(sim.tick_count):
            # find next tf
            tf = find_next_tf(
                sim.ego_vehicle,
                sim,
                self.log,
                max_cars_in_front=10,
                max_distance=20
            )
            if tf is None:
                self.scheduler.set_not_possible(sim.tick_count)
                return
            else:
                # case tf is existing
                self.tf = tf

                if random.random() < 0.3:
                    self.switching_traffic_light(sim, tf)

        # check if first not existing frame
        if self.scheduler.is_not_scheduled_anymore(sim.tick_count):
            # reset group
            self.tf.reset_group()
            self.tf = None


class YellowBlinkingTrafficLightsCallback(FlickerTrafficLightCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs,)
        self.changed_last_tick = False

    def switching_traffic_light(self, sim, closest_tl):

        # if tl is off --> set to yellow
        if closest_tl.get_state() == carla.TrafficLightState.Off:
            closest_tl.set_state(carla.TrafficLightState.Yellow)
        # if tl is yellow --> set to off
        else:
            closest_tl.set_state(carla.TrafficLightState.Off)
        self.changed_last_tick = True


class OffTrafficLightCallback(FlickerTrafficLightCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.changed_last_tick = False

    def switching_traffic_light(self, sim, closest_tl):
        self.changed_last_tick = True
        # set closest tl off
        closest_tl.set_state(carla.TrafficLightState.Off)

class SpawnActorCallback(SampleLevelAnomalousCallback):
    """
    Spawn some actors and treat them as anomalous for a couple of timesteps

    """

    # TODO: make configurable
    broken_blueprints = [
        "vehicle.mitsubishi.fusorosa",
        "vehicle.carlamotors.firetruck"
    ]

    def __init__(
        self,
        filter="*vehicle*",
        min_distance=5,
        max_distance=30,
        mark_for=5,
        **kwargs,
    ):
        """
        :param mark_for: for how long newly spawned objects will be marked as anomalies
        """
        super().__init__(**kwargs)
        self.filter = filter
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.mark_for = mark_for

        self.new_actor = None
        self.last_frame_spawned = False

        self.logbook: Dict[int, List[carla.Actor]] = {0: []}


    def on_simulation_tick(self, sim, measurements, **kwargs):
        tick = sim.tick_count
        self.register_anomaly_tick(sim, anomaly_obj_ids=self.logbook[tick])

        for i in range(1, self.mark_for + 1):
            if tick + i not in self.logbook:
                self.logbook[tick + i] = []

        if self.scheduler.is_scheduled_for(tick):
            actor = self._spawn(sim)
            if actor is not None:
                # keep as anomaly for 3 timesteps
                for i in range(1, self.mark_for + 1):
                    self.logbook[tick + i].append(actor.id)
            else:
                self.scheduler.set_not_possible(sim.tick_count)

    def _spawn(self, sim) -> carla.Actor:
        loc = sim.ego_vehicle.get_location()
        self._blueprints: List[carla.ActorBlueprint] = (
            sim.world.get_blueprint_library().filter(self.filter)
        )

        if SpawnActorCallback.broken_blueprints:
            # remove broken cars
            self._blueprints = [
                x for x in self._blueprints if x.id not in SpawnActorCallback.broken_blueprints
            ]
            self.log.debug(f"Ignoring {len(SpawnActorCallback.broken_blueprints)} blueprints")
            for broken in SpawnActorCallback.broken_blueprints:
                self.log.info(f"Broken car: {broken}")

        bp = self.rng.choice(self._blueprints)
        for spawn in sim.world.get_map().get_spawn_points():
            d = spawn.location.distance(sim.ego_vehicle.get_transform().location)
            if (
                self.max_distance >= d >= self.min_distance
                    and (
                    check_if_is_in_front_of(
                        sim.ego_vehicle.get_location(),
                        sim.ego_vehicle.get_transform().get_forward_vector(),
                        spawn.location,
                    )
                    and angle_between_vectors_2d(
                        sim.ego_vehicle.get_transform().get_forward_vector(),
                        spawn.location - sim.ego_vehicle.get_location(),
                    )
                    < 60
                )
            ):

                npc = sim.world.try_spawn_actor(bp, spawn)
                if npc is not None:
                    self.log.info(f"Spawning actor {bp} at {loc}")
                    # actor will display after next tick
                    if "vehicle" in self.filter:
                        npc.set_autopilot(True)

                    return npc

        return None


class VanishActorCallback(SampleLevelAnomalousCallback):
    """
    Randomly destroy some actor. This only causes anomalies on the observation level.
    """

    def __init__(self, filter="*vehicle*", min_distance=0, max_distance=50, **kwargs):
        super().__init__( **kwargs)
        self.filter = filter
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.destroyed_actor = None
        self.last_frame_despawned = False
        self.meta = None

    def on_simulation_tick(self, sim, measurements, **kwargs):
        self.register_observation_anomaly_tick(sim, self.last_frame_despawned, meta=self.meta)

        if self.last_frame_despawned:
            self.last_frame_despawned = False
            self.meta = None

        if self.scheduler.is_scheduled_for(sim.tick_count):
            # try to despawn actor
            actor = self._despawn(sim)
            if actor is not None:
                self.last_frame_despawned = True
                self.meta = {"actor": int(actor.id)}
            else:
                self.scheduler.set_not_possible(sim.tick_count)
                self.meta = None

    def _despawn(self, sim):
        # find actors in front of ego vehicle
        actors = sim.world.get_actors().filter(self.filter)
        possible_despawns = []
        for actor in actors:
            if actor.id == sim.ego_vehicle.id:
                continue
            if (
                self.min_distance
                < actor.get_location().distance(sim.ego_vehicle.get_location())
                < self.max_distance
            ):
                # check if is in front of ego vehicle
                if check_if_is_in_front_of_ego_vehicle(sim, actor):
                    # check if ist not on the side
                    if (
                        angle_between_vectors_2d(
                            sim.ego_vehicle.get_transform().get_forward_vector(),
                            actor.get_location() - sim.ego_vehicle.get_location(),
                        )
                        < 60
                    ):
                        possible_despawns.append(actor)

        if not possible_despawns:
            self.log.info(f"No actor to despawn")
            return None

        # despawn random actor
        actor = self.rng.choice(possible_despawns)
        actor.destroy()
        self.destroyed_actor = actor
        self.log.info(f"Despawning actor {actor.id}")
        return actor


class InstantWeatherChangeCallback(SampleLevelAnomalousCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_tick_changed = False
        self.change_information = None

    def on_simulation_tick(self, sim: "Simulation", **kwargs):
        self.register_observation_anomaly_tick(sim, self.last_tick_changed, meta=self.change_information)
        self.last_tick_changed = False

        if self.scheduler.is_scheduled_for(sim.tick_count):
            self._change_weather(sim)

    @staticmethod
    def _convert_weather_to_dict(weather):
        return {
            "cloudiness": weather.cloudiness,
            "precipitation": weather.precipitation,
            "precipitation_deposits": weather.precipitation_deposits,
            "sun_azimuth_angle": weather.sun_azimuth_angle,
            "sun_altitude_angle": weather.sun_altitude_angle,
            "fog_density": weather.fog_density,
            "fog_distance": weather.fog_distance,
            "fog_falloff": weather.fog_falloff,
            "wind_intensity": weather.wind_intensity,
            "wetness": weather.wetness,
            "scattering_intensity": weather.scattering_intensity,
            "mie_scattering_scale": weather.mie_scattering_scale,
            "rayleigh_scattering_scale": weather.rayleigh_scattering_scale,
            "dust_storm": weather.dust_storm,
        }

    def _change_weather(self, sim):
        # check if first existing frame
        if self.scheduler.is_first_scheduled_for(sim.tick_count):
            self.start_changing_tick = sim.tick_count
            self.current_duration = self.scheduler.get_current_event_duration(
                sim.tick_count
            )
            # start weather
            self.start_weather = self._convert_weather_to_dict(sim.world.get_weather())

            self.destination_weather = self.next_weather(self.start_weather)

            # convert weather to dict
            # self.destination_weather = self.destination_weather.__dict__
            self.log.info(f"Start weather: {self.start_weather}")
            self.log.info(f"Destination weather: {self.destination_weather}")

            self._transition = {}
            # get intermediate weather steps
            for attribute in self.destination_weather:
                v0 = self.start_weather[attribute]
                vn = self.destination_weather[attribute]

                self._transition[attribute] = list(
                    np.linspace(v0, vn, self.current_duration + 1)
                )
            self.log.info(f"Weather transition: {self._transition}")

        # +1 because we start with the second element, while first element is the current state
        next_tick = sim.tick_count - self.start_changing_tick + 1
        next_weather = {a: self._transition[a][next_tick] for a in self._transition}
        # sim.world.set_weather(next_weather)
        set_weather(sim.world, next_weather)

        self.log.info(f"Set weather to {next_weather}")

        self.last_tick_changed = True

    @staticmethod
    def detect_weather_type(current_weather: Dict) -> str:
        if (
            current_weather["precipitation"] > 40
            and current_weather["wind_intensity"] > 40
        ):
            return "rainy"
        elif (
            current_weather["fog_density"] > 70
            and current_weather["wind_intensity"] == 0
        ):
            return "dense_fog"
        elif (
            current_weather["fog_density"] > 20
            and current_weather["precipitation"] > 20
        ):
            return "light_fog_rain"
        else:
            return "clear"

    def next_weather(self, current_weather: Dict) -> Dict:
        # get sun position
        sun_azimuth = current_weather["sun_azimuth_angle"]
        sun_altitude = current_weather["sun_altitude_angle"]

        # get current weather type
        current_type = self.detect_weather_type(current_weather)
        possible_types = ["rainy", "dense_fog", "clear"]  # "light_fog_rain",

        if current_type in possible_types:
            possible_types.remove(current_type)
        else:
            self.log.warning(f"Current weather {current_type} not in possible types: {possible_types}")

        weather_type = self.rng.choice(possible_types)

        self.change_information = {"description": f"{current_type} -> {weather_type}"}

        # default values
        new_weather = {
            "cloudiness": 0,
            "precipitation": 0,
            "precipitation_deposits": 0,
            "wind_intensity": 0,
            "sun_azimuth_angle": sun_azimuth,
            "sun_altitude_angle": sun_altitude,
            "fog_density": 0,
            "fog_distance": 0,
            "wetness": 0,
            "fog_falloff": 0,
            "scattering_intensity": 0,
            "mie_scattering_scale": 0.03,
            "rayleigh_scattering_scale": 0.033100,
        }

        if weather_type == "rainy":
            new_weather["precipitation"] = 50
            new_weather["precipitation_deposits"] = 50
            new_weather["wind_intensity"] = 50
            new_weather["cloudiness"] = 30
            new_weather["wetness"] = 50
            new_weather["dust_storm"] = 30
        elif weather_type == "dense_fog":
            new_weather["fog_density"] = 100
            new_weather["fog_distance"] = 3

        return new_weather



# class ErraticDriverCallback(SampleLevelAnomalousCallback):
#     """ """
#
#     def __init__(
#         self,
#         percentage_speed=-1000,
#         distance_to_leading=0.5,
#         ignore_lights_percentage=100,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.distance_to_leading = distance_to_leading
#         self.percentage_speed = percentage_speed
#         self.ignore_lights_percentage = ignore_lights_percentage
#
#     def on_simulation_start(self, sim: "Simulation", **kwargs):
#         drivers = sim.world.get_actors().filter("*vehicle*")
#
#         for driver in drivers:
#             if driver.id == sim.ego_vehicle.id:
#                 continue
#
#             physics_control = driver.get_physics_control()
#
#             # physics_control.torque_curve = [carla.Vector2D(x=0, y=400), carla.Vector2D(x=1300, y=600)]
#             # physics_control.max_rpm = 10000
#             # physics_control.moi = 1.0
#             # physics_control.damping_rate_full_throttle = 0.0
#             # physics_control.use_gear_autobox = True
#             # physics_control.gear_switch_time = 0.5
#             # physics_control.clutch_strength = 10
#             physics_control.mass = 1
#             # physics_control.drag_coefficient = 0.25
#             # physics_control.steering_curve = [carla.Vector2D(x=0, y=1), carla.Vector2D(x=100, y=1),
#             #                                   carla.Vector2D(x=300, y=1)]
#             # physics_control.use_sweep_wheel_collision = True
#             # physics_control.wheels = wheels
#             driver.apply_physics_control(physics_control)
#
#             sim.tm.ignore_lights_percentage(driver, self.ignore_lights_percentage)
#
#             sim.tm.distance_to_leading_vehicle(driver, int(self.distance_to_leading))
#             sim.tm.vehicle_percentage_speed_difference(
#                 driver, int(self.percentage_speed)
#             )
#
#             sim.tm.set_desired_speed(driver, 300)
#
#     def on_simulation_tick(self, sim, measurements, **kwargs):
#
#
#         possible_drivers = []
#
#         drivers = sim.world.get_actors().filter("*vehicle*")
#         for driver in drivers:
#             if driver.id == sim.ego_vehicle.id:
#                 continue
#
#             sim.tm.set_desired_speed(driver, 300)


# class StreetLightColor(Callback):
#     """
#     Changes color of streetlights
#     """

#     def __init__(self, color=(0, 255, 0),**kwargs):
#         super().__init__(**kwargs)
#         self.color = color

#     def on_simulation_start(self, sim: "Simulation", **kwargs):
#         lmanager = sim.world.get_lightmanager()

#         my_lights = lmanager.get_all_lights(light_group=carla.LightGroup.Street)
#         self.log.info(f"Setting color of {len(my_lights)} lights")
#         lmanager.set_color(my_lights, carla.Color(*self.color, 255))

#         # for some reason we have to turn lights on again
#         lmanager.turn_on(my_lights)

#     def on_simulation_tick(self, sim: "Simulation", **kwargs):
#         lmanager = sim.world.get_lightmanager()

#         my_lights = lmanager.get_all_lights(light_group=carla.LightGroup.Street)

#         for light in my_lights:
#             register_anomalous_actor(sim, light)


# class AnomalousChangeWeatherCallback(Callback):
#     """
#     Interpolate from current weather to given weather over some period
#     """

#     def __init__(self, weather_states=None,weather_state_names =None, random_rate=0.5, cooldown_time=50,duration=1,**kwargs):
#         super().__init__(**kwargs)
#         self.start_changing_tick = 0
#         if duration>30:
#             raise Exception(f"The duration of {duration} is to long, please use a maximum of 30 instead or use ChangeWeatherCallback for not anomalous weather change!")
#         self.duration = duration
#         if weather_states==None or len(weather_states)<2:
#             self.log.info("Not enough different Weather states are given. Some more will be created")
#             self.weather_states=self._init_weather_states()
#         else:
#             self.weather_states =weather_states
#         self.target_weather = None
#         self.weather_is_changing_start=False
#         self._start_weather = None
#         self.last_picked_weather="None"
#         self._transition = {}
#         self.random_rate=random_rate
#         self.cooldown_time=cooldown_time
#         self.myAnomalousActor=AnomalousActor("instantWetherChanging","instandChanging.csv")
#     def _init_weather_states(self):
#         weathers={}
#         weathers.update({"good_night":{
#             # this is night
#             "cloudiness": 5.0,
#             "precipitation": 0.0,
#             "sun_altitude_angle": 0.0,
#             "sun_azimuth_angle": -1.0,
#             "fog_density": 2.0,
#             "fog_distance": 0.75,
#             "fog_falloff": 0.1,
#             "precipitation_deposits": 0.0,
#             "wind_intensity": 10.0,
#             "wetness": 0.0,
#             "scattering_intensity": 1.0,
#             "mie_scattering_scale": 0.03,
#             "rayleigh_scattering_scale": 0.033100,
#             "dust_storm": 0.0
#                     }
#                          })
#         weathers.update({"good_day":{
#             # this is good weather
#             "cloudiness": 5.0,
#             "precipitation": 0.0,
#             "sun_altitude_angle": 45.0,
#             "sun_azimuth_angle": -1.0,
#             "fog_density": 2.0,
#             "fog_distance": 0.75,
#             "fog_falloff": 0.1,
#             "precipitation_deposits": 0.0,
#             "wind_intensity": 10.0,
#             "wetness": 0.0,
#             "scattering_intensity": 1.0,
#             "mie_scattering_scale": 0.03,
#             "rayleigh_scattering_scale": 0.033100,
#             "dust_storm": 0.0
#                     }
#                          })
#         weathers.update({"bad_day":{
#             # this is bad weather
#             "cloudiness": 5.0,
#             "precipitation": 10.0,
#             "sun_altitude_angle": 45.0,
#             "sun_azimuth_angle": -1.0,
#             "fog_density": 1000.0,
#             "fog_distance": 0.0,
#             "fog_falloff": 0.0,
#             "precipitation_deposits": 0.0,
#             "wind_intensity": 40.0,
#             "wetness": 100.0,
#             "scattering_intensity": 1.0,
#             "mie_scattering_scale": 0.03,
#             "rayleigh_scattering_scale": 0.033100,
#             "dust_storm": 10.0
#                     }
#                          })
#         weathers.update({"bad_night":{
#             # this is bad weather
#             "cloudiness": 5.0,
#             "precipitation": 10.0,
#             "sun_altitude_angle": 0.0,
#             "sun_azimuth_angle": -1.0,
#             "fog_density": 1000.0,
#             "fog_distance": 0.0,
#             "fog_falloff": 0.0,
#             "precipitation_deposits": 0.0,
#             "wind_intensity": 40.0,
#             "wetness": 100.0,
#             "scattering_intensity": 1.0,
#             "mie_scattering_scale": 0.03,
#             "rayleigh_scattering_scale": 0.033100,
#             "dust_storm": 10.0
#                     }
#             })
#         return weathers

#     def _change_weather(self,sim,pre=False):
#             if self.weather_is_changing_start:
#                 self.start_changing_tick=sim.tick_count
#                 self._start_weather = sim.world.get_weather()

#                 for attribute in self.target_weather:
#                     v0 = getattr(self._start_weather, attribute)
#                     vn = self.target_weather[attribute]

#                     self._transition[attribute] = list(
#                         np.linspace(v0, vn, self.duration)
#                     )

#                 self.log.debug(f"Weather transition: {self._transition}")
#                 self.weather_is_changing_start=False

#             current = sim.tick_count - self.start_changing_tick
#             current_weather = {
#                 a: self._transition[a][current] for a in self._transition
#             }
#             self.log.info(
#                 f"Weather transition: {current+1}/{self.duration} ({current / self.duration:.2%}) {self.last_picked_weather}"
#             )
#             set_weather(sim.world, current_weather)
#             if not pre:
#                 self.myAnomalousActor.add_anomalous_behaviour(sim,sim.tick_count+1,{"instantWeatherChange": self.last_picked_weather})

#     def on_simulation_start(self, sim: "Simulation", **kwargs):
#         self.last_picked_weather=list(self.weather_states.keys())[0]
#         self.target_weather=self.weather_states[self.last_picked_weather]
#         self.weather_is_changing_start=True
#         self._change_weather(sim,pre=True)
#     def on_simulation_tick(self, sim: "Simulation", **kwargs):
#         if sim.tick_count - self.start_changing_tick < self.duration:
#             self._change_weather(sim)
#         else:
#             # self.log.info(f"last {self.myAnomalousActor.last_data_frame_number()} ")
#             if self.myAnomalousActor.last_data_frame_number()+self.cooldown_time <= sim.tick_count and np.random.random()<self.random_rate :
#                 # select new target weather
#                 weather_number = np.random.choice(len(self.weather_states))
#                 # make sure that we change the weather
#                 # self.log.info(f"NEW picked: {list(self.weather_states.keys())[weather_number]}")
#                 if list(self.weather_states.keys())[weather_number]==self.last_picked_weather:
#                     weather_number=(weather_number+1)%len(self.weather_states)
#                 self.log.info(f"NEW picked: {list(self.weather_states.keys())[weather_number]}")

#                 self.target_weather=self.weather_states[list(self.weather_states.keys())[weather_number]]
#                 # self.log.info(self.target_weather)
#                 self.last_picked_weather=list(self.weather_states.keys())[weather_number]
#                 self.weather_is_changing_start=True
#                 self._change_weather(sim)

#     def on_simulation_end(self, sim: "Simulation", **kwargs):
#         self.myAnomalousActor.postprocess(sim)


# class AnomalousActor():
#     anomaly_types=[
#         "spawnMiscActor"
#         "vanishActor",
#         "spawnActor",
#         "backwardsActor",
#         "unknownObject",
#         "unusualContext",
#         "trafficLightDisco",
#         "streetLightDisco",
#         "instantWetherChanging",
#         "cameraNoise"
#     ]
#     semantic_tags=[
#         'None',
#         'Roads',
#         'Sidewalks',
#         'Buildings',
#         'Walls',
#         'Fences',
#         'Poles',
#         'TrafficLight',
#         'TrafficSigns',
#         'Vegetation',
#         'Terrain',
#         'Sky',
#         'Pedestrians',
#         'Rider',
#         'Car',
#         'Truck',
#         'Bus',
#         'Train',
#         'Motorcycle',
#         'Bicycle',
#         'Static',
#         'Dynamic',
#         'Other',
#         'Water',
#         'RoadLines',
#         'Ground',
#         'Bridge',
#         'RailTrack',
#         'GuardRail',
#         'MyAnimals',
#         'MyCars',
#         'MyReverseCars',
#     ]

#     def __init__(
#         self,
#         anomaly_type:str,
#         filename:str,
#         update_step: int =1
#         ):
#         self.data=[]
#         if anomaly_type not in self.anomaly_types:
#             raise NotImplementedError(f"For {anomaly_type} the anomalous actor class is not implemented yet. Please you a type of {self.anomaly_types} !")
#         self.anomaly_type=anomaly_type
#         # if usage not in self.usages:
#         #     raise NotImplementedError(f"For {usage} the anomalous actor class is not implemented yet. Please you a type of {self.usages} !")

#         # self.usage=usage
#         self.update_step=update_step
#         self.filename=filename
#         self.log = logging.getLogger(type(self).__name__)


#     def __call__(self, sim, frame, npc, class_):
#         self.add_actor(  sim,frame, npc, class_)

#     # todo ggf sim und npc übergeben???
#     def add_actor(self, sim, frame, npc, class_):
#         self.data.append(
#             {"frame_number": frame, "id": npc.id,"class":class_, "type":npc.type_id,"attributes":npc.attributes}
#         )
#         if frame%self.update_step==0:
#             self.write_data(sim)
#     def add_anomalous_behaviour(self, sim, frame, information):
#         self.data.append(
#             {"frame_number": frame, "information": information}
#         )
#         if frame%self.update_step==0:
#             self.write_data(sim)
#     def len_data(self):
#         return len(self.data)
#     def last_data_frame_number(self):
#         if self.len_data()>0:
#             return self.data[-1]["frame_number"]
#         else:
#             return -1
#     def write_data(self,sim):
#         pd.DataFrame(self.data).to_csv(join(sim.root,self.filename),index=False)

#     def find_id_in_segm(self,sim,camera,frame,datum, paint):
#         boundary_value = 65536
#         if paint:
#             segm=f"{sim.root}/{camera}-anomalie"
#         else:
#             segm = join(sim.root, camera)

#         id=datum["id"]
#         class_=datum["class"]
#         id -= math.floor(id / boundary_value) * boundary_value
#         # create rgb color of id pixel
#         pixel = [class_, (id >> 0) % 256, (id >> 8) % 256]

#         # open
#         file = f"{segm}/{frame:06d}.png"
#         img_arr = np.array(Image.open(file))[:, :, :3]
#         if True in (np.array(pixel) == img_arr).all(2):
#             if paint:
#                 img_arr[(np.array(pixel) == img_arr).all(2)]=[255,255,255]
#                 Image.fromarray(img_arr).save(file)
#             return True


#     def  imageBased_check(self,sim,camera,frame,anomalous_entry):
#         if self.anomaly_type=="vanishActor":
#             # +1 because on the next frame the actor is killed
#             if frame == anomalous_entry["frame_number"]+1:
#                 return self. find_id_in_segm(
#                     sim,
#                     camera,
#                     anomalous_entry["frame_number"], # is -1 smaller the frame
#                     anomalous_entry,
#                     paint=False)
#         if self.anomaly_type=="spawnActor":
#             if frame == anomalous_entry["frame_number"]:
#                 return self. find_id_in_segm(
#                     sim,
#                     camera,
#                     anomalous_entry["frame_number"],
#                     anomalous_entry,
#                     paint=False)
#         if self.anomaly_type=="backwardsActor":
#             if frame == anomalous_entry["frame_number"]+1:
#                 return self.find_id_in_segm(
#                         sim,
#                         camera,
#                         frame, # current frame
#                         anomalous_entry,
#                         paint=True) # painte das Bild
#         if self.anomaly_type=="trafficLightDisco":
#             if frame == anomalous_entry["frame_number"]:
#                 return self.find_id_in_segm(
#                         sim,
#                         camera,
#                         frame, # current frame
#                         anomalous_entry,
#                         paint=True) # painte das Bild


#     def LidarBased_check(self,sim,frame,anomalous_entry):
#         if self.anomaly_type=="vanishActor":
#             # +1 because on the next frame the actor is killed
#             if frame == anomalous_entry["frame_number"]+1:
#                 return True
#         if self.anomaly_type=="spawnActor":
#             if frame == anomalous_entry["frame_number"]:
#                 return True
#         if self.anomaly_type=="backwardsActor":
#             # wird erst ein frame später deutlich
#             if frame == anomalous_entry["frame_number"]+1:
#                 return True
#         if self.anomaly_type=="trafficLightDisco":
#             if frame == anomalous_entry["frame_number"]:
#                 return True


#     def postprocess(self,sim,class_=None):
#         # find all segmentation folders
#         cameras={}
#         for elem in os.listdir(sim.root):
#             if "seg" in elem:
#                 cameras.update({elem: []})


#         if self.anomaly_type in ["trafficLightDisco","backwardsActor","unknownObject","unusualContext"]:
#             # copy to camera-anomaly folder
#             pass

#         if self.anomaly_type not in ["streetLightDisco","instantWetherChanging","cameraNoise"]:
#             ano_data_lidar=[]
#         for frame in range(0,sim.tick_count):
#             self.log.info(f"Frame: {frame} of {sim.tick_count}")
#             if frame%100==0:
#                 self.log.info(f"Frame: {frame} of {sim.tick_count}")
#             # with extra csv file databased
#             # camera no iterieren
#             informations_lidar=[]
#             for camera in cameras:
#                 informations_img=[]


#                 if self.anomaly_type in ["vanishActor","spawnActor","backwardsActor", "trafficLightDisco"]:
#                     for anomalous_entry in self.data:
#                         if self.imageBased_check(sim,camera,frame,anomalous_entry):
#                             informations_img.append(
#                             {
#                                 "id": anomalous_entry["id"],
#                                 "class":anomalous_entry["class"],
#                                 "classname":self.semantic_tags[anomalous_entry["class"]],
#                                 "type":anomalous_entry["type"],
#                                 "attributes":anomalous_entry["attributes"]
#                             }
#                             )


#                 elif self.anomaly_type in ["unknownObject","unusualContext"]:
#                     if class_ ==None:
#                         raise Exception("Please specify a class. None is not a class!!!")
#                     anomaly,ids= self.find_class_in_segm(sim,camera,frame,class_,paint=True)
#                     if anomaly:
#                         for id in ids:
#                             informations_img.append(
#                                 {
#                                     "id": id,
#                                     "class":class_,
#                                     "classname":self.semantic_tags[class_],
#                                     "type":self.anomaly_type,
#                                     "attributes":None # TODO ggf alle npc in der welt abspeichern und dann danach danach suchen oderso
#                                 }
#                                 )


#                 elif self.anomaly_type in  ["streetLightDisco","instantWetherChanging","cameraNoise"]:
#                     for anomalous_entry in self.data:
#                         if frame==anomalous_entry["frame_number"]+1:
#                             informations_img.append(
#                                     anomalous_entry["information"]
#                                     )
#                 elif self.anomaly_type=="cameraNoise":
#                     for anomalous_entry in self.data:
#                         if frame==anomalous_entry["frame_number"]:
#                             informations_img.append(
#                                     anomalous_entry["information"]
#                                     )

#                 # for this frame no anomaly data exists
#                 if len(informations_img)==0:
#                     cameras[camera].append(
#                         {"frame_number": frame, "anomaly":False, "anomaly_type":None, "informations": None}
#                     )
#                 # case anomaly exist
#                 else:
#                     cameras[camera].append(
#                         {"frame_number": frame, "anomaly":True, "anomaly_type":self.anomaly_type, "informations": informations_img}
#                     )
#                     # append information to lidar
#                     informations_lidar.append(informations_img)
#             ### Lidar
#             if self.anomaly_type not in ["streetLightDisco","instantWetherChanging","cameraNoise"]:
#                 # for this frame no anomaly data exists
#                 if len(informations_lidar)==0:
#                     ano_data_lidar.append(
#                         {"frame_number": frame, "anomaly":False, "anomaly_type":None, "informations": None}
#                     )
#                 # case anomaly exist
#                 else:
#                     ano_data_lidar.append(
#                         {"frame_number": frame, "anomaly":True, "anomaly_type":self.anomaly_type, "informations": informations_lidar}
#                     )
#             informations_lidar=[]

#         # ### LIDAR

#         # if self.anomaly_type in ["vanishActor","spawnActor","backwardsActor", "trafficLightDisco"]:
#         #     for anomalous_entry in self.data:
#         #         if self.LidarBased_check(sim,frame,anomalous_entry):
#         #         informations_lidar.append(
#         #         {
#         #             "id": anomalous_entry["id"],
#         #             "class":anomalous_entry["class"],
#         #             "classname":self.semantic_tags[anomalous_entry["class"]],
#         #             "type":anomalous_entry["type"],
#         #             "attributes":anomalous_entry["attributes"]
#         #         }
#         #         )

#         # elif self.anomaly_type in ["unknownObject","unusualContext"]:
#         #     if class_ ==None:
#         #         raise Exception("Please specify a class. None is not a class!!!")
#         #     anomaly,ids= self.find_class_in_segm(sim,camera,frame,class_,paint=True)
#         #     if anomaly:
#         #         for id in ids:
#         #             informations_lidar.append(
#         #                 {
#         #                     "id": id,
#         #                     "class":class_,
#         #                     "classname":self.semantic_tags[class_],
#         #                     "type":self.anomaly_type,
#         #                     "attributes":None # TODO ggf alle npc in der welt abspeichern und dann danach danach suchen oderso
#         #                 }
#         #                 )


#         self.log.info ("Anomalous Done")
#         for camera in cameras:
#             camera_perspektive=camera.replace("segmentation-","")
#             pd.DataFrame(cameras[camera]).to_csv(join(sim.root,f"anomaly_{camera_perspektive}_img.csv"),index=False)
#         if self.anomaly_type not in ["streetLightDisco","instantWetherChanging","cameraNoise"]:
#             pd.DataFrame(ano_data_lidar).to_csv(join(sim.root,"anomaly_lidar.csv"),index=False)
