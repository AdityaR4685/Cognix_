import errno
import logging
import os
import queue
import subprocess as sp
import time
from functools import lru_cache
from typing import List, Dict

import carla
import numpy as np
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from carlagen.callback.callback import Callback
from carlagen.utils.carla import (
    connect_client,
    apply_settings,
    configure_traffic_manager,
    create_vehicles,
    create_pedestrians,
    create_egovehicle,
    destroy_actors,
    build_projection_matrix,
    set_weather,
    teardown,
)

CARLA_PEDESTRIAN_CLASS = 12
CARLA_CAR_CLASS = 14
CARLA_BUS_CLASS = 15
CARLA_BUS2_CLASS = 16
CARLA_CLASS_POLE = 6
CARLA_CLASS_SKY = 11
CARLA_TRAFFIC_LIGHT_CLASS = 7
CARLA_TRAFFIC_SIGN_CLASS = 8
CARLA_MOTORBIKE_CLASS = 18
CARLA_BIKE_CLASS = 19
CARLA_STREET_STRIPES = 24


log = logging.getLogger(__name__)


class Sensor(object):
    """ """

    def __init__(
        self,
        sim: "Simulation",
        name,
        blueprint,
        attributes: Dict[str, str],
        location=(0, 0, 0),
        rotation=(0, 0, 0),
        every_n_frames=1,
    ):
        """

        :param sim:
        :param name:
        :param blueprint:
        :param attributes:
        :param location:
        :param rotation:
        """
        self.name = name
        self.every_n_frames = every_n_frames
        sensor_location = carla.Location(*location)
        sensor_rotation = carla.Rotation(*rotation)
        self.sensor_transform = carla.Transform(sensor_location, sensor_rotation)

        self.sensor_bp = sim.world.get_blueprint_library().find(blueprint)

        if attributes:
            for att, value in attributes.items():
                self.sensor_bp.set_attribute(str(att), str(value))

        self.queue = queue.Queue()

        if self.sensor_bp == "sensor.other.collision":
            self.actor = sim.world.spawn_actor(
                self.sensor_bp,
                carla.Transform(),
                attach_to=sim.ego_vehicle,
                # attachment_type=carla.AttachmentType.Rigid,
            )
        else:

            self.actor = sim.world.spawn_actor(
                self.sensor_bp,
                self.sensor_transform,
                attach_to=sim.ego_vehicle,
                attachment_type=carla.AttachmentType.Rigid,
            )

        self.actor.listen(self.queue.put)

    def get_measurement(self):
        # Collision sensors are event-driven; do not block.
        if self.sensor_bp.id == "sensor.other.collision":
            items = []
            try:
                while True:  # drain this tick
                    items.append(self.queue.get_nowait())
            except queue.Empty:
                pass
            return items  # [] if no events this tick

        # All other sensors: keep existing blocking behavior
        try:
            return self.queue.get(timeout=2)
        except queue.Empty:
            log.warning(f"Sensor {self.name} has no measurement")
            return None

    def clear(self):
        """
        Empty queue of this sensor
        :return:
        """
        i = 0

        while not self.queue.empty() and i < 10:
            self.queue.get(timeout=5)
            i += 1

        if i >= 10:
            log.error(f"Could not clear sensor {self.name}")


class Simulation(object):
    """ """

    def __init__(self, cfg, root):
        """
        :param cfg:
        :param root: where to store stuff
        """
        self.cfg = cfg
        self.root = root

        self.ego_vehicle = None
        self.world = None
        self.client = None
        self.tm = None  # traffic manager
        self.proc = None  # running carla process
        self.callbacks: List[Callback] = []
        self.tick_count = 0
        self.sensors: Dict[str, Sensor] = {}
        # self.anomaly_data = []

        # a datastore for data shared between callbacks etc.
        self.data = {}

    def add_callback(self, cb: Callback):
        self.callbacks.append(cb)

    def add_sensor(self, sensor: Sensor):
        self.sensors[sensor.name] = sensor

    def get_world_camera(self, i: int = 0) -> np.ndarray:
        # Get the camera matrix
        act = [sensor.actor for sensor in self.sensors.values()]
        return np.array(act[i].get_transform().get_inverse_matrix())

    @lru_cache()
    def get_proj_matrix(self) -> np.ndarray:
        # Calculate the camera projection matrix to project from 3D -> 2D
        # TODO: MAKE THIS CONFIGURABLE!
        image_w, image_h, fov = (1920, 1080, 105)  # 1280,720
        return build_projection_matrix(image_w, image_h, fov)

    def do_simulation_tick(self) -> None:
        """
        This function does to the processing of a single tick
        :return:
        """
        measurements = {}
        for k, sensor in self.sensors.items():
            # only get measurements every n frames
            if self.tick_count % sensor.every_n_frames == 0:
                measurements[k] = sensor.get_measurement()

        # Call callbacks only every n frames
        for callback in self.callbacks:
            if self.tick_count % callback.every_n_frames == 0:
                callback.on_simulation_tick_starts(
                    self,
                    measurements=measurements,
                )

        for callback in self.callbacks:
            if self.tick_count % callback.every_n_frames == 0:
                callback.on_simulation_tick(
                    self,
                    measurements=measurements,
                )

        for callback in self.callbacks:
            if self.tick_count % callback.every_n_frames == 0:
                callback.on_simulation_tick_ends(
                    self,
                    measurements=measurements,
                )

        return

    def spawn_carla(self) -> sp.Popen:
        # TODO: make quality level etc. configurable
        log.info(f"Spawning environment")
        proc = sp.Popen(
            [
                os.path.join(self.cfg.carla.bin, self.cfg.carla.file),
                "-RenderOffScreen",
                f"-quality-level={self.cfg.carla.quality_level}",
            ]
        )
        log.info(f"Waiting")
        time.sleep(10)
        return proc

    def run(self) -> None:
        """ """
        self.tick_count = 0
        frames_per_second = 1 / self.cfg.fixed_delta_seconds

        self.end_tick = int(self.cfg.scene_duration * frames_per_second)

        try:
            log.info(f"Starting simulation")
            for callback in self.callbacks:
                callback.on_simulation_start(self)

            log.info(f"Skipping {self.cfg.skip_start_ticks} ticks")
            for _ in tqdm.tqdm(range(self.cfg.skip_start_ticks)):
                # this will not trigger callback ticks
                self.world.tick()

            self.clear_sensor_queues()

            with logging_redirect_tqdm():
                bar = tqdm.tqdm(range(self.end_tick), smoothing=0.05)
                for frame in bar:
                    self.world.tick()

                    self.do_simulation_tick()

                    self.clear_sensor_queues()
                    self.tick_count += 1

                for callback in self.callbacks:
                    callback.on_simulation_end(self)

        except Exception as e:
            log.exception(e)
        finally:
            self.terminate()

    def setup(self):
        """
        Sets up the simulation and the entire world, spawns stuff
        """
        if self.cfg.carla.spawn:
            self.proc = self.spawn_carla()
        else:
            self.proc = None

        os.makedirs(self.root, exist_ok=True)

        self.client = connect_client(self.cfg)

        # load world
        log.info(f"Loading World '{self.cfg.world_name}'")
        self.world = self.client.load_world(self.cfg.world_name)

        time.sleep(1)

        # TODO: we could use a callback for this
        set_weather(self.world, self.cfg.weather)
        time.sleep(1)
        apply_settings(self.world, self.cfg)
        self.world.tick()

        self.tm = configure_traffic_manager(self.client)

        # create cars
        log.info(f"Spawning cars")
        # TODO: we could use a callback for this
        create_vehicles(self.world, **self.cfg.vehicles)

        vehicles_list = self.world.get_actors().filter("*vehicle*")

        for v in vehicles_list:
            v.set_autopilot(True)

        self.world.tick()

        for v in vehicles_list:
            self.tm.update_vehicle_lights(v, True)
            self.tm.ignore_vehicles_percentage(v, 0.0)
            self.tm.ignore_lights_percentage(v, 0.0)
            self.tm.ignore_signs_percentage(v, 0.0)

            # walkers close to the street will force vehicles to break, which we want to avoid here
            self.tm.ignore_walkers_percentage(v, 0.0)
            self.tm.keep_right_rule_percentage(v, 100.0)
            self.tm.random_left_lanechange_percentage(v, 00.0)
            self.tm.random_right_lanechange_percentage(v, 00.0)
            self.tm.vehicle_percentage_speed_difference(v, -10.0)
            self.tm.auto_lane_change(v, True)

        self.world.tick()

        # create pedestrians
        # TODO: we could use a callback for this
        log.info(f"Spawning pedestrians")
        create_pedestrians(self.client, self.world, **self.cfg.pedestrians)
        self.world.tick()

        # ego vehicle
        log.info(f"Spawning ego")
        self.ego_vehicle = create_egovehicle(
            self.world,
            mode=self.cfg.ego.mode,
            seed=self.cfg.ego.seed,
            spawn=self.cfg.ego.index,
            model_filter=self.cfg.ego.model,
        )
        self.world.tick()

        # self.tm.update_vehicle_lights(v, True)

        # configure ego vehicle
        self.ego_vehicle.set_autopilot(self.cfg.ego.autopilot)
        # self.tm.distance_to_leading_vehicle(
        #     self.ego_vehicle, self.cfg.ego.distance_to_leading
        # )
        self.tm.vehicle_percentage_speed_difference(
            self.ego_vehicle, self.cfg.ego.percentage_speed
        )
        self.tm.auto_lane_change(
            self.ego_vehicle, self.cfg.ego.auto_lane_change
        )
        self.tm.ignore_walkers_percentage(
            self.ego_vehicle, self.cfg.ego.ignore_walkers_percentage
        )
        self.tm.ignore_lights_percentage(
            self.ego_vehicle, self.cfg.ego.ignore_lights_percentage
        )

    def clear_sensor_queues(self):
        for sensor in self.sensors.values():
            sensor.clear()

    def terminate(self) -> None:
        destroy_actors([sensor.actor for sensor in self.sensors.values()])

        if self.proc:
            bonk(self.proc)
        else:
            teardown(client=self.client, tm=self.tm)


def bonk(proc):
    """
    kill the process until it is dead.
    :param proc:
    :return:
    """
    log.info("Terminating")
    assert proc is not None
    pid = proc.pid
    proc.terminate()
    time.sleep(5)

    while is_running(pid):
        proc.kill()
        outs, errs = proc.communicate()
        time.sleep(1)

    log.info("Killed")
    time.sleep(10)


def is_running(pid):
    try:
        os.kill(pid, 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
            return False
    return True
