"""
Callbacks related to saving info from the simulation

"""
import logging
import os
from os.path import join
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

import carla
import cv2
import numpy as np
import pandas as pd
import math

from carlagen.kitti import create_kitti_data
from carlagen.utils.timer import log_timer
from .callback import Callback


class CarlaRecorderCallback(Callback):
    """
    Record everything using carla
    """

    def __init__(self, recorder_file="carla.log", **kwargs):
        super().__init__(**kwargs)
        self.recorder_file = recorder_file

    def on_simulation_start(self, sim: "Simulation"):
        path = join(os.getcwd(), self.recorder_file)
        r = sim.client.start_recorder(path)
        self.log.info(f"Recording on file: {r}")

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        sim.client.stop_recorder()


class SaveIMUCallback(Callback):
    def __init__(self, filename, steps=10, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.data = []
        self.steps = steps

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        m = measurements["IMU"]
        self.data.append(
            {
                "acceleration_x": m.accelerometer.x,
                "acceleration_y": m.accelerometer.y,
                "acceleration_z": m.accelerometer.z,
                "compass": m.compass,
                "longitude_x": m.gyroscope.x,
                "longitude_y": m.gyroscope.y,
                "longitude_z": m.gyroscope.z,
            }
        )

        # if not sim.tick_count % self.steps:
        #     pd.DataFrame(self.data).to_csv(self.filename)

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        pd.DataFrame(self.data).to_feather(self.filename, compression="zstd")


class SaveGNSSCallback(Callback):
    def __init__(self, filename, steps=10, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self._data = []
        self.steps = steps

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        m = measurements["GNSS"]
        self._data.append(
            {"altitude": m.altitude, "latitude": m.latitude, "longitude": m.longitude}
        )

        # if not sim.tick_count % self.steps:
        #     pd.DataFrame(self._data).to_csv(self.filename)

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        pd.DataFrame(self._data).to_feather(self.filename, compression="zstd")


class SaveSemanticLidarCallback(Callback):
    def __init__(
        self,
        directory,
        sensor_name="Semantic LIDAR",
        point_cloud_format="feather",
        **kwargs,
    ):
        """

        :param directory:
        :param sensor_name: which sensor to use
        """
        super().__init__(**kwargs)
        assert point_cloud_format in ["ply", "bin", "feather"]

        self.root = directory
        self.directory = directory
        self.sensor_name = sensor_name
        self.point_cloud_format = point_cloud_format

    def on_simulation_start(self, sim: "Simulation"):
        os.makedirs(self.directory, exist_ok=True)

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        try:
            lidar = measurements[self.sensor_name]
            # NOTE: this is very crude multitasking, not yet platform independent
            # we prevent this call to save from blocking, reduces execution time by ~10%
            # proc = mp.Process(target=self.save, args=(sim, lidar))
            # proc.start()

            self.save(sim, lidar)
        except KeyError:
            self.log.warning(f"Lidar {self.sensor_name} not found")

    def save(self, sim, lidar):
        with log_timer(
            self.log,
            msg="Saved semantic lidar to disk",
        ):
            if self.point_cloud_format == "ply":
                # NOTE: in ply format, the y axis will not be inverted
                lidar.save_to_disk(join(self.directory, f"{sim.tick_count:06d}.ply"))
            elif self.point_cloud_format == "bin":

                data = np.empty(shape=(len(lidar), 4), dtype=np.dtype("f4"))
                meta = np.empty(shape=(len(lidar), 2), dtype=np.uint32)

                for i, detection in enumerate(lidar):
                    data[i] = (
                        detection.point.x,
                        -detection.point.y,
                        detection.point.z,
                        detection.cos_inc_angle,
                    )
                    meta[i] = detection.object_idx, detection.object_tag

                data.tofile(join(self.directory, f"{sim.tick_count:06d}.bin"))
                meta.tofile(join(self.directory, f"labels-{sim.tick_count:06d}.bin"))
            elif self.point_cloud_format == "feather":
                data = [{
                    "x": detection.point.x,
                    "y":  - detection.point.y,
                    "z": detection.point.z,
                    "cos_inc_angle": detection.cos_inc_angle,
                    "object_id": detection.object_idx,
                    "object_tag": detection.object_tag
                } for detection in lidar]

                pd.DataFrame(data).to_feather(join(self.directory, f"{sim.tick_count:06d}.feather"), compression="zstd")


class SaveLidarCallback(Callback):
    def __init__(
        self, directory, sensor_name="LIDAR", point_cloud_format="bin", **kwargs
    ):
        """

        :param directory:
        :param sensor_name: which sensor to use
        :param point_cloud_format: can be 'ply' or 'bin'
        """
        super().__init__(**kwargs)
        self.root = directory
        # in the kitti dataset, the lidar data is stored in the velodyne folder
        self.directory = join(directory, "velodyne")
        self.sensor_name = sensor_name
        self.point_cloud_format = point_cloud_format

    def on_simulation_start(self, sim: "Simulation"):
        os.makedirs(self.directory, exist_ok=True)

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        try:
            lidar = measurements[self.sensor_name]

            if self.point_cloud_format == "ply":
                lidar.save_to_disk(join(self.directory, f"{sim.tick_count:06d}.ply"))
            elif self.point_cloud_format == "bin":
                points = np.frombuffer(lidar.raw_data, dtype=np.dtype("f4"))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                # invert y axis
                points = points.copy()
                points[:, 1] *= -1
                points.tofile(join(self.directory, f"{sim.tick_count:06d}.bin"))
            elif self.point_cloud_format == "feather":
                points = np.frombuffer(lidar.raw_data, dtype=np.dtype("f4"))
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
                points[:, 1] *= -1
                df = pd.DataFrame(points, columns=["x", "y", "z", "cos_inc_angle"])
                df.to_feather(join(self.directory, f"{sim.tick_count:06d}.feather"), compression="zstd")

        except KeyError:
            self.log.warning(f"Lidar {self.sensor_name} not found")


class SaveKITTICallback(Callback):
    def __init__(
        self,
        directory,
        lidar_name,
        rgb_cam_name,
        depth_cam_name,
        max_distance=50,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.directory = directory
        self.lidar_name = lidar_name
        self.rgb_cam_name = rgb_cam_name
        self.depth_cam_name = depth_cam_name
        self.max_distance = max_distance

    def on_simulation_start(self, sim: "Simulation"):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # these directories are needed for kitti
        os.makedirs(join(self.directory, "calib"))
        os.makedirs(join(self.directory, "label_2"))
        os.makedirs(join(self.directory, "extended"))

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        try:
            with log_timer(
                self.log,
                msg="Created Kitti",
            ):
                create_kitti_data(
                    sim,
                    measurements,
                    self.directory,
                    lidar_sensor_name=self.lidar_name,
                    rgb_cam_name=self.rgb_cam_name,
                    depth_cam_name=self.depth_cam_name,
                    max_distance=self.max_distance,
                )

        except KeyError as e:
            self.log.warning(
                f"Could not export to KITTI Format. Sensor not found: {str(e)}"
            )


class SaveInstanceSegmentationCallback(Callback):
    def __init__(self, directory, sensor_name: str = "Instance Segmentation", **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.sensor_name = sensor_name

    def on_simulation_start(self, sim: "Simulation"):
        os.makedirs(self.directory)

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        try:
            img = measurements[self.sensor_name]
            self.save(sim, img)
            # context = mp.get_context('fork')
            # proc = context.Process(target=self.save, args=(sim, img,))
            # proc.start()
        except KeyError:
            self.log.warning(f"Instance Segmentation not found")

    def save(self, sim, img):
        with log_timer(self.log, msg="Created Instance Segmentation"):

            img: np.ndarray = np.reshape(
                np.copy(img.raw_data), (img.height, img.width, 4)
            )

            cv2.imwrite(
                join(sim.root, join(self.directory, f"{sim.tick_count:06d}.png")), img
            )


class SaveWeatherCallback(Callback):
    """ """

    def __init__(self, filename, steps=10, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.steps = steps
        self._data = []

    def on_simulation_tick_ends(self, sim: "Simulation", **kwargs):
        weather = sim.world.get_weather()

        self._data.append(
            {
                "cloudiness": weather.cloudiness,
                "precipitation": weather.precipitation,
                "sun_altitude_angle": weather.sun_altitude_angle,
                "sun_azimuth_angle": weather.sun_azimuth_angle,
                "fog_density": weather.fog_density,
                "fog_distance": weather.fog_distance,
                "fog_falloff": weather.fog_falloff,
                "precipitation_deposits": weather.precipitation_deposits,
                "wind_intensity": weather.wind_intensity,
                "wetness": weather.wetness,
                "scattering_intensity": weather.scattering_intensity,
                "mie_scattering_scale": weather.mie_scattering_scale,
                "rayleigh_scattering_scale": weather.rayleigh_scattering_scale,
                "dust_storm": weather.dust_storm,
            }
        )

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        pd.DataFrame(self._data).to_feather(self.filename, compression="zstd")


class DebugCallback(Callback):
    """ """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_simulation_start(self, sim: "Simulation"):
        self.log.debug(f"dumping actors")
        for actor in sim.world.get_actors():
            dx = actor.bounding_box.extent.x
            dy = actor.bounding_box.extent.y
            dz = actor.bounding_box.extent.z

            self.log.debug(
                f"Actor: {actor.id} {actor.type_id} {actor.semantic_tags} - extent: {dx,dy,dz=}"
            )

        self.log.debug(f"Dumping world spawn points")
        spawn_points = sim.world.get_map().get_spawn_points()
        for loc in spawn_points:
            self.log.debug(loc)


# class SaveSegmentationCallback(Callback):
#     """
#     Lossy compression like jpg will result in wrong labels for some pixels, so
#     we use lossless png instead.
#     """
#     def __init__(self, directory="seg"):
#         super().__init__()
#         self.directory = directory
#
#     def on_simulation_start(self, sim: "Simulation", **kwargs):
#         os.makedirs(join(sim.root, self.directory), exist_ok=True)
#
#     def on_simulation_tick_ends(self, sim: "Simulation", seg_mask, **kwargs):
#         seg_mask.save_to_disk(
#             join(sim.root, self.directory, f"{sim.tick_count:010d}.png"),
#             carla.ColorConverter.CityScapesPalette,
#         )


class SaveCameraCallback(Callback):
    """
    We save as jpg for compression
    """

    def __init__(self, directory, sensor_name: str = None, extension="jpg", **kwargs):
        super().__init__(**kwargs)
        self.directory = directory
        self.name = sensor_name
        self.extension = extension

    def on_simulation_start(self, sim: "Simulation", **kwargs):
        os.makedirs(join(sim.root, self.directory), exist_ok=True)

    def on_simulation_tick_ends(self, sim: "Simulation", measurements: Dict, **kwargs):
        img: np.ndarray = measurements[self.name]
        self.save(sim, img)
        # context = mp.get_context('fork')
        # proc = context.Process(target=self.save, args=(sim, img))
        # proc.start()

    def save(self, sim, img):
        path = join(
            sim.root, self.directory, f"{sim.tick_count:06d}.{self.extension}"
        )

        with log_timer(self.log, msg=f"Created Camera -> {path}"):
            img: np.ndarray = np.reshape(
                np.copy(img.raw_data), (img.height, img.width, 4)
            )

            cv2.imwrite(path, img)


class SaveDepthCameraCallback(SaveCameraCallback):
    """
    Save camera image with depth information
    """

    def __init__(self, directory, sensor_name: str = None, extension="png", **kwargs):
        super().__init__(directory, sensor_name, extension, **kwargs)

    def save(self, sim, img):
        # Step 1: Convert raw BGRA image data to a NumPy array of shape (H, W, 4)
        # img: np.ndarray = np.reshape(
        #     np.copy(img.raw_data), (img.height, img.width, 4)
        # )

        # # Step 2: Extract RGB channels from the image (ignore alpha channel)
        # B = img[:, :, 0].astype(np.float32)
        # G = img[:, :, 1].astype(np.float32)
        # R = img[:, :, 2].astype(np.float32)

        # # Step 3: Convert RGB-encoded depth to normalized float using provided formula
        # normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)

        # # # Step 4: Convert normalized depth to meters
        # # in_meters = 1000 * normalized

        # # Step 5: Construct the save path using simulation root, target directory, and tick count
        path = join(sim.root, self.directory, f"{sim.tick_count:06d}.{self.extension}")

        # # Step 6: Save the depth image (in meters) as a grayscale image
        # # Normalize to 0–255 for visualization or save raw with appropriate format if needed
        # depth_img = np.clip(normalized / normalized.max() * 255, 0, 255).astype(np.uint8)
        # cv2.imwrite(path, depth_img)
        img.save_to_disk(path, carla.ColorConverter.LogarithmicDepth)


class SaveCameraNoiseCallback(SaveCameraCallback):
    """
    Save camera image with added guassian noise

    # TODO: add some delay and buildup for the noise
    """

    def __init__(self, noise=1.0, **kwargs):
        super().__init__(**kwargs)
        self.noise = noise

    def on_simulation_tick_ends(self, sim: "Simulation", measurements, **kwargs):
        image: np.ndarray = measurements[self.name]
        img: np.ndarray = np.reshape(
            np.copy(image.raw_data), (image.height, image.width, 4)
        )
        img = img + self.noise * np.random.randn(*img.shape)
        img = np.clip(img, 0, 255)
        path = join(sim.root, self.directory, f"{sim.tick_count:06d}.{self.extension}")
        cv2.imwrite(path, img)


class CreateVideoCallback(Callback):
    """
    Create video after simulation ends, using FFMPEG.
    """

    def __init__(self, directories = None, filename=None, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.directories = directories

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        if not self.directories:
            # iterate over all directories
            ds = os.listdir(sim.root)
            # filter for directories which are folders with rgb or segmentation in the name
            ds = [
                d for d in ds if os.path.isdir(join(sim.root, d))
            ]
            self.directories = [
                d for d in ds if "rgb" in d or "segmentation" in d
            ]

        video_path = join(sim.root, "video")
        os.makedirs(video_path, exist_ok=True)
        for directory in self.directories:
            path = join(sim.root, directory)
            video_output_file = join(video_path, str(directory) + ".mp4")
            image_format = "png"
            for elem in os.listdir(path):
                if "jpg" in elem:
                    image_format = "jpg"
                    break
            os.system(
                f"ffmpeg -framerate {int(1/sim.cfg.fixed_delta_seconds)} -pattern_type glob -i '{path}/*.{image_format}' -c:v libx264 -pix_fmt yuv420p {video_output_file}"
            )

class SaveDepthCameraCallback(SaveCameraCallback):
    """
    Save camera image with depth information
    """

    def __init__(self, directory, sensor_name: str = None, extension="png",**kwargs):
        super().__init__(directory, sensor_name, extension, **kwargs)

    def save(self, sim, img):
        path = join(sim.root, self.directory, f"{sim.tick_count:06d}.{self.extension}")
        img.save_to_disk(path, carla.ColorConverter.LogarithmicDepth)


class SaveAutopilotActions(Callback):
    """
    Record the last-applied VehicleControl each tick and save to CSV at the end.

    Args:
        csv_path (str): output CSV path.
        every_n_frames (int): sample every n-th sim frame.
    """
    def __init__(self, filename: str = "actions.feather", every_n_frames: int = 1, **kwargs):
        super().__init__(every_n_frames=every_n_frames, **kwargs)
        self.filename = filename
        self._rows = []
        self.log = logging.getLogger(type(self).__name__)
        self.frame_count = 0

    def on_simulation_start(self, sim: "Simulation", **kwargs):
        self._rows = []

    def on_simulation_tick_ends(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        ctrl = sim.ego_vehicle.get_control()  # last-applied control from previous tick

        self._rows.append(
            {
                "frame": self.frame_count,
                "throttle": float(ctrl.throttle),
                "steer": float(ctrl.steer),
                "brake": float(ctrl.brake),
                "hand_brake": bool(ctrl.hand_brake),
                "reverse": bool(ctrl.reverse),
                "manual_gear_shift": bool(ctrl.manual_gear_shift),
                "gear": int(ctrl.gear),
            }
        )

        self.frame_count += 1


    def on_simulation_end(self, sim: "Simulation", **kwargs):
        if not self._rows:
            self.log.warning("No control records collected; skipping CSV write.")
            return
        df = pd.DataFrame(self._rows)
        df.to_feather(join(sim.root, self.filename), compression="zstd")


class SaveCollisions(Callback):
    def __init__(self,
                 sensor_name: str,
                 filename: str = "collisions.feather",
                 every_n_frames: int = 1,
                 compression: str = "zstd",
                 compression_level: Optional[int] = None,
                 **kwargs):
        super().__init__(every_n_frames=every_n_frames, **kwargs)
        self.sensor_name = sensor_name
        self.filename = filename
        self.compression = compression
        self.compression_level = compression_level
        self._rows: List[Dict[str, Any]] = []
        self.frame_count = 0
        self.log = logging.getLogger(type(self).__name__)

    def on_simulation_start(self, sim: "Simulation", **kwargs):
        self._rows.clear()
        self.frame_count = 0

    def on_simulation_tick_ends(self, sim: "Simulation",
                                measurements: Dict[str, Any], **kwargs):
        if self.every_n_frames > 1 and (self.frame_count % self.every_n_frames != 0):
            self.frame_count += 1
            return

        events = measurements.get(self.sensor_name)
        if events is None:
            self.frame_count += 1
            return

        for ev in events:  # list of carla.CollisionEvent, possibly empty
            ni = ev.normal_impulse
            impulse = math.sqrt(ni.x*ni.x + ni.y*ni.y + ni.z*ni.z)
            a = getattr(ev, "actor", None)
            b = getattr(ev, "other_actor", None)
            self._rows.append({
                "frame": self.frame_count,
                "ego_id": int(a.id) if a else None,
                "ego_type": getattr(a, "type_id", None),
                "other_id": int(b.id) if b else None,
                "other_type": getattr(b, "type_id", None),
                "normal_impulse_x": float(ni.x),
                "normal_impulse_y": float(ni.y),
                "normal_impulse_z": float(ni.z),
                "normal_impulse_norm": float(impulse),
            })

        self.frame_count += 1

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        out_path = Path(sim.root) / self.filename

        # Write even if no collisions: create a 0-row DF with schema.
        if not self._rows:
            cols = ["frame","ego_id","ego_type","other_id","other_type",
                    "normal_impulse_x","normal_impulse_y","normal_impulse_z","normal_impulse_norm"]
            df = pd.DataFrame(columns=cols)
        else:
            df = pd.DataFrame(self._rows)

        df.to_feather(
            out_path,
            compression=self.compression,              # e.g., "zstd" or "lz4"
            compression_level=self.compression_level   # e.g., 3, or None
        )
        self.log.info("Wrote %d collision rows to %s", len(df), str(out_path))