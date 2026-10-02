import json
import logging
from os.path import join
from typing import TYPE_CHECKING, Any, Dict
import carla
import numpy as np
from carla import ColorConverter as cc
import math

from carlagen.utils.carla import (
    is_far_away,
    print_agent_attributes,
    get_bicycle_extent_dict,
    transform_bbox_object_to_agent,
    get_agent_attributes,
    get_prop_extent_dict,
)
from carlagen.kitti.bounding_box import (
    transforms_from_agent,
    get_bounding_box_and_refpoint,
    calculate_occlusion_stats,
    crop_boxes_in_canvas,
    calc_projected_2d_bbox,
    calc_bbox2d_area,
    calculate_occlusion,
    calculate_truncation,
    get_relative_rotation_y,
    get_alpha,
    MIN_VISIBLE_VERTICES_FOR_RENDER,
    MIN_BBOX_AREA_IN_PX,
)

import pandas as pd
from carlagen.kitti.datadescriptor import KittiDescriptor
from carlagen.utils.timer import log_timer

if TYPE_CHECKING:
    from carlagen import Simulation

log = logging.getLogger(__name__)


def create_kitti_datapoint(
    agent, camera, cam_calibration, image, depth_map, player_transform, max_render_depth
):
    """
    Calculates the bounding box of the given agent, and
    returns a KittiDescriptor which describes the object to be labeled

    """

    obj_type, agent_transform, bbox_transform, ext, location = transforms_from_agent(
        agent
    )

    # this checks if the vehicle is a bicycle/ motorcycle, if it is, then we manually need to change the bounding box extent
    # this is redundant, but we cannot change it, as the agent does not have setters for the extent :)))
    bicycle_fix_dict_town_01 = get_bicycle_extent_dict()
    if agent.type_id in bicycle_fix_dict_town_01:
        ext.x = bicycle_fix_dict_town_01[agent.type_id]["ext"][0]
        ext.y = bicycle_fix_dict_town_01[agent.type_id]["ext"][1]

    prop_extent_dict = get_prop_extent_dict()
    if agent.type_id in prop_extent_dict:
        ext.x = prop_extent_dict[agent.type_id]["ext"][0] / 2
        ext.y = prop_extent_dict[agent.type_id]["ext"][1] / 2
        log.info(f"Changed extent for {agent.type_id} to {ext.x*2}, {ext.y*2}")
        # ext.z = prop_extent_dict[agent.type_id]["ext"][2]

    if obj_type is None:
        log.warning(
            "Could not get bounding box for agent. Object type is None. This signals something is going wrong! Probably you forgot to specify the object type in the code."
        )
        return image, None

    # in this method manipulate the bounding box, as we cant do it in the agent itself...

    (camera_bbox, camera_refpoint), (
        sensor_bbox,
        sensor_refpoint,
    ) = get_bounding_box_and_refpoint(agent, camera, cam_calibration)

    if agent.type_id in get_prop_extent_dict():
        sensor_refpoint_str = np.array2string(
            sensor_refpoint, precision=6, separator=",", suppress_small=True
        )
        log.info(f"Sensor refpoint: {sensor_refpoint_str} for id {agent.type_id}")

    num_visible_vertices, num_vertices_outside_camera = calculate_occlusion_stats(
        image, camera_bbox, depth_map, max_render_depth, draw_vertices=False
    )

    # At least N vertices have to be visible in order to draw bbox
    try:
        if (
            num_visible_vertices
            >= MIN_VISIBLE_VERTICES_FOR_RENDER
            #   > num_vertices_outside_camera
        ):
            # Visualize midpoint for agents
            # draw_rect(image, (camera_refpoint[1], camera_refpoint[0]), 4)
            uncropped_bbox_2d = calc_projected_2d_bbox(camera_bbox)

            # Crop vertices outside camera to image edges
            crop_boxes_in_canvas(camera_bbox)

            bbox_2d = calc_projected_2d_bbox(camera_bbox)

            area = calc_bbox2d_area(bbox_2d)
            if area < MIN_BBOX_AREA_IN_PX:
                if obj_type not in ["Bicycle", "Motorcycle"]:
                    log.debug("Filtered out bbox with too low area {}".format(area))
                    return image, None, None

            alpha = get_alpha(agent, player_transform)

            if alpha < -3.1415 or alpha > 3.1415 or math.isnan(alpha):
                log.warning(f"Invalid alpha value: {alpha}")
                return image, None, None

            # occlusion = calculate_occlusion(camera_bbox, agent, depth_map)
            rotation_y = get_relative_rotation_y(agent, player_transform)

            truncation = calculate_truncation(uncropped_bbox_2d, bbox_2d)
            datapoint = KittiDescriptor()
            datapoint.set_type(obj_type)
            datapoint.set_bbox(bbox_2d)
            datapoint.set_3d_object_dimensions(ext)
            datapoint.set_3d_object_location(sensor_refpoint)
            datapoint.set_rotation_y(rotation_y)

            # log.info(f"{alpha=}")
            datapoint.set_alpha(alpha)
            datapoint.set_truncated(truncation)
            datapoint.set_occlusion(0)  #  occlusion)

            return image, datapoint, camera_bbox
    except ValueError:
        # nan in get_alpha
        pass

    return image, None, None


def get_intrinsic_matrix(camera):
    width = int(camera.sensor_bp.get_attribute("image_size_x"))
    height = int(camera.sensor_bp.get_attribute("image_size_y"))
    fov = 105  # hardcoded because it is somehow not accessible from the camera sensor data

    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / (2.0 * np.tan(fov * np.pi / 360.0))

    return k


def save_calibration_matrices(filename, intrinsic_mat, lidar_cam_mat):
    """Saves the calibration matrices to a file.
    AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
    The resulting file will contain:
    3x4    p0-p3      Camera P matrix. Contains extrinsic
                    and intrinsic parameters. (P=K*[R;t])
    3x3    r0_rect    Rectification matrix, required to transform points
                    from velodyne to camera coordinate frame.
    3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                coordinate frame according to:
                                Point_Camera = P_cam * R0_rect *
                                                Tr_velo_to_cam *
                                                Point_Velodyne.
    3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                imu data.
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = "C"
    P0 = intrinsic_mat
    P0 = np.column_stack((P0, np.array([0, 0, 0])))
    P0 = np.ravel(P0, order=ravel_mode)
    R0 = np.identity(3)

    # TODO Remove the bellow hardcoded rotation matrix and write it based on the lidar_cam_mat matrix.
    # Currently it is assumed that the lidar has the same rotation as the camera but can vary in location (see below)
    #  We need to convert left-hand camera frame to right-hand kitti camera frame.

    R_velodyne = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

    # Add translation vector from velo to camera.
    T_velodyne = np.array(
        [lidar_cam_mat[1, 3], -lidar_cam_mat[2, 3], lidar_cam_mat[0, 3]]
    )
    TR_velodyne = np.column_stack((R_velodyne, T_velodyne))
    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write(
            "{}: {}\n".format(
                name, " ".join(map(str, arr.flatten(ravel_mode).squeeze()))
            )
        )

    # All matrices are written on a line with spacing
    with open(filename, "w") as f:
        for i in range(
            4
        ):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    # logging.info("Wrote all calibration matrices to %s", filename)


def convert_depth_map_to_normalized_depth(depth_map_image):
    color_converter = cc.Depth
    depth_map_image.convert(color_converter)
    array = np.frombuffer(depth_map_image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (depth_map_image.height, depth_map_image.width, 4))
    array = array[:, :, :3]
    array = array.astype(np.float32)
    normalized_depth = np.dot(array, np.array([65536.0, 256.0, 1.0]))
    normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
    depth_map = 1000.0 * normalized_depth
    return depth_map


def get_sensor_matrices(sim, lidar_sensor_name, rgb_camera_name):
    """
    :param rgb_camera_name:
    :param lidar_sensor_name:
    :param sim:
    :return:
    """
    camera_rgb_intrinsic = get_intrinsic_matrix(sim.sensors[rgb_camera_name])
    veh_cam_mat = sim.sensors[rgb_camera_name].sensor_transform.get_inverse_matrix()
    lidar_veh_mat = sim.sensors[lidar_sensor_name].sensor_transform.get_matrix()
    lidar_cam_mat = np.dot(veh_cam_mat, lidar_veh_mat)
    return camera_rgb_intrinsic, lidar_cam_mat


# create city object datapoints, they require special transformations
city_objects_list = [
    (carla.CityObjectLabel.TrafficLight, "TrafficLight"),
    (carla.CityObjectLabel.TrafficSigns, "TrafficSign"),
    (carla.CityObjectLabel.Buildings, "Buildings"),
    (carla.CityObjectLabel.Fences, "Fences"),
    (carla.CityObjectLabel.Other, "Other"),
    (carla.CityObjectLabel.Poles, "Poles"),
    # (carla.CityObjectLabel.RoadLines, "RoadLines"),
    # (carla.CityObjectLabel.Roads, "Roads"),
    ## (carla.CityObjectLabel.Sidewalks, "Sidewalks"),
    # (carla.CityObjectLabel.Vegetation, "Vegetation"),
    (carla.CityObjectLabel.Rider, "Rider"),
    (carla.CityObjectLabel.Train, "Train"),
    (carla.CityObjectLabel.Walls, "Walls"),
    # (carla.CityObjectLabel.Sky, "Sky"),
    # (carla.CityObjectLabel.Ground, "Ground"),
    # (carla.CityObjectLabel.Bridge, "Bridge"),
    # (carla.CityObjectLabel.RailTrack, "RailTrack"),
    (carla.CityObjectLabel.GuardRail, "GuardRail"),
    (carla.CityObjectLabel.Static, "Static"),
    (carla.CityObjectLabel.Dynamic, "Dynamic"),
    # (carla.CityObjectLabel.Water, "Water"),
    # (carla.CityObjectLabel.Terrain, "Terrain"),
    # (carla.CityObjectLabel.Any, "Any")
]


def create_agent_list(sim) -> list:
    agents_list = []

    # pedestrians
    pedestrians_list = sim.world.get_actors().filter("walker.pedestrian.*")
    agents_list.extend(pedestrians_list)

    # this does not only create car datapoints but also vehicle datapoints, thus we automatically create bicycle datapoints
    # cars, bicycles and motorcycles
    vehicles_list = sim.world.get_actors().filter("vehicle.*")
    agents_list.extend(vehicles_list)

    for city_object, obj_type in city_objects_list:
        object_list = sim.world.get_level_bbs(city_object)
        object_actor_list = [
            transform_bbox_object_to_agent(el, f"City{obj_type}") for el in object_list
        ]
        agents_list.extend(object_actor_list)

    # print(f"Using following anomaly list in KITTI: {sim.anomaly_data}")
    # add anomaly data
    # if len(sim.anomaly_data) != 0:
    #     # get a
    #     anomaly_datum = sim.anomaly_data[0]["id"]
#
    #     ano_actor = sim.world.get_actor(anomaly_datum)
    #     agents_list.extend([ano_actor])

    return agents_list


def create_kitti_data(
    sim: "Simulation",
    measurements: Dict[str, Any],
    directory: str,
    max_distance: int = 50,
    rgb_cam_name: str = "RGB Camera",
    depth_cam_name: str = "Depth Camera",
    lidar_sensor_name: str = "LIDAR",
):
    """
    creates a kitti data file for the current frame
    code heavily inspired by https://github.com/fnozarian/CARLA-KITTI/

    this creates a calibration matrix that is saved in the 'calib' directory,
    as well as bounding boxes in a 'label_2' directory.

    :param lidar_sensor_name: name of the lidar sensor to use
    :param sim:
    :param measurements: dictionary with measurements from the current time step
    :param rgb_cam_name: name of an rgb cam sensor
    :param depth_cam_name: name of a depth-cam sensor
    :param directory: where to store the data
    :param max_distance: max distance before objects are considered too far away
    """
    camera_rgb_intrinsic, lidar_cam_mat = get_sensor_matrices(
        sim, lidar_sensor_name, rgb_cam_name
    )
    # NOTE: it would be sufficient to write this our once, since our sensors are statically attached
    save_calibration_matrices(
        join(directory, "calib", f"{sim.tick_count:06d}.txt"),
        camera_rgb_intrinsic,
        lidar_cam_mat,
    )

    # this map is needed to compute the occlusion
    with log_timer(log, msg="Normalizing Depth Map"):
        depth_map = convert_depth_map_to_normalized_depth(
            measurements[depth_cam_name],
        )

    agents_list = create_agent_list(sim)

    datapoints = []
    id_points = []

    with log_timer(log, msg=f"Created kitti for {len(agents_list)} agents"):
        for agent in agents_list:
            if is_far_away(max_distance, agent, sim.ego_vehicle):
                log.debug(f"Agent of type {agent.type_id} is too far away. Skipping")
                continue

            image, kitti_datapoint, bounding_box = create_kitti_datapoint(
                agent=agent,
                camera=sim.sensors[rgb_cam_name].actor,
                cam_calibration=camera_rgb_intrinsic,
                image=measurements[rgb_cam_name],
                depth_map=depth_map,
                player_transform=sim.ego_vehicle.get_transform(),
                max_render_depth=50,  # hardcoded max render depth,
            )

            if kitti_datapoint:
                agent_atts = get_agent_attributes(agent)

                datapoints.append(kitti_datapoint)
                id_points.append(agent_atts)
            else:
                log.debug(
                    f"Could not get bounding box for agent with id {agent.type_id}. Skipping"
                )

    with log_timer(
        log,
        msg="Saved kitti files to disk",
    ):
        label_filename = join(directory, "label_2", f"{sim.tick_count:06d}.txt")
        with open(label_filename, "w") as f:
            out_str = "\n".join([str(point) for point in datapoints if point])
            f.write(out_str)

        extended_label_filename = join(
            directory, "extended", f"{sim.tick_count:06d}.feather"
        )

        # create a mix with kitti json and id json
        with open(extended_label_filename, "w") as f:
            jsons_list = []
            for point_data, id_data in zip(datapoints, id_points):
                combined_dict = {}
                if point_data:
                    combined_dict["kitti"] = point_data.to_dict()
                    combined_dict["simulation_data"] = id_data
                    jsons_list.append(combined_dict)

            df = pd.json_normalize(jsons_list, sep=".").reset_index()
            df.to_feather(extended_label_filename)


