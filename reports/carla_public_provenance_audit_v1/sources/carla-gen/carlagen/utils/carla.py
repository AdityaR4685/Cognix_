import math
import random
import time
from typing import List

import carla
import numpy as np

import logging
from functools import lru_cache

log = logging.getLogger(__name__)


def connect_client(cfg: dict) -> carla.Client:
    """
    :return:
    """
    log.info(f"Connecting client")
    client = carla.Client(cfg.carla.host, cfg.carla.port)
    client.set_timeout(cfg.carla.timeout)
    log.info(f"Connected!")
    return client


def apply_settings(world: carla.World, cfg):
    log.info("Applying settings")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.substepping = True
    settings.fixed_delta_seconds = cfg.fixed_delta_seconds
    settings.max_substep_delta_time = cfg.max_substep_delta_time
    settings.max_substeps = cfg.max_substeps
    world.apply_settings(settings)
    return settings


def configure_traffic_manager(client: carla.Client):
    """
    TODO: make seed configurable

    :param client:
    :return:
    """
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(12345)
    tm.set_global_distance_to_leading_vehicle(2.0)
    tm.set_hybrid_physics_mode(False)
    tm.set_respawn_dormant_vehicles(True)
    tm.set_boundaries_respawn_dormant_vehicles(25, 700)

    return tm


def get_bounding_box(actor: carla.Actor, min_extent=0.5):
    """
    Some actors like motorbikes have a zero width bounding box. This is a fix to this issue.

    See https://github.com/carla-simulator/carla/issues/3801
    """
    if not hasattr(actor, "bounding_box"):
        # print(f"Bounding box for type {actor.type_id} not implemented. Returning default.")
        return carla.BoundingBox(
            carla.Location(0, 0, min_extent),
            carla.Vector3D(x=min_extent, y=min_extent, z=min_extent),
        )

    bbox = actor.bounding_box
    # Fixing bug related to Carla 9.11 onwards where the bounding box
    # location is wrongly registered for some 2-wheeled vehicles
    buggy_bbox = bbox.extent.x * bbox.extent.y * bbox.extent.z == 0
    if buggy_bbox:
        # print(f"Buggy bounding box found for {actor}")
        bbox.location = carla.Location(0, 0, max(bbox.extent.z, min_extent))
    # Fixing bug related to Carla 9.11 onwards where some bounding boxes have 0 extent
    # https://github.com/carla-simulator/carla/issues/3670
    if bbox.extent.x < min_extent:
        bbox.extent.x = min_extent
    if bbox.extent.y < min_extent:
        bbox.extent.y = min_extent
    if bbox.extent.z < min_extent:
        bbox.extent.z = min_extent

    return bbox


def create_vehicles(world, number, mode, filter, seed, att_filters, broken_cars):
    """

    :param world:
    :param number:
    :param mode:
    :param filter:
    :param seed:
    :param att_filters:
    :return:
    """

    cars = []
    spawns = []

    vehicle_blueprints: List[carla.ActorBlueprint] = (
        world.get_blueprint_library().filter(filter)
    )
    log.info(f"Got {len(vehicle_blueprints)} blueprints for vehicles")
    if len(broken_cars) != 0:
        # remove broken cars
        vehicle_blueprints = [x for x in vehicle_blueprints if x.id not in broken_cars]
        log.info(f"Removed {len(broken_cars)} broken cars")
        for broken in broken_cars:
            log.info(f"Broken car: {broken}")

    for bp in vehicle_blueprints:
        log.info(f"Blueprint: {bp.id} ->  {bp.tags}")

    # attribute filters
    if att_filters:
        for attribute, value in att_filters.items():
            vehicle_blueprints = [
                x
                for x in vehicle_blueprints
                if int(x.get_attribute(attribute)) == value
            ]

    log.info(f"Got {len(vehicle_blueprints)} blueprints for vehicles after filtering")

    # TODO: spawn points should be hardcoded
    spawn_points = world.get_map().get_spawn_points()

    if mode == "random":
        for _ in range(0, number):
            world.try_spawn_actor(
                random.choice(vehicle_blueprints), random.choice(spawn_points)
            )
    elif mode == "seed":
        rng = np.random.default_rng(seed)
        for _ in range(0, number):
            world.try_spawn_actor(
                rng.choice(vehicle_blueprints), rng.choice(spawn_points)
            )
    elif mode == "manual":
        for i in range(0, number):
            world.try_spawn_actor(vehicle_blueprints[cars[i]], spawns[cars[i]])

    # NOTE: if we do not tick here, the actors are not added to the world
    # and auto pilot will not be activated, causing the vehicles to just idle.
    world.tick()

def arr_to_location(arr):
    return carla.libcarla.Location(arr[0], arr[1], arr[2])


def create_pedestrians(
    client: carla.Client,
    world: carla.World,
    number=200,
    models=None,
    mode="seed",
    seed=20,
    pedestrians_cross_factor=0.0,
):
    """
    TODO: use configuration

    :param client:
    :param world:
    :param number:
    :param models: index to use for blueprint selection
    :param mode:
    :param seed:
    :param pedestrians_cross_factor:
    :return:
    """
    # NOTE: this should be set before any pedestrians are spawned, according to online doc
    log.info(f"Using seed {seed} for pedestrians")
    rng = np.random.default_rng(seed)
    world.set_pedestrians_seed(seed)

    # set crossing factor of pedestrians
    # NOTE: this should be set before any pedestrians are spawned, according to online doc
    # NOTE: this does not seem to work on some maps
    log.info(f"Setting pedestrians crossing factor to {pedestrians_cross_factor}")
    world.set_pedestrians_cross_factor(pedestrians_cross_factor)

    # lets try to tick here once
    world.tick()

    if models is None:
        models = []

    spawns = list(world.get_map().get_spawn_points())
    # make order stable
    spawns.sort(key=lambda x: str(x))

    n_spawns = len(spawns)
    spawns = [s for s in spawns if not is_broken_spawn(s)]
    n_spawns_allowed = len(spawns)
    log.info(f"Allowed spawns: {n_spawns_allowed} of {n_spawns}")

    for spawn in spawns:
        log.debug(f"Spawn: {spawn}")

    blueprints = world.get_blueprint_library().filter("walker.pedestrian.*")

    # make all pedestrians non-invincible
    for bp in blueprints:
        if bp.has_attribute('is_invincible'):
            bp.set_attribute('is_invincible', 'false')

    if mode == "seed":
        batch = []

        for i in range(number):
            if models:
                selected_blueprint = blueprints[models[i % len(models)]]
            else:
                selected_blueprint = rng.choice(blueprints)

            if i > len(spawns):
                log.warning(f"Too many pedestrians spawned. All spawns full. Stopping.")
                break

            point = spawns[i]
            log.debug(f"Selected Spawn: {point}")

            batch.append(carla.command.SpawnActor(selected_blueprint, point))
    else:
        raise NotImplementedError()

    walkers_list = []
    # apply the batch
    log.info(f"Spawning ... ")
    results = client.apply_batch_sync(batch, True)
    for r in results:
        if r.error:
            log.warning(f"Error: {r.error}")
        else:
            walkers_list.append({"id": r.actor_id})

    # 3. we spawn the walker controller
    log.info(f"Spawning {len(walkers_list)} walker controllers")
    batch = []
    walker_controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    for walker in walkers_list:
        batch.append(
            carla.command.SpawnActor(
                walker_controller_bp, carla.Transform(), walker["id"]
            )
        )

    # apply the batch
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            log.warning(f"Error: {results[i].error}")
        else:
            walkers_list[i]["con"] = results[i].actor_id

    # 4. we put altogether the walkers and controllers id to get the objects from their id
    all_id = []
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])

    world.tick()
    all_actors = world.get_actors(all_id)

    # TODO: filter dangerous targets
    # spawns = [s for s in spawns if not is_broken_target_location(s)]

    for actor in all_actors:
        log.debug(
            f"Actor: {actor} {actor.attributes} {actor.type_id} {actor.get_location().x}"
        )

    ####################################
    # set destinations and speed for walkers
    log.info(f"Setting destination for walkers")

    if mode == "seed":
        rng = np.random.default_rng(seed)

        for i in range(0, len(all_actors), 2):
            actor = all_actors[i]
            log.debug(f"Processing {actor}")

            point = rng.choice(spawns)

            # start walker
            actor.start()

            # set walk to random point
            point = (point.location.x, point.location.y, point.location.z)

            spawn_point = carla.Transform()
            spawn_point.location = arr_to_location(point)

            log.info(
                f"Setting location for actor at {actor.get_location().x} -> go to {spawn_point.location}"
            )
            actor.go_to_location(spawn_point.location)
            # NOTE: if the sim crashes here, you probably have to blacklist the actors spawn location

            actor.set_max_speed(1 + rng.random())
    else:
        raise NotImplementedError()

    log.info(f"Finished spawning walkers")


def is_broken_spawn(spawn_point) -> bool:
    """
    Carla crashed when pedestrians on some spawn points are given a target location.
    This function contains blacklists for those spawns, based on the x coordinate of the spawn point.
    One coordinate might block several spawns.

    :param spawn_point:
    :return:
    """
    broken_map_1 = [
        -2.4200096130371094,
        1.5599901676177979
    ]

    broken_map_3 = [
        5.989376068115234,
        -20.033100128173828,
        -11.42436695098877,
        -13.384322166442871,
        -23.575590133666992,
        -49.19198989868164,
        -51.339599609375,
    ]
    broken_map_5 = [-128.2843475341797, -131.91021728515625]
    r1 = any([math.isclose(spawn_point.location.x, x) for x in broken_map_1])
    r3 = any([math.isclose(spawn_point.location.x, x) for x in broken_map_3])
    r5 = any([math.isclose(spawn_point.location.x, x) for x in broken_map_5])
    r = r1 or r3 or r5
    log.debug(f"Checking spawn point {spawn_point.location.x} -> {'ok' if r else 'blacklisted'}")
    return r


def create_egovehicle(world: carla.World, mode, spawn, seed, model_filter):
    """

    :param model_filter:
    :param world:
    :param mode:
    :param spawn:
    :param seed:
    :return:
    """
    ego_bp = world.get_blueprint_library().find(model_filter)
    ego_bp.set_attribute("role_name", "hero")
    # TODO: ensure stable order
    spawn_points = world.get_map().get_spawn_points()
    spawned = False
    ego_vehicle = None

    if len(spawn_points) == 0:
        raise ValueError("There must be spawn points")

    while not spawned:
        try:
            if mode == "seed":
                rng = np.random.default_rng(seed)
                point_number = seed % len(spawn_points)
                # ego_vehicle = world.spawn_actor(ego_bp, np.random.choice(spawn_points))
                ego_vehicle = world.spawn_actor(ego_bp, spawn_points[point_number])
            elif mode == "random":
                ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))
            elif mode == "manual":
                ego_vehicle = world.spawn_actor(ego_bp, spawn_points[spawn])

            spawned = True
        except RuntimeError as e:
            log.warning(f"Spawn failed. Retrying ... ")
            if mode == "seed":
                seed += rng.integers(100, 1000)
                log.info(f"New spawn seed: {seed}")

    return ego_vehicle


def start_listen(cams, callbacks, path):
    for i, cam in enumerate(cams):
        cam.listen(lambda data: callbacks[i](data, path))


def destroy_actors(cams):
    log.info("Clearing actors")
    for cam in cams:
        cam.stop()
        try:
            cam.destroy()
        except RuntimeError:
            log.warning(f"Cam already destroyed")


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    k = np.identity(3)
    k[0, 0] = k[1, 1] = focal
    k[0, 2] = w / 2.0
    k[1, 2] = h / 2.0
    return k


def find_bb_limits(vertices, proj_matrix, world_2_camera):
    """
    No idea what this is doing

    :param vertices:
    :param proj_matrix:
    :param world_2_camera:
    :return:
    """
    x_max = -10000
    x_min = 10000
    y_max = -10000
    y_min = 10000

    for vert in vertices:
        p = get_image_point(vert, proj_matrix, world_2_camera)
        # Find the rightmost vertex
        if p[0] > x_max:
            x_max = int(p[0])
        # Find the leftmost vertex
        if p[0] < x_min:
            x_min = int(p[0])
        # Find the highest vertex
        if p[1] > y_max:
            y_max = int(p[1])
        # Find the lowest  vertex
        if p[1] < y_min:
            y_min = int(p[1])
    return int(x_max), int(x_min), int(y_max), int(y_min)


def is_far_away(max_distance, npc, vehicle):
    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
    return dist > max_distance


def is_too_cloose(min_distance, npc, vehicle):
    dist = npc.get_transform().location.distance(vehicle.get_transform().location)
    return dist < min_distance


def is_in_view_direction(npc, vehicle):
    # Calculate the dot product between the forward vector
    # of the vehicle and the vector between the vehicle
    # and the other vehicle. We apply a threshold this dot product
    # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
    forward_vec = vehicle.get_transform().get_forward_vector()
    ray = npc.get_transform().location - vehicle.get_transform().location
    return forward_vec.dot(ray) >= 1


def clip_to_image_resolution(resolution, x_max, x_min, y_max, y_min):
    # check if vehicle is visible:
    y_min = max(int(y_min), 0)
    y_max = min(resolution[0], int(y_max))
    x_min = max(int(x_min), 0)
    x_max = min(resolution[1], int(x_max))

    return x_max, x_min, y_max, y_min


def set_weather(world, cfg):
    log.info("Setting weather")
    if isinstance(cfg, str):
        weather = getattr(carla.WeatherParameters, cfg)
    else:
        weather = carla.WeatherParameters(**cfg)
    world.set_weather(weather)


def teardown(client, tm):
    log.info("Tearing down")
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = False
    tm.set_synchronous_mode(False)
    log.info(f"Setting Synchronous")
    apply_settings(world, settings)
    log.info(f"reloading once again, resetting settings")
    client.reload_world(True)
    log.info(f"Sleeping")
    time.sleep(10)


def get_image_point(loc, k, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth component also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(k, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


class BoundingBoxActor:
    # wrapper class for bounding box actors
    def __init__(self, bounding_box, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bounding_box: carla.BoundingBox = bounding_box
        self.transform: carla.Transform = None
        self.type_id = None
        self.id = -1
        self.attributes = "No Attributes, this is a static object"
        self.semantic_tags = []

    def set_transform(self, transform):
        self.transform = transform

    def get_transform(self):
        return self.transform

    def get_bounding_box(self):
        return self.bounding_box

    def get_location(self):
        return self.bounding_box.location

    def get_rotation(self):
        return self.bounding_box.rotation

    def get_type_id(self):
        return self.type_id

    def get_attributes(self):
        return self.attributes

    def get_id(self):
        return self.id

    def get_semantic_tags(self):
        return self.semantic_tags

    # print
    def __str__(self):
        return f"BoundingBoxActor: {self.bounding_box}, transform: {self.transform}, type_id: {self.type_id}"


def transform_bbox_object_to_agent(object, agent_type_id=None):
    """
    For some functionality, it is useful to transform boundingbox objects like trafficlights or speed signs into an actor object.
    This is explicitly needed for the KITTI format extraction of said objects, as their usual actors are buggy/ are not what we think they are.
    agent_ids:
    - traffic_light
    - speed_limit
    """

    bounding_box_actor = BoundingBoxActor(bounding_box=object)
    bounding_box_actor.set_transform(carla.Transform(object.location, object.rotation))
    bounding_box_actor.bounding_box.location = carla.Location(
        0, 0, 0
    )  # set the transform of the bbox to 0 to get the
    # correct results
    if agent_type_id is None:
        raise ValueError("agent_type_id must be specified")
    bounding_box_actor.type_id = agent_type_id
    return bounding_box_actor


def get_vehicle_type(npc):
    """

    :param npc:
    :return:
    """
    classes = npc.semantic_tags
    if len(classes) == 1:
        return classes[0]
    else:
        # pedestrians is 13, all other vehicle classes have an higher number
        # manchmal auch wall tec, warum auch immer
        return np.max(classes)


@lru_cache
def get_bicycle_extent_dict():
    return {
        "vehicle.vespa.zx125": {
            "ext": [0.93, 0.45],
        },
        "vehicle.bh.crossbike": {
            "ext": [0.8, 0.45],
        },
        "vehicle.gazelle.omafiets": {
            "ext": [0.95, 0.3],
        },
        "vehicle.yamaha.yzf": {
            "ext": [1.15, 0.45],
        },
        "vehicle.kawasaki.ninja": {
            "ext": [1.05, 0.35],
        },
        "vehicle.harley-davidson.low_rider": {
            "ext": [1.2, 0.35],
        },
        "vehicle.diamondback.century": {"ext": [0.85, 0.25]},
    }


@lru_cache
def get_prop_extent_dict():
    return {
        "static.prop.atm": {
            "ext": [1.0, 0.9, 2.27533221244812],
            "offset": [0, 0, 0],
        },
        "static.prop.plantpot06": {
            "ext": [1.6, 1.6, 0.8515846133232117],
            "offset": [0.3, 0, 0, 0],
        },
        "static.prop.plantpot01": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.plantpot05": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.plantpot02": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.plantpot08": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.bench01": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.bench02": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.bench03": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.haybalelb": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.box02": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.box03": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.barbeque": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.guitarcase": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.trashcan03": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.vendingmachine": {
            "ext": [1.2, 1.0, 2.107013463973999],
            "offset": [0, 0, 0],
        },
        "static.prop.gnome": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.travelcase": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.streetsign01": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
        "static.prop.gardenlamp": {
            "ext": [0.5, 0.5, 0.5],
            "offset": [0, 0, 0],
        },
    }


def print_agent_attributes(agent):
    log.info(f"Agent type: {agent.type_id}")
    log.info(f"Agent transform: {agent.get_transform()}")
    log.info(f"Bounding box transform: {carla.Transform(agent.bounding_box.location)}")
    log.info(f"Extent: {agent.bounding_box.extent}")
    log.info(f"Location: {agent.get_transform().location}")
    log.info(f"attributes: {agent.attributes}")
    log.info(f"id: {agent.id}")
    log.info(f"semantic tags: {agent.semantic_tags}")


def get_agent_attributes(agent):

    json_formatted_transform = {
        "x": agent.get_transform().location.x,
        "y": agent.get_transform().location.y,
        "z": agent.get_transform().location.z,
        "pitch": agent.get_transform().rotation.pitch,
        "yaw": agent.get_transform().rotation.yaw,
        "roll": agent.get_transform().rotation.roll,
    }

    json_formatted_bbox_transform = {
        "x": agent.bounding_box.location.x,
        "y": agent.bounding_box.location.y,
        "z": agent.bounding_box.location.z,
        "pitch": agent.bounding_box.rotation.pitch,
        "yaw": agent.bounding_box.rotation.yaw,
        "roll": agent.bounding_box.rotation.roll,
    }

    json_formatted_extent = {
        "x": agent.bounding_box.extent.x,
        "y": agent.bounding_box.extent.y,
        "z": agent.bounding_box.extent.z,
    }

    json_formatted_location = {
        "x": agent.get_transform().location.x,
        "y": agent.get_transform().location.y,
        "z": agent.get_transform().location.z,
    }

    return {
        "type_id": agent.type_id,
        "transform": json_formatted_transform,
        "bounding_box_transform": json_formatted_bbox_transform,
        "extent": json_formatted_extent,
        "location": json_formatted_location,
        "attributes": agent.attributes,
        "id": agent.id,
        "semantic_tags": agent.semantic_tags,
    }
