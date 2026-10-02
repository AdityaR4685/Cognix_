import math


def find_next_tf(
    current_vehicle,
    sim,
    log,
    max_cars_in_front=10,
    max_car_distance=25,
    max_distance=None,
):
    tf = current_vehicle.get_traffic_light()
    if tf is not None:
        log.info(
            f"Traffic Light found. Traffic light at {tf.get_transform().location.x} | {tf.get_transform().location.y} | {tf.get_transform().location.z}"
        )
        return tf
    for i in range(max_cars_in_front):
        next_vehicle = find_next_car(current_vehicle, sim, log, max_car_distance)
        if next_vehicle is not None:
            log.info(
                f"Tick: {sim.tick_count} | Next vehicle found: id {next_vehicle.id} ; location {next_vehicle.get_location().x} | {next_vehicle.get_location().y} | {next_vehicle.get_location().z}"
            )

            tf = next_vehicle.get_traffic_light()
            if tf is None:
                log.info(f"Next vehicle found, but traffic light is None")
            else:
                log.info(
                    f"Next vehicle found. Traffic light at {tf.get_transform().location.x} | {tf.get_transform().location.y} | {tf.get_transform().location.z}"
                )
                return tf

        else:
            log.info(f"No vehicle in front of the ego vehicle. Tick: {sim.tick_count}")
            return None
        current_vehicle = next_vehicle

    # check for the distance to the traffic light
    if max_distance is not None:
        if tf is not None:
            distance = current_vehicle.get_location().distance(
                tf.get_transform().location
            )
            if distance > max_distance:
                log.info(
                    f"Traffic light is too far away. Distance: {distance} | Max distance: {max_distance}"
                )
                return None
    return None


def find_next_car(ego_vehicle, sim, log, max_car_distance=25, max_angle=30):
    # get all the vehicles in the simulation
    vehicles = sim.world.get_actors().filter("vehicle.*")
    # get the ego vehicle location
    ego_location = ego_vehicle.get_location()
    # get the ego vehicle forward vector
    ego_forward = ego_vehicle.get_transform().get_forward_vector()

    # get the closest vehicle in front of the ego vehicle
    closest_vehicle = None
    closest_distance = float("inf")
    for vehicle in vehicles:
        # check if the vehicle is the ego vehicle
        if vehicle.id == ego_vehicle.id:
            continue
        # get the vehicle location
        vehicle_location = vehicle.get_location()
        # get the vehicle forward vector
        vehicle_forward = vehicle.get_transform().get_forward_vector()
        # get the distance between the ego vehicle and the other vehicle
        distance = ego_location.distance(vehicle_location)
        # direction vector between the ego vehicle and the other vehicle

        direction = vehicle_location - ego_location

        # scalar product between the ego vehicle forward vector and the direction vector
        if (
            ego_forward.x * direction.x + ego_forward.y * direction.y > 0
        ):  # check if the other vehicle is in front of the ego vehicle
            # get forward vector of the other vehicle
            forward = vehicle.get_transform().get_forward_vector()
            # calculate the angle between the ego vehicle forward vector and the other vehicle forward vector
            angle = angle_between_vectors_2d(ego_forward, forward)
            # check if the angle is smaller than the threshold
            if angle < max_angle:
                if distance < closest_distance:
                    closest_distance = distance
                    closest_vehicle = vehicle

    # check if there is a vehicle in front of the ego vehicle
    if closest_vehicle is not None:
        if closest_distance > max_car_distance:
            log.info(
                f"Vehicle to far away in front of the ego vehicle. Tick: {sim.tick_count}"
            )
            return None
        return closest_vehicle
    else:
        log.info(f"No vehicle in front of the ego vehicle. Tick: {sim.tick_count}")
        return None


def check_if_is_in_front_of_ego_vehicle(sim, actor):
    ego_vehicle = sim.ego_vehicle
    ego_location = ego_vehicle.get_location()
    # get the ego vehicle forward vector
    ego_forward = ego_vehicle.get_transform().get_forward_vector()
    if actor.id == ego_vehicle.id:
        return False
    # get the vehicle location
    vehicle_location = actor.get_location()

    return check_if_is_in_front_of(ego_location, ego_forward, vehicle_location)


def check_if_is_in_front_of(ego_location, ego_forward, other_location):
    # direction vector between the ego vehicle and the other vehicle
    direction = other_location - ego_location
    # scalar product between the ego vehicle forward vector and the direction vector
    return (
        ego_forward.x * direction.x + ego_forward.y * direction.y > 0
    )  # check if the other vehicle is in front of the ego vehicle


def angle_between_vectors_2d(v1, v2) -> float:
    """Berechnet den Winkel zwischen zwei 2D-Vektoren (X,Y) in Grad."""
    dot_product = v1.x * v2.x + v1.y * v2.y
    norm_v1 = math.sqrt(v1.x**2 + v1.y**2)
    norm_v2 = math.sqrt(v2.x**2 + v2.y**2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 180  # Falls einer der Vektoren null ist, ist der Winkel maximal

    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = max(
        -1, min(1, cos_theta)
    )  # Begrenzen, um numerische Fehler zu vermeiden

    return math.degrees(math.acos(cos_theta))
