"""
Miscellaneous callbacks

"""

from typing import TYPE_CHECKING
import carla
import numpy as np

from .callback import Callback
from .utils import find_next_tf
from carlagen.utils.carla import set_weather
import os
import yaml
from hydra.utils import to_absolute_path


if TYPE_CHECKING:
    from carlagen import Simulation


class ChangeWeatherCallback(Callback):
    """
    Interpolate from current weather to given weather over some period
    """

    def __init__(self, weather, delay=0, duration=1, **kwargs):
        super().__init__(**kwargs)
        self.delay = delay
        self.duration = duration
        self.target_weather = weather
        self._set = False

        self._start_weather = None
        self._transition = {}

    def on_simulation_tick(self, sim: "Simulation", **kwargs):
        if sim.tick_count >= self.delay and sim.tick_count - self.delay < self.duration:
            if self._start_weather is None:
                self._start_weather = sim.world.get_weather()

                for attribute in self.target_weather:
                    v0 = getattr(self._start_weather, attribute)
                    vn = self.target_weather[attribute]

                    self._transition[attribute] = list(
                        np.linspace(v0, vn, self.duration)
                    )

                self.log.debug(f"Weather transition: {self._transition}")

            current = sim.tick_count - self.delay
            current_weather = {
                a: self._transition[a][current] for a in self._transition
            }
            self.log.debug(
                f"Weather transition: {current}/{self.duration} ({current / self.duration:.2%})"
            )
            set_weather(sim.world, current_weather)


class ChangeWeatherTrainingCallback(Callback):
    """
    Interpolate from current weather to given weather over some period.

    Assumes a length of 7200 (2h).

    """

    def __init__(self, fps=10, **kwargs):
        super().__init__(**kwargs)
        self.transition_duration = int(200 * fps)
        self.no_change_duration = int(55 * fps)

        self._set = False
        self._start_weather = None
        self._transition = {}
        self.change_start_times = [
            210,
            465,
            720,
            975,
            1230,
            1485,
            1740,
            1995,
            2250,
            2505,
            2760,
            3015,
            3270,
            3525,
            3780,
            4035,
            4290,
            4545,
            4800,
            5055,
            5310,
            5565,
            5820,
            6075,
            6330,
            6585,
            6840,
            7095,
        ]

        self.change_start_times = np.array(
            fps * np.array(self.change_start_times), dtype=int
        )

        self.presets_folder = to_absolute_path("config/predefined_weather/")

        self.destination_weathers = [
            "rainy_sunrise",
            "dense_fog_sunrise",
            "clear_sunrise",
            "dense_fog_sunrise",
            "rainy_sunrise",
            "clear_sunrise",
            "clear_noon",
            "rainy_noon",
            "dense_fog_noon",
            "clear_noon",
            "dense_fog_noon",
            "rainy_noon",
            "clear_noon",
            "clear_sunset",
            "rainy_sunset",
            "dense_fog_sunset",
            "clear_sunset",
            "dense_fog_sunset",
            "rainy_sunset",
            "clear_sunset",
            "clear_midnight",
            "rainy_midnight",
            "dense_fog_midnight",
            "clear_midnight",
            "dense_fog_midnight",
            "rainy_midnight",
            "clear_midnight",
            "clear_sunrise",
        ]

        self.start_changing_tick = 0

    def weather_is_changing(self, sim, tick):
        """
        Check if the weather is changing

        :param sim: simulation object
        :param tick: current tick
        :return: True if the weather is changing, False otherwise
        """
        for i in range(len(self.change_start_times)):
            if (
                self.change_start_times[i]
                <= tick
                < self.change_start_times[i] + self.transition_duration
            ):
                return True

        return False

    def load_weather(self, weather_name):
        """
        Load the weather from the presets folder

        :param weather_name: name of the weather
        :return: weather object
        """
        # load the weather from the presets folder
        with open(
            os.path.join(
                self.presets_folder,
                f"{weather_name}.yaml",
            ),
            "r",
        ) as f:
            # TODO: this is not a good idea
            # start loading as yaml after 3 lines
            for _ in range(3):
                next(f)
            # load the rest of the file as yaml
            weather = yaml.safe_load(f)

        # create a dictionary with the weather attributes

        return weather

    def weather_change(self, sim):

        # check if is first tick of change
        if sim.tick_count in self.change_start_times:
            # recompute _transition array with weather states
            self.start_changing_tick = sim.tick_count
            # self.change_start_times is np array
            idx = np.where(self.change_start_times == sim.tick_count)[0][0]
            destination_weather_name = self.destination_weathers[idx]
            destination_weather = self.load_weather(destination_weather_name)
            current_weather = sim.world.get_weather()

            # calculate the transition
            for attribute in destination_weather:
                v0 = getattr(current_weather, attribute)
                vn = destination_weather[attribute]
                self._transition[attribute] = list(
                    np.linspace(v0, vn, self.transition_duration + 1)
                )
            self.log.debug(f"Weather transition: {self._transition}")
            self.log.info(
                f"Weather change at tick {sim.tick_count} | idx: {idx} | {len(self._transition[list(self._transition)[0]])} | {destination_weather_name} | {self.change_start_times[idx]}"
            )

        next_tick = sim.tick_count - self.start_changing_tick + 1
        self.log.info(
            f"{sim.tick_count} | Weather transition: {next_tick}/{self.transition_duration} ({next_tick / self.transition_duration:.2%})"
        )

        # get next weather from array
        next_weather = {a: self._transition[a][next_tick] for a in self._transition}

        set_weather(sim.world, next_weather)

    def on_simulation_tick(self, sim: "Simulation", **kwargs):
        # check id in
        if self.weather_is_changing(sim, sim.tick_count):
            # weather change
            self.weather_change(sim)


class RandomDayTimeCallback(Callback):
    """
    Set a random time of day for the simulation on start
    """

    def __init__(self, every_n_frames=1, cb_seed=1234):
        self.cb_seed = cb_seed
        self.rng = np.random.default_rng(self.cb_seed)
        super().__init__(every_n_frames)

    def on_simulation_start(self, sim, **kwargs):
        altitude = self.rng.integers(-90, 90)
        self.log.info(f"Setting random time of day: {altitude} degrees")

        current_weather = sim.world.get_weather()

        new_weather = carla.WeatherParameters(
            cloudiness=current_weather.cloudiness,
            precipitation=current_weather.precipitation,
            precipitation_deposits=current_weather.precipitation_deposits,
            wind_intensity=current_weather.wind_intensity,
            sun_azimuth_angle=current_weather.sun_azimuth_angle,
            sun_altitude_angle=float(altitude),
            fog_density=current_weather.fog_density,
            fog_distance=current_weather.fog_distance,
            wetness=current_weather.wetness,
            fog_falloff=current_weather.fog_falloff,
            scattering_intensity=current_weather.scattering_intensity,
            mie_scattering_scale=current_weather.mie_scattering_scale,
            rayleigh_scattering_scale=current_weather.rayleigh_scattering_scale,
        )

        sim.world.set_weather(new_weather)




class RandomWeatherCallback(Callback):
    """
    Set a random weather for the simulation
    """

    def __init__(self, every_n_frames=1, cb_seed=1234):
        self.cb_seed = cb_seed
        super().__init__(every_n_frames)
        self.rng = np.random.default_rng(self.cb_seed)
        self.presets_folder = to_absolute_path("config/predefined_weather")

    def on_simulation_start(self, sim, **kwargs):
        num_transition_states = 45
        # save current sun_altitude_angle for later
        current_weather = sim.world.get_weather()
        sun_altitude_angle = current_weather.sun_altitude_angle

        # pick two from 3 items
        predefined_weathers = ["clear_noon", "dense_fog_noon", "rainy_noon"]
        weather = self.rng.choice(predefined_weathers, size=2, replace=False)

        # get transition state random, random value between 0 and 45
        transition_state = self.rng.integers(0, num_transition_states)

        self.log.info(
            f"Set Weather of the clip between {weather[0].replace('_noon', '')} and {weather[1].replace('_noon', '')} with transition state {transition_state/num_transition_states:.2f}"
        )

        start_wether = self.load_weather(weather[0])
        destination_weather = self.load_weather(weather[1])
        _transition = {}
        for attribute in destination_weather:
            v0 = start_wether[attribute]
            vn = destination_weather[attribute]
            _transition[attribute] = list(np.linspace(v0, vn, num_transition_states))

        final_weather = {a: _transition[a][transition_state] for a in _transition}

        # set the sun altitude angle to the current one
        final_weather["sun_altitude_angle"] = sun_altitude_angle
        set_weather(sim.world, final_weather)

    def load_weather(self, weather_name):
        """
        Load the weather from the presets folder
        :param weather_name: name of the weather
        :return: weather object
        """
        # load the weather from the presets folder
        with open(
            os.path.join(
                self.presets_folder,
                f"{weather_name}.yaml",
            ),
            "r",
        ) as f:
            # TODO: this is not the best idea
            # start loading as yaml after 3 lines
            for _ in range(3):
                next(f)
            # load the rest of the file as yaml
            weather = yaml.safe_load(f)

        # create a dictionary with the weather attributes

        return weather


class GreenTrafficLightsCallback(Callback):
    """
    Set relevant traffic lights for the ego vehicle to green
    """

    def __init__(self, max_waiting_time=10, max_cars_in_front=5, every_n_frames=1):
        super().__init__(every_n_frames)
        self.max_waiting_time = max_waiting_time
        # carla Location
        self.last_time_ego_stopped = -1
        self.is_waiting = 0
        self.max_cars_in_front = max_cars_in_front

    def on_simulation_tick(self, sim: "Simulation", **kwargs):
        # check if the car is moving: velocity is lower than 0.1 m/s
        if sim.ego_vehicle.get_velocity().length() < 0.1:
            if self.is_waiting == 0:
                self.last_time_ego_stopped = sim.tick_count
            # case car is not moving
            self.is_waiting += 1
            self.log.info(
                f"Tick: {sim.tick_count} | Ego vehicle is waiting for {self.is_waiting} ticks"
            )
        else:
            # case car is moving
            self.is_waiting = 0

        # check if the car is waiting for more than max_waiting_time
        if self.is_waiting > self.max_waiting_time:
            # set traffic light to green
            tf = sim.ego_vehicle.get_traffic_light()
            if tf is None:
                self.log.info(f"Traffic light is None")
                current_vehicle = sim.ego_vehicle
                tf = find_next_tf(
                    current_vehicle, sim, self.log, self.max_cars_in_front
                )
                if tf is None:
                    self.log.info(f"Traffic light is None")
                else:
                    tf.set_state(carla.TrafficLightState.Green)
                    self.log.info(
                        f"Traffic light at {tf.get_transform().location.x} | {tf.get_transform().location.y} | {tf.get_transform().location.z} set to green. Ego vehicle was waiting for {self.is_waiting} ticks. Starting at tick {self.last_time_ego_stopped}"
                    )

            else:
                tf.set_state(carla.TrafficLightState.Green)
                self.log.info(
                    f"Traffic light at {tf.get_transform().location.x} | {tf.get_transform().location.y} | {tf.get_transform().location.z} set to green. Ego vehicle was waiting for {self.is_waiting} ticks. Starting at tick {self.last_time_ego_stopped}"
                )



class ControlCarLightsCallback(Callback):
    def __init__(self, every_n_frames=1):
        super().__init__(every_n_frames)

    def on_simulation_tick(self, sim: "Simulation", **kwargs):

        vehicles = sim.world.get_actors().filter("*vehicles*")

        weather = sim.world.get_weather()

        for vehicle in vehicles:
            if weather.sun_altitude_angle < 0:
                # turn lights on
                light_state_old: int = vehicle.get_light_state()
                light_state = light_state_old | carla.VehicleLightState.LowBeam | carla.VehicleLightState.HighBeam

            else:
                # turn lights off
                light_state_old: carla.VehicleLightState = vehicle.get_light_state()
                light_state = light_state_old & ~carla.VehicleLightState.LowBeam & ~carla.VehicleLightState.HighBeam

            if not light_state_old == light_state:
                vehicle.set_light_state(light_state)