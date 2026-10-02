import logging
import hydra
import time
import os
from os.path import join, exists

from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from carlagen import Simulation
import platform

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default.yaml", version_base="1.2")
def my_app(cfg):
    """

    :param cfg: hydra configuration
    :return:
    """
    if cfg.print_config:
        log.info(OmegaConf.to_yaml(cfg))

    OmegaConf.set_readonly(cfg, True)

    success = False

    while not success:
        try:
            log.info(os.getcwd())
            sim = Simulation(cfg.simulation, root=os.getcwd())

            for cb in cfg.callbacks:
                callback = hydra.utils.instantiate(cfg.callbacks[cb])
                sim.add_callback(callback)

            sim.setup()

            for s in cfg.sensors:
                sensor = hydra.utils.instantiate(cfg.sensors[s], sim=sim)
                sim.add_sensor(sensor)

            sim.run()
            log.info("Finished")
            success = True
        except Exception as e:
            log.exception(e)
            log.info(f"Retrying")
            success = True

    if cfg.kill_carla:
        # NOTE: this is a quick fix
        if platform.system() == "Linux":
            os.system("killall CarlaUE4-Linux-Shipping")
        elif platform.system() == "Windows":
            os.system("taskkill /IM CarlaUE4-Win64-Shipping.exe /F")
        else:
            print("Unsupported platform")

        time.sleep(10)

        if platform.system() == "Linux":
            os.system("killall CarlaUE4-Linux-Shipping")
        elif platform.system() == "Windows":
            os.system("taskkill /IM CarlaUE4-Win64-Shipping.exe /F")
        else:
            print("Unsupported platform")

        time.sleep(5)


if __name__ == "__main__":
    my_app()
