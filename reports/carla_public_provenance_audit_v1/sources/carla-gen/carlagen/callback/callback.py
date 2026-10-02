"""
Callback api
"""

import logging
from typing import TYPE_CHECKING, Dict, Any

if TYPE_CHECKING:
    from carlagen import Simulation


class Callback(object):
    """ """

    def __init__(self, every_n_frames=1, **kwargs):
        self.log = logging.getLogger(type(self).__name__)
        self.every_n_frames = every_n_frames
        pass

    def on_simulation_start(self, sim: "Simulation", **kwargs):
        pass

    def on_simulation_tick_starts(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        pass

    def on_simulation_tick(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        pass

    def on_simulation_tick_ends(
        self, sim: "Simulation", measurements: Dict[str, Any], **kwargs
    ):
        pass

    def on_simulation_end(self, sim: "Simulation", **kwargs):
        pass
