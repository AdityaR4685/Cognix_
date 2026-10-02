"""
Implements a simple agent that drives the ego vehicle. Requires pytorch.
"""
from typing import Dict, Any

import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import models, transforms

import carla

from carlagen.callback import Callback


class ModelDriverCallback(Callback):
    """
    Callback that disables the ego vehicle's autopilot and drives it using
    a DNN-based model that predicts (throttle, steer, brake) from a camera image.
    """

    def __init__(
        self,
        model_path: str,
        sensor_name: str,
        device: str = "cuda",
        every_n_frames: int = 1,
        **kwargs,
    ):
        super().__init__(every_n_frames=every_n_frames, **kwargs)

        self.model_path = model_path
        self.sensor_name = sensor_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(self.model_path, self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((720, 1280)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self, model_path: str, device: torch.device):
        """
        Construct a ResNet-18 with 3 outputs and load weights from model_path.
        Assumes state_dict was saved from such a model.
        """
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 3)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def _camera_to_tensor(self, img: carla.libcarla.Image) -> torch.Tensor:
        """
        Convert CARLA BGRA image to a normalized tensor suitable for ResNet.
        """
        # raw_data is BGRA uint8
        arr = np.frombuffer(img.raw_data, dtype=np.uint8)
        arr = arr.reshape((img.height, img.width, 4))

        # BGRA -> RGB
        rgb = arr[:, :, :3][:, :, ::-1]

        pil_img = Image.fromarray(rgb, mode="RGB")
        tensor = self.transform(pil_img)  # [3, H, W]
        return tensor

    def on_simulation_start(self, sim, **kwargs):
        """
        Disable autopilot on the ego vehicle at the beginning.
        """
        if hasattr(sim, "ego_vehicle") and sim.ego_vehicle is not None:
            sim.ego_vehicle.set_autopilot(enabled=False)
            self.log.info("Autopilot disabled for ego vehicle.")
        else:
            self.log.warning("Simulation has no ego_vehicle attribute.")

    def on_simulation_tick(
        self, sim, measurements: Dict[str, Any], **kwargs
    ):
        """
        At each tick (respecting every_n_frames), read the camera sensor,
        run the model, and apply the control to the ego vehicle.
        """
        self.log.info(f"Driving ego vehicle at tick {sim.tick_count}")
        # if sim.tick_count % self.every_n_frames != 0:
        #     return

        if not hasattr(sim, "ego_vehicle") or sim.ego_vehicle is None:
            self.log.warning("No ego vehicle available; skipping control.")
            return

        if self.sensor_name not in measurements:
            self.log.warning(
                f"Sensor '{self.sensor_name}' not found in measurements; skipping control."
            )
            return

        data = measurements[self.sensor_name]
        if not isinstance(data, carla.libcarla.Image):
            self.log.warning(
                f"Sensor '{self.sensor_name}' is not a camera image ({type(data)}); skipping control."
            )
            return

        # Convert image to tensor
        with torch.no_grad():
            img_tensor = self._camera_to_tensor(data).unsqueeze(0).to(self.device)
            pred = self.model(img_tensor)[0]  # shape: [3]

        # Predicted actions: [throttle, steer, brake]
        throttle = float(pred[0])
        steer = float(pred[1])
        brake = float(pred[2])

        # Simple clamping to valid CARLA ranges
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        brake = 1 if brake > 0.5 else 0
        steer = float(np.clip(steer, -1.0, 1.0))

        control = carla.VehicleControl(
            throttle=throttle,
            steer=steer,
            brake=brake,
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0,
        )

        sim.ego_vehicle.set_autopilot(enabled=False)
        sim.ego_vehicle.apply_control(control)

        self.log.info(
            f"Tick {sim.tick_count}: "
            f"throttle={throttle:.3f}, steer={steer:.3f}, brake={brake:.3f}"
        )
