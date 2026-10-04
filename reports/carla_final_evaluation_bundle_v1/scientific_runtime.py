"""Byte-identical v4 OOV and runtime functions."""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def map_segmentation_oov_to_frozen_vocabulary(array):
    """Fixed post-failure amendment: channel 0, all OOV -> existing Other=22."""
    import numpy as np
    if not isinstance(array, np.ndarray) or array.dtype != np.uint8 or array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[2] < 1):
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_UINT8_2D_OR_3D_CHANNEL_MAP")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("FROZEN_SEGMENTATION_SCHEMA_REQUIRES_NONEMPTY_SPATIAL_MAP")
    class_map = array[..., 0] if array.ndim == 3 else array
    mapped = class_map.copy()
    mapped[mapped > 28] = 22
    return mapped

def runtime_check():
    import platform
    import numpy as np
    import torch
    import PIL
    import pyarrow
    import pandas
    expected = json.loads((HERE / "runtime_lock.json").read_text())
    actual = {"Python": platform.python_version(), "NumPy": np.__version__, "PyTorch": torch.__version__,
              "Pillow": PIL.__version__, "PyArrow": pyarrow.__version__}
    for key in ("Python", "NumPy", "PyTorch"):
        if actual[key] != expected[key]:
            raise RuntimeError("FROZEN_KAGGLE_RUNTIME_MISMATCH: " + key)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or torch.cuda.get_device_name(0) != "Tesla T4":
        raise RuntimeError("REQUIRES_ONE_VISIBLE_T4")
    if torch.version.cuda != "12.8" or torch.backends.cudnn.version() != 91002:
        raise RuntimeError("FROZEN_CUDA_CUDNN_MISMATCH")
    # Decoder versions are fixed during preparation from verified local schema fixtures.
    for key in ("Pillow", "PyArrow"):
        if actual[key] != expected[key]:
            raise RuntimeError("SEALED_DECODER_VERSION_MISMATCH: " + key)
    if pandas.__version__ != "2.2.3":
        raise RuntimeError("SEALED_PANDAS_VERSION_MISMATCH: pandas==2.2.3")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return actual
