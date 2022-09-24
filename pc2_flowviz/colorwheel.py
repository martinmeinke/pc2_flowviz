from matplotlib.colors import hsv_to_rgb
from typing import Any, Optional

import numpy as np
from nptyping import Float, NDArray, Shape, UInt8


def flow_to_rgb(
    flow: NDArray[Shape["*, 3"], Float],
    flow_max_radius: Optional[float] = None,
    background: Optional[str] = "white",
) -> NDArray[Any, UInt8]:
    """Creates a RGB representation of 2d scene flow
    Args:
        flow: scene flow.
            flow[..., 0] should be the x-displacement
            flow[..., 1] should be the y-displacement
            flow[..., 2] should be the z-displacement
        flow_max_radius: Set the radius that gives the maximum color intensity.
            Default: The normalization is based on the input flow maximum radius.
        background: States if zero-valued flow is 'black' or 'white'
    Returns: shape like input, last dimension = RGB colors.
    """

    rx: NDArray[Shape["*, 3"], Float] = flow[..., 0]
    ry: NDArray[Shape["*, 3"], Float] = flow[..., 1]
    v_abs: NDArray[Any, Float] = (rx**2.0 + ry**2.0) ** 0.5

    if flow_max_radius is None:
        flow_max_radius = np.max(v_abs)

    s = np.clip(v_abs / flow_max_radius, 0.0, 1.0)

    h: NDArray[Any, Float] = ((np.arctan2(ry, rx) / np.pi) + 1.0) / 2.0

    if background == "white":
        v = np.ones_like(h)
    elif background == "black":
        v = np.zeros_like(h)
    else:
        raise RuntimeError(f"Background {background} not supported.")

    rgb = (hsv_to_rgb(np.stack((h, s, v), axis=-1)) * 255.0).astype(np.uint8)

    return rgb
