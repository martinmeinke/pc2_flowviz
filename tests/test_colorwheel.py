from typing import Tuple

import numpy as np
from numpy.random import randn
from pc2_flowviz.colorwheel import flow_to_rgb

from matplotlib import pyplot as plt


def test_colorwheel():
    shape_flat: Tuple[int, int] = (1000, 3)
    shape_batched: Tuple[int, int] = (1000, 3)

    flow = randn(*shape_flat)
    flow_batch = randn(*shape_batched)

    rgb_flat = flow_to_rgb(flow, flow_max_radius=1.0)
    rgb_batched = flow_to_rgb(flow_batch, flow_max_radius=1.0)
    assert rgb_flat.shape == shape_flat
    assert rgb_batched.shape == shape_batched

    assert np.all(np.logical_and(rgb_flat >= 0, rgb_flat <= 255.0))
    assert np.all(np.logical_and(rgb_batched >= 0, rgb_batched <= 255.0))

    # plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.scatter(flow_batch[..., 0], flow_batch[..., 1], c=rgb_batched / 255.0)
    # plt.savefig("flow.png")
