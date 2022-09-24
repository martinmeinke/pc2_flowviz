import logging
import pathlib
from importlib import reload
from typing import Dict, List

import numpy as np
import rosbag
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

from pc2_flowviz.colorwheel import flow_to_rgb

PC2_FIELDS: List[PointField] = [
    PointField("x", 0, PointField.FLOAT32, 1),
    PointField("y", 4, PointField.FLOAT32, 1),
    PointField("z", 8, PointField.FLOAT32, 1),
    PointField("vx", 12, PointField.FLOAT32, 1),
    PointField("vy", 16, PointField.FLOAT32, 1),
    PointField("vz", 20, PointField.FLOAT32, 1),
    PointField("rgba", 24, PointField.UINT32, 1),
    PointField("row", 28, PointField.INT32, 1),
    PointField("col", 32, PointField.INT32, 1),
    PointField("cost", 36, PointField.FLOAT32, 1),
]


def create_pc2(coordinates, ros_header, **kwargs) -> PointCloud2:
    points = coordinates

    if "flow" in kwargs and kwargs["flow"] is not None:
        flow = kwargs["flow"]
        colors = flow_to_rgb(flow, flow_max_radius=0.5)
    else:
        flow = np.zeros_like(coordinates)
        colors = np.zeros_like(coordinates, dtype=np.uint8)

    rgbas = np.frombuffer(
        np.hstack((colors, 255 * np.ones((points.shape[0], 1), dtype=np.uint8))),
        dtype=np.uint32,
    ).reshape(-1, 1)

    if "rowcol" in kwargs and kwargs["rowcol"] is not None:
        rowcol = kwargs["rowcol"]
    else:
        rowcol = np.zeros((points.shape[0], 2), dtype=np.uint32)

    if "cost" in kwargs and kwargs["cost"] is not None:
        cost = kwargs["cost"].reshape(-1, 1)
    else:
        cost = np.zeros((points.shape[0], 1), dtype=np.float32)

    data_arrays = [points, flow, rgbas, rowcol, cost]
    assert all(x.shape[0] == data_arrays[0].shape[0] for x in data_arrays)
    points_list = []
    for nth_pt in range(len(points)):
        point_data = []
        for da in data_arrays:
            point_data.extend(da[nth_pt])
        points_list.append(point_data)
        assert len(point_data) == len(PC2_FIELDS)
    pc2 = point_cloud2.create_cloud(ros_header, PC2_FIELDS, points_list)

    return pc2


class RosbagWriter:
    def __init__(self) -> None:
        self._out_bag = None
        rospy.init_node("create_cloud_xyzrgb", disable_rosout=True)
        reload(logging)
        self._publishers: Dict[str, rospy.Publisher] = {}

    def start_new_bag(self, bag_name: str) -> None:
        if self._out_bag is not None:
            self._out_bag.close()
        p = pathlib.Path(bag_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        self._out_bag = rosbag.Bag(bag_name, "w")

    def add(
        self,
        topic_name: str,
        pc: PointCloud2,
        flow=None,
        cost=None,
        t: int = None,
        write=False,
    ) -> None:
        header = Header()
        header.frame_id = "map"
        pc2 = create_pc2(pc[:, :3], header, flow=flow, cost=cost)

        if not rospy.is_shutdown():
            self._publishers.setdefault(
                topic_name, rospy.Publisher(topic_name, PointCloud2, queue_size=2)
            )
            publisher = self._publishers[topic_name]
            pc2.header.stamp = rospy.Time(t)
            publisher.publish(pc2)
            if write and self._out_bag:
                self._out_bag.write(publisher.name, pc2, rospy.Time(t))
