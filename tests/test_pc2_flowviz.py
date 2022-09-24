import numpy as np

from pc2_flowviz import __version__
from pc2_flowviz.pc2rgb import create_pc2
from std_msgs.msg import Header

def test_version():
    assert __version__ == '0.1.0'

def test_create_pc2():
    ros_header = Header()
    pc2 = create_pc2(np.zeros((100,3)), ros_header)
