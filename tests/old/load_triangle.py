#import os
#import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import __init__

from actuary.triangle import TriangleFactory
import numpy as np

if __name__ == '__main__':
    values = np.array([1, 2, 4, 7, 9, 10, 11])
    origin = np.array([0, 0, 1, 2, 0, 1, 5])
    dev = np.array([0, 2, 1, 0, 11, 10, 0])
    fac = TriangleFactory.from_movement_data(values, origin, dev, ori_size=6, dev_size=12)
    tri = fac.build_movement_triangle(2, 3)
    print(tri.values)