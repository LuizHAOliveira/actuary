import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from actuary.classes import AbstractTriangle
import numpy as np

if __name__ == '__main__':
    values = np.array([1, 2, 4, 7])
    origin = np.array([0, 0, 1, 2])
    dev = np.array([0, 2, 1, 0])
    tri = AbstractTriangle(values, origin, dev, ori_size=3, dev_size=3)
    print(tri._oneptri)