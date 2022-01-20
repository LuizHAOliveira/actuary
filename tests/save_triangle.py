import __init__

from actuary.triangle import Triangle, TriangleFactory
from tests.random_triangle import generate_onepertriangle
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=6, tail_perc=0.01, ultimate_mean=100, ultimate_std=20)
    #infos = np.delete(infos, [-1, -2, -3], 0)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_movement_triangle(2, 2)

    dict_infos = tri.to_json()
    print(dict_infos)
    tri2 = Triangle.from_json(dict_infos)
    print(tri2.to_json())

    folder = Path(__file__).parent / 'files'
    print(folder)
    tri.save_json(name='test_triangle.json', path=folder)

    tri3 = Triangle.load_json(name='test_triangle.json', path=folder)
    print(tri3.to_json())



