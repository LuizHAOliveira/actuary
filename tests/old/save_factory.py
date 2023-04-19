import __init__

from actuary.triangle import TriangleFactory
from tests.random_triangle import generate_onepertriangle
import numpy as np
from pathlib import Path

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=6, tail_perc=0.01, ultimate_mean=100, ultimate_std=20)
    #infos = np.delete(infos, [-1, -2, -3], 0)
    fac = TriangleFactory(base_triangle=infos)
    dict_infos = fac.to_json()
    print(dict_infos)
    fac2 = TriangleFactory.from_json(dict_infos)
    print(fac2.to_json())

    folder = Path(__file__).parent / 'files'
    print(folder)
    fac.json_save(name='test_factory.json', path=folder)

    fac3 = TriangleFactory.json_load(name='test_factory.json', path=folder)
    print(fac3.to_json())



