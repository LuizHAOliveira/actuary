import __init__
from actuary.triangle import TriangleFactory
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=6, tail_perc=0.01, ultimate_mean=100, ultimate_std=20)
    #infos = np.delete(infos, [-1, -2, -3], 0)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_movement_triangle(2, 2)
    print(infos)
    print('\n')
    print(tri.values)
    

    #np.savetxt("one_p_triangle.csv", infos, delimiter=";", fmt='%10.0f')
    #np.savetxt("triangle.csv", tri.values, delimiter=";", fmt='%10.0f')
    #print(list(map(lambda x: tri._maxcol_index(x), np.arange(5))))