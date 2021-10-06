import __init__
from actuary.classes import AbstractTriangle
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=16, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    #infos = np.delete(infos, [-1, -2, -3], 0)
    tri = AbstractTriangle(tri_val=infos, dev_per=3, origin_per=6)
    
    np.savetxt("one_p_triangle.csv", infos, delimiter=";", fmt='%10.0f')
    np.savetxt("triangle.csv", tri.values, delimiter=";", fmt='%10.0f')
    print(list(map(lambda x: tri._maxcol_index(x), np.arange(5))))