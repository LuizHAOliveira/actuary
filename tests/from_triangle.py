import __init__
from actuary.classes import AbstractTriangle
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=15, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    tri = AbstractTriangle(tri_val=infos, dev_per=6, origin_per=12)
    
    np.savetxt("one_p_triangle.csv", infos, delimiter=";", fmt='%10.0f')
    np.savetxt("triangle.csv", tri.values, delimiter=";", fmt='%10.0f')
    print(tri._internaltri)