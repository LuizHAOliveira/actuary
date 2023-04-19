import __init__
from actuary.triangle import TriangleFactory
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=30, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_movement_triangle(12, 12)
    print(tri.values)
    ws = tri.to_excel()
    wb = ws.book
    tri.toggle_cumulative()
    tri.to_excel(wb)