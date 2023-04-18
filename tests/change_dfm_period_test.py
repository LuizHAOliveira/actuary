import __init__
from actuary.dfm import DFCalculator, change_DFM_period
from actuary.triangle import TriangleFactory
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=24, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_cumulative_triangle(3, 3)
    tri2 = fac.build_cumulative_triangle(6, 6)
    #tri.to_excel()
    #tri2.to_excel()
    dfm = DFCalculator(tri)
    factors = dfm.calculate()
    print(factors.cdf)
    print(factors.df)
    new_factors = change_DFM_period(factors=factors, new_period=1)
    print(new_factors.cdf)
    print(new_factors.df)