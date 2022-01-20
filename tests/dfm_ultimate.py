import __init__
from actuary.triangle import TriangleFactory
from actuary.dfm import DFCalculator, calculate_dfm_ultimate
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=24, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_cumulative_triangle(3, 1)
    dfm = DFCalculator(tri)
    df_factors = dfm.calculate()
    df_factors.tail = 0.01
    print(df_factors.cdf)
    ult = calculate_dfm_ultimate(tri, df_factors)
    print(df_factors.cdf)
    ws = ult.to_excel()
    tri.to_excel(wb=ws.book)
