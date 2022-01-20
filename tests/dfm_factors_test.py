import __init__
from actuary.triangle import TriangleFactory
from actuary.dfm import DFCalculator
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=24, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    fac = TriangleFactory(base_triangle=infos)
    tri = fac.build_cumulative_triangle(3, 1)
    print('The triangle:')
    print(tri.values)
    print('-----------------')
    dfm = DFCalculator(tri)
    print('The DFs triangle:')
    print(dfm.dfs_triangle)
    print('-----------------')
    #dfm.selection_tri[0, 6] = np.nan
    print('The Selection triangle:')
    print(dfm.selection_tri)
    print('-----------------')
    df_factors = dfm.calculate()
    print('to json:')
    print(df_factors.to_json())
    print('CDFs:')
    print(df_factors.cdf)
    
