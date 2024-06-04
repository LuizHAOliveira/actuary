from actuary import Triangle, Vector, calculate_reserves
from actuary.dfm import DFCalculator, calculate_dfm_ultimate
from tests.random_triangle import generate_triangle

if __name__ == '__main__':
    tri: Triangle = generate_triangle(60, 3, 3)
    tri.change_to_cumulative()
    print(tri)
    dfm_calc = DFCalculator(tri)
    print(dfm_calc.selection_tri)
    dfs = dfm_calc.calculate()
    dfs.tail = 0.1

    ult = calculate_dfm_ultimate(tri, dfs)
    print(ult)

    ibnr = calculate_reserves(ult, tri)
    print(ibnr)


