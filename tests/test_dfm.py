from actuary.dfm import DFCalculator
from actuary import Triangle
from tests.random_triangle import generate_triangle
from unittest import TestCase
import numpy as np

class TestDFCalculator(TestCase):
    def test_calculate(self):
       tri: Triangle = generate_triangle(24, 1, 1)
       cum_numerator = np.zeros(tri.shape[1] - 1)
       cum_denominator = np.zeros(tri.shape[1] - 1)
       tri.change_to_cumulative()
       for j in range(tri.months_span[1] - 1):
            
            cum_numerator[j] = tri.values[:(tri.months_span[1]-j-1), j+1].sum()
            cum_denominator[j] = tri.values[:(tri.months_span[1]-j-1), j].sum()
       manual_dfs: np.array = cum_numerator / cum_denominator
       df_calculator = DFCalculator(tri)
       dfs = df_calculator.calculate()

       np.testing.assert_almost_equal(dfs.df.tolist(), manual_dfs.tolist(),
                err_msg=f'DFMCalculator Calculate Error', decimal=3)



if __name__ == '__main__':
    tester: TestDFCalculator = TestDFCalculator()
    tester.test_calculate()

    
