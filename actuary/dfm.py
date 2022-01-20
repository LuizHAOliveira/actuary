from actuary.triangle import HorizontalHeader
from actuary.basic import Ultimate
from actuary.triangle import Triangle

import numpy as np
 
class DFMFactors:
    """ Holds the data structure for DF factors. """
    # Should be able to work with partial months?
    cdf: np.array
    df: np.array
    _tail: float
    month_span: int
    period: int
    header: HorizontalHeader
    @property
    def tail(self):
        return self._tail
    @tail.setter
    def tail(self, new_tail):
        self._tail = new_tail
        self._calculate_cdf()

    def __init__(self, df: np.array, month_span: int, period: int, tail: float = 1.00):
        self.df = df
        self.period = period
        self.month_span = month_span
        self.header = HorizontalHeader(self.period, self.month_span)
        self._tail = tail
        self._calculate_cdf()

    def _calculate_cdf(self):
        self.cdf = np.ones(self.df.size + 1)
        for i in range(len(self.df) + 1):
            self.cdf[i] = np.prod(self.df[i:]) * (1+self._tail)

    def to_json(self):
        return {
            'df': self.df.tolist(),
            'tail': self.tail,
            'month_span': self.month_span,
            'period': self.period,
        }

    
class DFCalculator:
    triangle: Triangle
    dfs_triangle: np.array
    dfm_factors: DFMFactors
    selection_tri: np.array

    def __init__(self, triangle: Triangle):
        self.triangle = triangle
        self._calculate_dfs_triangle()

    def _calculate_dfs_triangle(self):
        shape = self.triangle.shape
        self.triangle.change_to_cumulative()
        tri = self.triangle.values
        self.dfs_triangle = np.full(shape, np.nan)
        for i in range(shape[0]):
            for j in range(self.triangle.maxcol_index(i)):
                self.dfs_triangle[i, j] = tri[i, j+1] / tri[i, j]
        self.selection_tri = self.dfs_triangle / self.dfs_triangle

    def calculate(self) -> DFMFactors:
        self.triangle.change_to_cumulative()
        tri_below = self.triangle.values
        tri_above = np.roll(tri_below, -1, 1)
        df = np.nansum(tri_above * self.selection_tri, 0) / np.nansum(tri_below * self.selection_tri, 0)
        df = df[:-1]
        df[np.isnan(df)] = 1
        return DFMFactors(df, self.triangle.months_span[1], self.triangle.periods[1])
    
def calculate_dfm_ultimate(triangle: Triangle, factors: DFMFactors) -> Ultimate:
    if triangle.development_period != factors.period:
        raise
    dev_ori_ratio = triangle.origin_period / triangle.development_period
    triangle.change_to_cumulative()
    ultimate_val = np.zeros(triangle.shape[0])
    cdfs = np.flip(factors.cdf)
    for i in range(triangle.shape[0]):
        ultimate_val[i] = triangle.get_diagonal()[i] * cdfs[int(i*dev_ori_ratio)]
    return Ultimate(ultimate_val, triangle.months_span[0], triangle.origin_period, triangle.ref_date)


        
        
        
        