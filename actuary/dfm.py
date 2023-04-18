from hashlib import new
from actuary.triangle import HorizontalHeader
from actuary.basic import Ultimate
from actuary.triangle import Triangle

import numpy as np
 
class DFMFactors:
    """ Holds the data structure for DF factors. """
    # Should be able to work with partial months?
    # Update: Be able to change the period of the factors (approximation)
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

    def __init__(self, df: np.array, month_span: int, period: int, tail: float = 0.0):
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
    selection_tri: np.array

    def __init__(self, triangle: Triangle) -> None:
        self.triangle = triangle
        self._calculate_dfs_triangle()

    def _calculate_dfs_triangle(self) -> None:
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

def __change_DFM_one_period(factors: DFMFactors, new_period: int) -> np.array:
    """ Subfunction of 'change_DFM_period' to get the DFs for one period """
    infos = factors.to_json()
    original_header = factors.header
    months = np.diff(original_header.h_numbers, axis=0)[0, :-1] + 1 # The last one is the tail
    raw_one_p_dfs = infos['df'] ** (1/months) # Not structured
    one_p_dfs = np.zeros(original_header.h_numbers[1, -2] + 1)

    accum_num_months = 0
    for new_df, num_months in zip(raw_one_p_dfs, months):
        one_p_dfs[accum_num_months:accum_num_months+num_months] = new_df
        accum_num_months += num_months
    return one_p_dfs

def __reunite_DFM(one_p_dfs, new_period, month_span):
    """ Subfunction of 'change_DFM_period' to join one periods DFs into correct time period """
    new_header = HorizontalHeader(new_period, month_span)
    print(new_header.h_numbers)
    months = np.diff(new_header.h_numbers, axis=0)[0] + 1 # The last one is the tail
    new_dfs = np.zeros(months.size)
    accum_num_months = 0
    for i, num_months in enumerate(months):
        new_dfs[i] = np.product(one_p_dfs[accum_num_months:accum_num_months+num_months])
        accum_num_months += num_months
    print(one_p_dfs)
    print(new_dfs)
    return new_dfs

def change_DFM_period(factors: DFMFactors, new_period: int) -> DFMFactors:
    """
        Return a new DFMFactors with different periods. If new_period > current_period => Loses info
        Step 1: Turn into a one period.
        Step 2: Get the desired period.
    """
    one_p_dfs = __change_DFM_one_period(factors, new_period)
    new_dfs = __reunite_DFM(one_p_dfs, new_period, factors.month_span)
    return DFMFactors(new_dfs, factors.month_span, new_period, factors.tail)

        
        
        