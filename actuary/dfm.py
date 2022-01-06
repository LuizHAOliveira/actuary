from actuary.basic import Ultimate
from actuary.triangle import Triangle

import numpy as np
 
class DFMFactor:
    def get_cdf(self, i):
        return np.prod(self.df[i:])
    def get_df(self, i):
        return self.df[i]
    def __init__(self, factors, periods):
        self.df = factors
        self.periods = periods
    def __repr__(self) -> str:
        return self.df.__repr__()
    

class DFMUltimate(Ultimate):
    @property
    def shape(self):
        return self._dfm_tri.shape
    
    def header_numbers(self, vertical=False):
        return self.triangle.header_numbers(vertical=vertical)
        
    def periods(self):
        tri_h = self.triangle.header_numbers(vertical=False)
        per = np.c_[tri_h, np.array([tri_h[1, -1]+1, np.inf])]
        return per

    def get_factors(self):
        return self._dfm_tri

    def __init__(self, triangle: Triangle):
        self.triangle = triangle
        self._calculate_factors_triangle()
    
    def _calculate_factors_triangle(self):
        shape = self.triangle.shape
        tri = self.triangle.get_values(cumulative=True, calendar=False)
        self._dfm_tri = np.full(shape, np.nan)
        for i in range(shape[0]):
            for j in range(self.triangle.maxcol_index(i)):
                self._dfm_tri[i, j] = tri[i, j+1] / tri[i, j]
        self.selection_tri = self._dfm_tri / self._dfm_tri
    
    def calculate_dfm(self):
        tri_below = self.triangle.get_values(cumulative=True, calendar=False)
        tri_above = np.roll(tri_below, -1, 1)
        dfs = np.nansum(tri_above * self.selection_tri, 0) / np.nansum(tri_below * self.selection_tri, 0)
        dfs[-1] = 1
        self.factors = DFMFactor(dfs, self.header_numbers())
        return self.factors

    def calculate_ultimate(self):
        pass
    
        
        
        
        
        