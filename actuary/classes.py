import numpy as np

class AbstractOnePeriodTriangle:
    _oneptri = np.zeros((1, 1))
    def __init__(self, val=np.zeros(0), ori=np.zeros(0), dev=np.zeros(0), tri_val=np.zeros(0), **kwargs):
        if val.size > 0:
            self._load_mov_data(val, ori, dev, **kwargs)
        elif tri_val.ndim == 2:
            self._oneptri = tri_val
        else:
            self._oneptri = np.zeros((kwargs.get('ori_size', 1), kwargs.get('dev_size', 1)))
        self._n_periods = self._oneptri.shape
    def _load_mov_data(self, val: np.array, ori: np.array, dev: np.array, **kwargs):
        if not (val.size == ori.size == dev.size):
            raise
        max_dev = max(ori + dev)
        ori_size = kwargs.get('ori_size', max(ori) + 1)
        dev_size = kwargs.get('dev_size', max_dev + 1)
        self._oneptri = np.zeros((ori_size, dev_size))
        for v, o, d in zip(val, ori, dev):
            if o > ori_size or o + d + 1 > dev_size:
                continue
            self._oneptri[o, d] += v
    def __repr__(self):
        return str(self._oneptri)

class Triangle(AbstractOnePeriodTriangle):
    def _v_header_numbers(self):
        h = np.zeros((self.shape[0], 2))
        for i in range(self.shape[0]):
            h[i, 0] = i * self.origin_per
            h[i, 1] = min((i + 1) * self.origin_per, self._n_periods[0]) - 1
        return h
    def _h_header_numbers(self):
        h = np.zeros((2, self.shape[1]))
        for i in range(self.shape[1]):
            h[0, -(i+1)] = max(self._n_periods[0] - (i + 1) * self.dev_per, 0)
            h[1, -(i+1)] = self._n_periods[0] - i * self.dev_per - 1
        return h
    def header_numbers(self, vertical=True):
        if vertical:
            return self._v_header_numbers()
        else:
            return self._h_header_numbers()
        
    def get_values(self, cumulative=True, calendar=False):
        if cumulative:
            tri = self._internal_accumtri
        else:
            tri = self._internaltri
        if calendar:
            for i in range(self.shape[0]):
                index = (self.shape[1] - self.maxcol_index(i) - 1)
                tri[i, :] = np.roll(tri[i, :], index)
        return tri
    def get_diagonal(self, index=0, cumulative=True):
        tri = self.get_values(cumulative)
        diag = np.zeros(self.shape[0])
        for i in range(diag.size):
            diag[i] = tri[i, self.maxcol_index(i) - index]
        return diag
    def maxcol_index(self, row):
        # n_months = (self.shape[1] - self._correction) * self.dev_per + self._left_overs
        dev_ori_ratio = int(self.origin_per / self.dev_per)
        index = self.shape[1] - row * dev_ori_ratio - 1
        return index
    @property
    def shape(self):
        return self._internaltri.shape

    def __init__(self, val=np.zeros(0), ori=np.zeros(0), dev=np.zeros(0), tri_val=np.zeros(0),
        origin_per=1, dev_per=1, **kwargs):
        super().__init__(val, ori, dev, tri_val, **kwargs)
        self._define_basic_infos(origin_per, dev_per)
        self._construct_triangle()
        self._construct_accum_triangle()
        
    def _define_basic_infos(self, origin_per, dev_per):
        self._left_overs = self._oneptri.shape[1] % dev_per
        self._correction = 1 if self._left_overs > 0 else 0 # If left_overs we must use it to correct the dev months
        self.origin_per = origin_per
        self.dev_per = dev_per
        
    def _construct_triangle(self):
        one_size = self._oneptri.shape
        origin_size = one_size[0] //  self.origin_per + min(one_size[0] % self.origin_per, 1)
        dev_size = one_size[1] //  self.dev_per + min(one_size[1] % self.dev_per, 1)
        self._internaltri = np.zeros((origin_size, dev_size))
        for i in range(one_size[0]):
            for j in range(one_size[1] - i):
                i_new = i // self.origin_per
                rel_months = i % self.origin_per + j
                j_new = (rel_months - self._left_overs) // self.dev_per + self._correction
                self._internaltri[i_new, j_new] += self._oneptri[i, j]
        
    def _construct_accum_triangle(self):
        self._internal_accumtri = np.zeros(self.shape)
        self._internal_accumtri[:, 0] = self._internaltri[:, 0]
        for i in range(self.shape[0]):
            for j in range(1, self.maxcol_index(i) + 1):
                self._internal_accumtri[i, j] = self._internal_accumtri[i, j-1] + self._internaltri[i, j]
    
    def __repr__(self):
        return str(self._internaltri)

class Ultimate:
    def __init__(self):
        pass
    
class DFMFactor:
    def get_cdf(self, i):
        pass
    def __init__(self, factors, periods):
        pass
    

class DFMUltimate(Ultimate):
    @property
    def shape(self):
        return self._dfm_tri.shape
    
    def periods(self):
        tri_h = self.triangle.header_numbers(vertical=False)
        per = np.c_[tri_h, np.array([tri_h[1, -1]+1, np.inf])]
        return per
        
    def __init__(self, triangle: Triangle):
        self.triangle = triangle
        self._calculate_factors()
    
    def _calculate_factors(self):
        shape = self.triangle.shape
        tri = self.triangle.get_values(cumulative=True, calendar=False)
        self._dfm_tri = np.full(shape, np.nan)
        for i in range(shape[0]):
            for j in range(self.triangle.maxcol_index(i)):
                self._dfm_tri[i, j] = tri[i, j+1] / tri[i, j]
        self.selection_tri = self._dfm_tri / self._dfm_tri
    
    def calculate_ultimate(self):
        pass
    
        
        
        
        
        