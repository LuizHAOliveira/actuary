from typing import Dict
import numpy as np
from actuary.errors import ArrayDifferentSizesError, InvalidPeriodCombinationError
import xlwings as xw
from datetime import date
from dateutil.relativedelta import relativedelta

class Header:
    """ Class to manipulate the periods of time. Abstract class. """
    period: int
    month_span: int
    ref_date: date
    h_numbers: np.array
    h_dates: np.array
    def __init__(self, period: int, month_span: int, ref_date: date = date(2000, 1, 1)) -> None:
        self.period = period
        self.month_span = month_span
        self.ref_date = ref_date
        self._calculate()
    def _calculate(self):
        raise NotImplementedError

class VerticalHeader(Header):
    def _calculate(self) -> None:
        shape = self.month_span // self.period + (self.month_span % self.period > 0)
        h_numbers = np.zeros((shape, 2))
        h_dates = np.zeros((shape, 2), dtype='object')
        for i in range(shape):
            h_numbers[i, 0] = i * self.period
            h_numbers[i, 1] = min((i + 1) * self.period, self.month_span) - 1
            h_dates[i, 0] = (self.ref_date + relativedelta(months=h_numbers[i, 0])).strftime('%Y-%m')
            h_dates[i, 1] = (self.ref_date + relativedelta(months=h_numbers[i, 1])).strftime('%Y-%m')
        self.h_numbers = h_numbers
        self.h_dates = h_dates
    def __repr__(self) -> str:
        return f"VerticalHeader(period={self.period}, month_span={self.month_span}, ref_date={self.ref_date})"

class HorizontalHeader(Header):
    def _calculate(self) -> None:
        shape = self.month_span // self.period + (self.month_span % self.period > 0)
        h_numbers = np.zeros((2, shape))
        h_dates = np.zeros((2, shape), dtype='object')
        for i in range(shape):
            h_numbers[0, -(i+1)] = max(self.month_span - (i + 1) * self.period, 0)
            h_numbers[1, -(i+1)] = self.month_span - i * self.period - 1
            h_dates[0, -(i+1)] = (self.ref_date + relativedelta(months=h_numbers[0, -(i+1)])).strftime('%Y-%m')
            h_dates[1, -(i+1)] = (self.ref_date + relativedelta(months=h_numbers[1, -(i+1)])).strftime('%Y-%m')
        self.h_numbers = h_numbers
        self.h_dates = h_dates
    def __repr__(self) -> str:
        return f"HorizontalHeader(period={self.period}, month_span={self.month_span}, ref_date={self.ref_date})"

class NTriangle: # Should be divided into 2 classes, cumulative and movement? IDK
    """ Basic data structure. Should NEVER be directly instantiated, use a factory instead or load from json. """
    values: np.array
    months_span: list # (ori, dev)
    ref_date: date
    periods: list # (ori, dev)
    v_header: VerticalHeader
    h_header: HorizontalHeader
    cumulative: bool
    @property
    def shape(self):
        return self.value.shape
    def maxcol_index(self, row):
        dev_ori_ratio = int(self.periods[0] / self.periods[1])
        index = self.shape[1] - row * dev_ori_ratio - 1
        return index

    def __init__(self, values: list,
            months_span: list,
            periods: list,
            cumulative: bool,
            ref_date: date = date(2000, 1, 1)) -> None:
        self.values = np.array(values)
        self.months_span = months_span
        self.v_header = VerticalHeader(periods[0], months_span[0], ref_date)
        self.h_header = HorizontalHeader(periods[1], months_span[1], ref_date)
        self.ref_date = ref_date
        self.cumulative = cumulative
    def get_diagonal(self, index: int=0):
        tri = self.values
        diag = np.zeros(self.shape[0])
        for i in range(diag.size):
            diag[i] = tri[i, self.maxcol_index(i) - index]
        return diag
    def toggle_cumulative(self) -> None:
        if not self.cumulative:
            self._change_to_cumulative()
        else:
            self._change_to_movement()
    def _change_to_cumulative(self) -> None:
        new_tri = np.zeros(self.values.shape)
        for i in range(self.values.shape[0]):
            for j in range(self.maxcol_index(i)):
                new_tri[i, j] = new_tri[i, j-1] + self.values[i, j]
        self.values = new_tri
        self.cumulative = True
    def _change_to_movement(self) -> None:
        new_tri = np.zeros(self.values.shape)
        new_tri[:, 0] = self.values[:, 0]
        new_tri[:, 1:] = np.diff(self.values, axis=1)
        self.values = new_tri
        self.cumulative = False

    def save_json(self, name: str, path: str='.') -> None:
        pass
    @classmethod
    def load_json():
        pass
    def from_json(self, json_info: dict) -> None:
        self.__init__(**json_info)
    def to_json(self) -> dict:
        return {'ref_date': self.ref_date.strftime('%Y-%m'),
            'periods': list(self.periods),
            'months_span': list(self.months_span),
            'cumulative': self.cumulative,
            'values': self.values.tolist(),
            'type': 'NTriangle',
            }
    def to_excel(self, wb: xw.Book = None, ws: xw.Sheet = None) -> xw.Sheet:
        pass

class TriangleFactory:
    """ Used to build triangles. """
    base_triangle: np.array
    ref_date: date
    @classmethod
    def from_movement_data(cls, val: np.array, ori: np.array, dev: np.array, **kwargs):
        """ Constructor when you have movements relative to a date. """
        if not (val.size == ori.size == dev.size):
            raise ArrayDifferentSizesError
        max_dev = max(ori + dev)
        ori_size = kwargs.get('ori_size', max(ori) + 1)
        dev_size = kwargs.get('dev_size', max_dev + 1)
        tri = np.zeros((ori_size, dev_size))
        for v, o, d in zip(val, ori, dev):
            if o > ori_size or o + d + 1 > dev_size:
                continue # Should put an error here
            tri[o, d] += v
        cls(tri, kwargs.get('ref_date', date(2000, 1, 1)))

    def __init__(self, base_triangle: np.array, ref_date: date):
        """ The 'true' constructor is the basic movement triangle. """
        self.base_triangle = base_triangle
        self.ref_date = ref_date
    def _check_valid_periods(self, origin_per, dev_per) -> bool:
        return origin_per % dev_per == 0

    def build_movement_triangle(self, origin_period: int, dev_period: int) -> NTriangle:
        if not self._check_valid_periods(origin_period, dev_period):
            raise InvalidPeriodCombinationError(origin_period, dev_period)
        one_size = self.base_triangle.shape
        origin_size = one_size[0] //  origin_period + min(one_size[0] % origin_period, 1)
        dev_size = one_size[1] //  dev_period + min(one_size[1] % dev_period, 1)
        tri_values = np.zeros((origin_size, dev_size))
        left_overs = self.base_triangle.shape[1] % dev_period
        correction = 1 if left_overs > 0 else 0
        for i in range(one_size[0]):
            for j in range(one_size[1] - i):
                i_new = i // origin_period
                rel_months = i % origin_period + j
                j_new = (rel_months - left_overs) // dev_period + correction
                tri_values[i_new, j_new] += self.tri_values[i, j]
        return NTriangle(tri_values, self.base_triangle.shape, (origin_period, dev_period), 0, self.ref_date)

    def build_cumulative_triangle(self, origin_period: int, dev_period: int) -> NTriangle:
        tri = self.build_movement_triangle(origin_period, dev_period)
        tri.toggle_cumulative()
        return tri
    
    def json_save(self, name: str, path: str='.') -> None:
        pass
    @classmethod
    def json_load(cls, name: str, path: str='.'):
        pass
    @classmethod
    def from_json(cls, json_info: dict) -> None:
        cls(**json_info)
    def to_json(self) -> dict:
        return {'base_triangle': self.base_triangle,
            'ref_date': self.ref_date,
            }

# --------------------
class OnePeriodTriangle:
    """The basic data structure. Triangles should be saved in this level.
    Should not be directly instantiated by the user."""
    _oneptri = np.zeros((1, 1))
    def __init__(self, val: np.array=np.zeros(0), # Rework me into a triangle factory please!
            ori: np.array=np.zeros(0),
            dev: np.array=np.zeros(0),
            tri_val: np.array=np.zeros(0), **kwargs):
        if val.size > 0:
            self._load_mov_data(val, ori, dev, **kwargs)
        elif tri_val.ndim == 2:
            self._oneptri = tri_val
        else:
            self._oneptri = np.zeros((kwargs.get('ori_size', 1), kwargs.get('dev_size', 1)))
        self._n_periods = self._oneptri.shape
    def _load_mov_data(self, val: np.array, ori: np.array, dev: np.array, **kwargs):
        if not (val.size == ori.size == dev.size):
            raise ArrayDifferentSizesError
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

class Triangle(OnePeriodTriangle):
    """ Class representing a triangle with a certain shape. """
    def vertical_header_numbers(self):
        h = VerticalHeader(self.origin_per, self._n_periods[0])
        #h = np.zeros((self.shape[0], 2))
        #for i in range(self.shape[0]):
        #    h[i, 0] = i * self.origin_per
        #    h[i, 1] = min((i + 1) * self.origin_per, self._n_periods[0]) - 1
        return h

    def horizontal_header_numbers(self):
        h = HorizontalHeader(self.origin_per, self._n_periods[0])
        #h = np.zeros((2, self.shape[1]))
        #for i in range(self.shape[1]):
        #    h[0, -(i+1)] = max(self._n_periods[0] - (i + 1) * self.dev_per, 0)
        #    h[1, -(i+1)] = self._n_periods[0] - i * self.dev_per - 1
        return h

    def header_numbers(self, vertical=True):
        if vertical:
            return self.vertical_header_numbers()
        else:
            return self.horizontal_header_numbers()

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

    def __init__(self, val: np.array=np.zeros(0), # Muitos argumentos. Transformar em factory
            ori: np.array=np.zeros(0),
            dev: np.array=np.zeros(0),
            tri_val=np.zeros(0),
            origin_per: int=1,
            dev_per: int=1,
            **kwargs):
        super().__init__(val, ori, dev, tri_val, **kwargs)
        self._define_basic_infos(origin_per, dev_per)
        self._construct_triangle()
        self._construct_accum_triangle()
        
    def _define_basic_infos(self, origin_per, dev_per):
        """ Define internal variables. """
        self._left_overs = self._oneptri.shape[1] % dev_per
        self._correction = 1 if self._left_overs > 0 else 0 # If left_overs we must use it to correct the dev months
        self.origin_per = origin_per
        self.dev_per = dev_per
        
    def _construct_triangle(self):
        """ From the one period triangle, build the movement internal. """
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
        """ From the movement internal triangle, build the cummulative form. """
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
    
        
        
        
        
        