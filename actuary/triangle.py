from actuary.errors import ArrayDifferentSizesError, InvalidPeriodCombinationError

import numpy as np
import xlwings as xw
from datetime import date
from dateutil.relativedelta import relativedelta

_DEFAULT_DATE = date(2000, 1, 1)

class Header:
    """ Class to manipulate the periods of time. Abstract class. """
    period: int
    month_span: int
    ref_date: date
    h_numbers: np.array
    h_dates: np.array
    def __init__(self, period: int, month_span: int, ref_date: date = _DEFAULT_DATE) -> None:
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

class Triangle: # Should be divided into 2 classes, cumulative and movement? IDK
    """ Basic data structure. Should NEVER be directly instantiated, use a factory instead or load from json. """
    values: np.array
    months_span: list # (ori, dev)
    ref_date: date
    periods: list # (ori, dev)
    v_header: VerticalHeader
    h_header: HorizontalHeader
    cumulative: bool
    @property
    def shape(self) -> tuple:
        return self.values.shape
    @property
    def origin_period(self) -> int:
        return self.periods[0]
    @property
    def development_period(self) -> int:
        return self.periods[1]
    def maxcol_index(self, row) -> int:
        dev_ori_ratio = int(self.periods[0] / self.periods[1])
        index = self.shape[1] - row * dev_ori_ratio - 1
        return index + 1

    def __init__(self, values: list,
            months_span: list,
            periods: list,
            cumulative: bool,
            ref_date: date = date(2000, 1, 1)) -> None:
        self.values = np.array(values)
        self.periods = periods
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
            'type': 'Triangle',
            }
    def to_excel(self, wb: xw.Book = None, ws: xw.Sheet = None) -> xw.Sheet:
        if not ws and not wb:
            wb = xw.Book()
            ws = wb.sheets.add()
        elif not ws:
            ws = wb.sheets.add()
        ws.range('D1').value = self.h_header.h_numbers
        ws.range('A4').value = self.v_header.h_dates
        ws.range('D4').value = self.values
        return ws

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
                raise # Should put an error here
            tri[o, d] += v
        return cls(tri, kwargs.get('ref_date', _DEFAULT_DATE))

    def __init__(self, base_triangle: np.array, ref_date: date = _DEFAULT_DATE):
        """ The 'true' constructor is the basic movement triangle. """
        self.base_triangle = base_triangle
        self.ref_date = ref_date
    def _check_valid_periods(self, origin_per, dev_per) -> bool:
        return origin_per % dev_per == 0

    def build_movement_triangle(self, origin_period: int, dev_period: int) -> Triangle:
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
                tri_values[i_new, j_new] += self.base_triangle[i, j]
        return Triangle(tri_values, self.base_triangle.shape, (origin_period, dev_period), 0, self.ref_date)

    def build_cumulative_triangle(self, origin_period: int, dev_period: int) -> Triangle:
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


