
from actuary.triangle import VerticalHeader
import numpy as np
from datetime import date
import datetime
import xlwings as xw
class Ultimate:
    """ Data structure for holding ultimate losses. """
    values: np.array
    month_span: int
    period: int
    ref_date: date
    header: VerticalHeader
    def __init__(self, values: np.array, month_span: int, period: int, ref_date: date):
        self.values = values
        self.month_span = month_span
        self.period = period
        self.ref_date = ref_date
        self.header = VerticalHeader(period, month_span, ref_date)
    
    @classmethod
    def from_json(cls, json_info: dict):
        if not isinstance(json_info['ref_date'], date):
            json_info['ref_date'] = datetime.strptime(json_info['ref_date'], '%Y-%m').date()
        return cls(**json_info)
    def to_json(self) -> dict:
        return {'ref_date': self.ref_date.strftime('%Y-%m'),
            'period': list(self.period),
            'month_span': list(self.month_span),
            'values': self.values.tolist(),
            'type': 'Ultimate',
            }
    def to_excel(self, wb: xw.Book = None, ws: xw.Sheet = None) -> xw.Sheet:
        if not ws and not wb:
            wb = xw.Book()
            ws = wb.sheets.add()
        elif not ws:
            ws = wb.sheets.add()
        ws.range('A4').value = self.header.h_dates
        ws.range('D4').options(transpose=True).value = self.values
        return ws