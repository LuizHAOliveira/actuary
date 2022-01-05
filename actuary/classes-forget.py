import numpy as np
from datetime import date
from actuary.errors import ArrayDifferentSizesError
import xlwings as xw

_REFDATE = date(2000, 1, 1)

class Header:
    """ Control periods """
    vertical: bool
    month_size: int
    period: int
    def __init__(self, vertical: int, month_size: int,
                period: int = 1) -> None:
        self.vertical = vertical
        self.month_size = month_size
        self.period = period
        self._calculate
    def _calculate(self) -> None:
        """ Calculate the internal variables """
        pass
    def display(self) -> np.array:
        """ Display the header in a 2d np.array """
        return self._internalvalues

class Triangle(np.array):
    """ Basic triangle class """
    values: np.array
    v_header: Header
    h_header: Header

    def __init__(self) -> None:
        pass
    def get_diagonal(self, index: int) -> np.array:
        """ Get the diagonal with index starting at 0 (most recent) """
        pass
    def to_excel(self) -> xw.Sheet:
        """ Display the triangle into an Excel sheet """
        pass

class TriangleFactory:
    """ Class used to build triangles in anyway we want. """
    ref_date: date = _REFDATE
    internal_triangle: np.array
    def build_cumulative(self) -> Triangle:
        """ Build the cumulative form of the triangle """
        pass
    def build_movements(self) -> Triangle:
        """ Build the movements form of the triangle """
        pass