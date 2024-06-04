from actuary.triangle import Triangle, Vector
import numpy as np
from datetime import date
    
def calculate_reserves(ultimate: Vector, tri: Triangle) -> Vector:
    tri.change_to_cumulative()
    ibnr_vals = ultimate.values - tri.get_diagonal().values
    ibnr = Vector(ibnr_vals, ultimate.month_span, ultimate.period)
    return ibnr
