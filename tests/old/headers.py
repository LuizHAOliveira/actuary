import __init__
from actuary.triangle import HorizontalHeader, VerticalHeader

if __name__ == '__main__':
    h = HorizontalHeader(12, 30)
    v = VerticalHeader(12, 30)
    print(h.h_numbers)
    print(h.h_dates)
    print(v.h_numbers)
    print(v.h_dates)