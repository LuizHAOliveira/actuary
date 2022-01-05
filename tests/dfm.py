# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:51:23 2021

@author: luizheol
"""

import __init__
from actuary.classes import Triangle, DFMUltimate
from tests.random_triangle import generate_onepertriangle
import numpy as np

if __name__ == '__main__':
    np.random.seed(1)
    infos = generate_onepertriangle(size=24, tail_perc=0.01, ultimate_mean=10000, ultimate_std=2000)
    tri = Triangle(tri_val=infos, dev_per=3, origin_per=3)
    
    dfm = DFMUltimate(tri)
    
    print(dfm.selection_tri)
    print(dfm.periods())
    factors = dfm.calculate_dfm()
    print(factors.get_cdf(0))
