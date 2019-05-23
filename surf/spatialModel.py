# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|     Spatial Uncertainty Research Framework            |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    04/07/2019                                   |
*------------------------------------------------------*/
"""

import numpy as np


def exponential(d, s, sill):
    '''
    Inputs: d, distance
            s, scaler
            sill, sill
    Outputs: exp 
    Description: Exponential model. 
    '''

    if np.ndim( s ) > 1:
        h = np.sum(np.square(d/s))**0.5
    else:
        h = d / s
    return sill * ( 1.0 - np.exp( -3.0 * h ) )

def cov(f, pars): 
    '''
    Inputs: f,   model func
            pars, pars taken by f
    Outputs: 
    '''
    def func( d ):
        return pars[-1] - f(d,*pars)
    return func