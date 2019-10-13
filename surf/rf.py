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


from scipy.spatial.distance import squareform, cdist, pdist
import numpy as np


def getSIGMA( data, covfunc, n, N=0 ):

    # check dimensions of next 
    if np.ndim( n ) == 1:
        n = [n]

    
    # check dimensions, make sure data and 
    if(len(n[0]) != len(data[0,:])-1):
        print("Data dimension != input dimension!")
        print([len(n), len(data[0,:])-1])
        exit()
    

    # get dnp
    d = cdist( data[:,:-1], n )
    P = np.hstack(( data, d ))

    if N > 0: # use N nearest neighbor
        P = P[d[:,0].argsort()[:N]]
    else: # include all known data
        N = len(P)

    # get SIGMAnp
    SIGMA12 = covfunc( P[:,-1] )
    SIGMA12 = np.matrix( SIGMA12 ).T

    # get SIGMApp
    dMatrix = squareform( pdist( P[:,:-2] ) )
    SIGMA22 = np.array ( covfunc( dMatrix.ravel() ) )
    SIGMA22 = SIGMA22.reshape(N,N)
    SIGMA22 = np.matrix( SIGMA22 )

    return SIGMA22, SIGMA12, P

def SK( data, covfunc, n, N = 0, nug = 0 ):
    
    # get matrix SIGMA22, SIGMA12
    SIGMA22, SIGMA12, newData = getSIGMA( data, covfunc, n, N )

    # calculate weights
    w = np.linalg.inv( SIGMA22 ) * SIGMA12
    w = np.array( w )

    # normalize to 1
    w = np.dot(w, 1.0/np.sum(w))

    # SIGMA21 * SIGMA22 * SIGMA12 
    k = SIGMA12.T * w
    

    # get original mean 
    mu = np.mean( data[:,-1] )
    
    # get the residuals
    residuals = newData[:,-2] - mu

    # best mean
    mu = np.dot( w.T, residuals ) + mu
    #mu = np.dot( w.T, newData[:,-2] )

    # get sigma
    sill = np.var( data[:,-1] )

    k = float( sill + nug - k )
    std = np.sqrt( k )

    return float(mu), std