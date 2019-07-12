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
import json
import geopandas as gpd
from shapely.geometry import Point, Polygon

#def splitPolygon(o):

def mesh(oldFile, newFile, esize=0, density=100.):
    f = gpd.read_file(oldFile)
    jsn = json.loads(f.to_json())
    pts = jsn["features"][0]['geometry']['coordinates'][0]
    poly = Polygon(pts)

    
    newnodes = np.array(pts)

    xmax = max(newnodes[:,0])
    xmin = min(newnodes[:,0])
    ymax = max(newnodes[:,1])
    ymin = min(newnodes[:,1])

    xv = newnodes[:,0].T
    yv = newnodes[:,1].T

    if esize < 1e-10:
        esize = -(ymin-ymax)/density
    m = round(-(ymin-ymax)/esize)
    n = round(-(xmin-xmax)/esize)

    X = np.linspace(xmin, xmax, n )
    Y = np.linspace(ymin, ymax, m )
 
    
    features = []
    for x in X:
        for y in Y:
            p = Point((x,y))
            if p.within(poly):
                feature={"type": "Feature","properties": {"fill": "#ffffff",},}
                feature['geometry']={'type' : "Polygon"}
                i = [x-esize/2., y-esize/2., 0]
                j = [x+esize/2., y-esize/2., 0]
                k = [x+esize/2., y+esize/2., 0]
                l = [x-esize/2., y+esize/2., 0]
                feature['geometry']['coordinates']=[[i,j,k,l]]
                features.append(feature)


    jsn["features"] = features
    with open(newFile,'w') as out:
        json.dump(jsn, out)

def readInput(inputJsonFileName):
    inj = json.load(open(inputJsonFileName,'r'))

    inputNames = ["workDir", "boundaryFile", "dataFile", "meshDensity"]
    outputNames = ["resultFile"]

    data = []
    ET = inj["ET"]
    weakAPI = ET["APIs"]["weakCouplings"]

    inputPars = {}
    inputs = weakAPI["inputs"]
    for input in inputs:
        name = input["name"]
        value = input["value"]
        inputPars[name] = value
    
    outputPars = {}
    outputs = weakAPI["outputs"]
    for output in outputs:
        name = output["name"]
        value = output["value"]
        outputPars[name] = value
    
    return inputPars, outputPars

    '''
    workDir = '/Users/simcenter/Codes/SimCenter/SURF/data'
    boundaryFile = '/Users/simcenter/Codes/SimCenter/SURF/data/EastBayBoundary.geojson'
    dataFile = '/Users/simcenter/Codes/SimCenter/SURF/data/EastBayVs30Data.geojson'
    resultFile = '/Users/simcenter/Codes/SimCenter/SURF/data/data.geojson'
    meshDensity = 10
    return workDir, boundaryFile, dataFile, resultFile, meshDensity
    '''