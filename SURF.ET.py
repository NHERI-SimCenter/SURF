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


import os
from shutil import copyfile
import json
#import geojsonio
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

from surf.rf import SK
import surf.spatialModel as m
from surf.utils import mesh, readInput

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors


inputPars, outputPars = readInput("cetus.json")

Vs30File = gpd.read_file(inputPars["dataFile"])
Vs30j = json.loads(Vs30File.to_json())
#geojsonio.display(json.dumps(Vs30j))

data = []
vspts = Vs30j["features"]
for vspt in vspts:
    lonlat = vspt['geometry']['coordinates']
    vs30 = vspt['properties']['Vs30']
    data.append([*lonlat,vs30])
data = np.array(data)

mesh(inputPars["boundaryFile"], inputPars["workDir"]+'/Mesh.geojson', density = inputPars["meshDensity"])

colLength = 0.1
sill = np.var( data[:,2] )
cov = m.cov( m.exponential, ( colLength, sill ) )

meshFile = gpd.read_file(inputPars["workDir"]+'/Mesh.geojson')
meshjson = json.loads(meshFile.to_json())
meshfeatures = meshjson["features"]
newfeatures = []
cmap = cm.get_cmap('Spectral')

for mf in meshfeatures:
    lonlat = mf['geometry']['coordinates'][0][0]
    pt = lonlat[:2]
    mu, std = SK( data, cov, pt, N=100 )
    rgba = cmap(mu/700)
    mf['properties']['mu_Vs30'] = mu
    mf['properties']['std_Vs30'] = std
    mf['properties']['fill'] = colors.rgb2hex(rgba)
    newfeatures.append(mf)

meshjson["features"] = newfeatures
with open(outputPars["resultFile"],'w') as out:
        json.dump(meshjson, out)

print("Results written "+outputPars["resultFile"])

'''
#---------------------------------------------------------------------------------------
# Visulize result
#---------------------------------------------------------------------------------------

#geojsonio.display(json.dumps(meshjson))
os.chdir("web")
copyfile(outputPars["resultFile"], 'static/Result.geojson')
os.system('sh start.sh')
'''








