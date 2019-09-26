# -*- coding: utf-8 -*-
"""
/*------------------------------------------------------*
|     Spatial Uncertainty Research Framework            |
|                                                       |
| Author: Charles Wang,  UC Berkeley, c_w@berkeley.edu  |
|                                                       |
| Date:    07/11/2019                                   |
*------------------------------------------------------*/
"""


import os
from shutil import copyfile
import json
#import geojsonio
import numpy as np
import geopandas as gpd
from scipy import spatial
from shapely.geometry import Point, Polygon

from surf.rf import SK
import surf.spatialModel as m
from surf.utils import mesh, readInput

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#---------------------------------------------------------------------------------------
# 1. Prepare data
#---------------------------------------------------------------------------------------

inputPars, outputPars = readInput("cetus.json")

# mesh 
mesh(inputPars["boundaryFile"], inputPars["workDir"]+'/Mesh.geojson', density = inputPars["meshDensity"])

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
maxValue = max(data[:,2])


#---------------------------------------------------------------------------------------
# 2. Train the NN
#---------------------------------------------------------------------------------------
from surf.NN import SpatialNeuroNet

nn = SpatialNeuroNet(rawData = data, numNei = 20)
nn.build_model()
nn.train()

#---------------------------------------------------------------------------------------
# 3. Use NN to predict
#---------------------------------------------------------------------------------------

meshFile = gpd.read_file(inputPars["workDir"]+'/Mesh.geojson')
meshjson = json.loads(meshFile.to_json())
meshfeatures = meshjson["features"]
newfeatures = []
cmap = cm.get_cmap('Spectral')

for mf in meshfeatures:
    lonlat = mf['geometry']['coordinates'][0][0]
    pt = lonlat[:2]
    #mu, std = SK( data, cov, pt, N=100 )
    mu = nn.predict(pt)
    #std = 0.0
    rgba = cmap(1-mu/maxValue)
    mf['properties']['mu_Vs30'] = mu
    #mf['properties']['std_Vs30'] = std
    mf['properties']['fill'] = colors.rgb2hex(rgba)
    newfeatures.append(mf)

meshjson["features"] = newfeatures
with open(outputPars["resultFile"],'w') as out:
        json.dump(meshjson, out)
print("Results written "+outputPars["resultFile"])

'''
#---------------------------------------------------------------------------------------
# 4. Visulize result
#---------------------------------------------------------------------------------------

#geojsonio.display(json.dumps(meshjson))
os.chdir("web")
copyfile(outputPars["resultFile"], 'static/Result.geojson')
os.system('sh start.sh')
'''





