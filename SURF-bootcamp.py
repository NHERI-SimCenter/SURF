

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



resultBIMFileName = '/Users/simcenter/Downloads/Atlantic_Cities_BIM-someRoof.geojson'
newBIMFileName = '/Users/simcenter/Downloads/New_BIM.geojson'
missingName = 'yearBuilt'
isCategorical = False
isInt = True


#---------------------------------------------------------------------------------------
# 1. Prepare data
#---------------------------------------------------------------------------------------


BIMFile = gpd.read_file(resultBIMFileName)
BIMJ = json.loads(BIMFile.to_json())
#geojsonio.display(json.dumps(Vs30j))

fs = BIMJ["features"]

# get category names
types = []
for j in fs:
    lat = j['properties']['lat']
    lon = j['properties']['lon']
    missingValue = j['properties'][missingName]
    if missingValue is not None:
        types.append(missingValue)

types = list(set(types))
typeDict = {}
typeDictReverse = {}
for i in range(len(types)):
    typeDict[types[i]] = i
    typeDictReverse[i] = types[i]
print(typeDict)
print(typeDictReverse)
#exit()

# build raw data
data = []
for j in fs:
    lat = j['properties']['lat']
    lon = j['properties']['lon']
    missingValue = j['properties'][missingName]
    if missingValue is not None:
        data.append([lon, lat, typeDict[missingValue] if isCategorical else missingValue])

data = np.array(data)

#---------------------------------------------------------------------------------------
# 2. Build and Train the NN
#---------------------------------------------------------------------------------------











#---------------------------------------------------------------------------------------
# 3. Use NN to predict
#---------------------------------------------------------------------------------------


newfeatures = []

for j in fs:
    lat = j['properties']['lat']
    lon = j['properties']['lon']
    pt = [lon, lat]
    missingValue = j['properties'][missingName]
    if missingValue is None:
        missingV = nn.predict(pt)
        j['properties'][missingName] = int(missingV) if isInt else missingV

    newfeatures.append(j)
#exit()
print("Missing name predicted. Now writing new BIM file.")

BIMJ["features"]= newfeatures
#newBIMFileName = resultBIMFileName #dataDir+"/new2.geojson"
with open(newBIMFileName, 'w+') as newBIMFile:
    json.dump(BIMJ, newBIMFile)
    print("new bim has been added to {}".format(newBIMFileName))














