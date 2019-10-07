# <i>Spatial Uncertainty Research Framework</i>

<img src="https://raw.githubusercontent.com/NHERI-SimCenter/SURF/master/doc/surf.png" alt="pelicun" height="250"/>

## What is <i>SURF</i>?

<i>SURF</i> is a Python package for performing spatial uncertainty analysis uisng random fields and machine leaning.

## Install

```sh
pip install pySURF
```

## Examples

The example below shows how to train a neural network on a data set, and use it to predict.

```python
    from surf.NN import SpatialNeuroNet

    #---------------------------------------
    # 1. Prepare your data
    #---------------------------------------

    # ... see SURF.ET-AI.py 

    #---------------------------------------
    # 2. Train the neural network
    #---------------------------------------
    
    nn = SpatialNeuroNet(rawData = data, numNei = 20)
    nn.build_model()
    nn.train()

    #---------------------------------------
    # 3. Predict
    #---------------------------------------

    unkown_point = [x, y] # define a point  
    predicted = nn.predict(unkown_point) # predict

```

The example below shows how to define a spatial model for a data set, and insert it into random field to predict.

```python
    from surf.rf import SK
    import surf.spatialModel as m

    #---------------------------------------
    # 1. Prepare your data
    #---------------------------------------

    # ... see SURF.ET.py 

    #---------------------------------------
    # 2. Define spatial model 
    #---------------------------------------

    colLength = 0.1
    sill = np.var( data[:,2] )
    cov = m.cov( m.exponential, ( colLength, sill ) )

    #---------------------------------------
    # 3. Predict
    #---------------------------------------
    
    unkown_point = [x, y] # define a point  
    predicted_mu, predicted_std = SK( data, cov, unkown_point, N=100 )
```

## Application examples

[BRAILS](https://github.com/NHERI-SimCenter/BRAILS)

[Soft-Story Detection](https://github.com/charlesxwang/Soft-Story-Detection)

## License

<i>SURF</i> is distributed under the BSD 3-Clause license.

## Acknowledgement

This material is based upon work supported by the National Science Foundation under Grant No. 1612843. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

## Contact

Charles Wang, NHERI SimCenter, University of California, Berkeley, c_w@berkeley.edu