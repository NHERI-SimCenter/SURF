.. _lbl-usage:

Usage
================

The example below shows how to train a ``neural network`` on a data set, and use it to predict.

.. code-block:: python 

    from surf.NN import SpatialNeuralNet

    #---------------------------------------
    # 1. Prepare your data
    #---------------------------------------

    # ... see SURF.ET-AI.py 

    #---------------------------------------
    # 2. Train the neural network
    #---------------------------------------
    
    nn = SpatialNeuralNet(rawData = data, numNei = 20)
    nn.build_model()
    nn.train()

    #---------------------------------------
    # 3. Predict
    #---------------------------------------

    unkown_point = [x, y] # define a point  
    predicted = nn.predict(unkown_point) # predict



The example below shows how to define a `spatial model` for a data set, and insert it into `random field` to predict.

.. code-block:: python 

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

