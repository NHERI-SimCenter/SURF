.. _lbl-usage:

Tutorial
================

In this section, code snippets showing how to use |surfName| are presented for quick start. 
A more detailed example can be found in :ref:`lbl-vnv`, in which there is a jupyter notebook that can run in Google Colab.

Neural network
---------------

The example below shows how to train a ``neural network`` on a data set, and use it to predict.

.. code-block:: python 

    from surf.NN import SpatialNeuralNet

    #---------------------------------------
    # 1. Prepare your data
    #---------------------------------------

    data = load your data here           
    # data is a numpy matrix with columns: [x,y,value]
    # ... see SURF.ET-AI.py for an example:
    # https://github.com/NHERI-SimCenter/SURF/blob/master/examples/demo-NN.py

    #---------------------------------------
    # 2. Train the neural network
    #---------------------------------------
    
    # define a spatial neural network
    # numNei is the number of nearest neighbors to be considered
    nn = SpatialNeuralNet(rawData = data, numNei = 20) 
    
    # create a neural network that can take information of numNei points as input
    nn.build_model()
    
    # this trains the neural network on rawData
    nn.train()
    
    #---------------------------------------
    # 3. Predict
    #---------------------------------------

    # define a point located at (x,y)
    unkown_point = [x, y] 

    # predict a value at the unkown_point
    predicted = nn.predict(unkown_point) 


Random field
--------------

The example below shows how to define a `spatial model` for a data set, and insert it into `random field` to predict.

.. code-block:: python 

    from surf.rf import SK
    import surf.spatialModel as m

    #---------------------------------------
    # 1. Prepare your data
    #---------------------------------------

    data = load your data here         
    # data is a numpy matrix with columns: [x,y,value]
    # ... see SURF.ET.py for an example:
    # https://github.com/NHERI-SimCenter/SURF/blob/master/examples/demo-NN.py

    #---------------------------------------
    # 2. Define spatial model 
    #---------------------------------------

    # correlation length 
    colLength = 0.1  

    # sill of the semivariogram
    sill = np.var( data[:,2] )  

    # cov is covariance function; 
    # exponential means using exponential as the covariance function
    cov = m.cov( m.exponential, ( colLength, sill ) ) 

    #---------------------------------------
    # 3. Predict
    #---------------------------------------
    
    # define a point located at (x,y) 
    unkown_point = [x, y] 

    # predict a value at the unkown_point
    # N is the number of nearest neighbors to depend on
    predicted_mu, predicted_std = SK( data, cov, unkown_point, N=100 ) 
    

