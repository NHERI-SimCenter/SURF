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


from __future__ import absolute_import, division, print_function
import pathlib
import random
import numpy as np
import pandas as pd
from scipy import spatial
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.spatial.distance import squareform, cdist, pdist

# fix random seed for reproducibility
#tf.set_random_seed(1234)

class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: 
      print(epoch)
    print('.', end='')


class SpatialNeuroNet:
    """ A Neural Net Doing Spatial Predictions. """


    def __init__(self, X=None, Y=None, rawData=None, numNei=10, trainFrac=0.8):
        '''
        X: input
        Y: output
        rawData: [x1,x2,value]
        numNei: number of neighbor to be considered
        trainFrac: fraction of data used for training
        '''
        self.numNei = numNei

        if rawData is not None:
            self.rawData = rawData
            self.processRawData()
        elif X is not None:
            self.X = X
            self.Y = Y
        else:
            print("input data must be provided")
            exit()

        
        self.trainFrac = trainFrac

        self.EPOCHS = 5000


        n = len(self.Y)
        ind = random.sample(range(n),n)
        indTrain = ind[0:np.floor(n*trainFrac).astype(int)]
        indTest = ind[np.floor(n*trainFrac).astype(int):]

 
        self.train_labels = self.Y[indTrain]
        self.train_dataset = self.X[indTrain]
        self.test_labels = self.Y[indTest]
        self.test_dataset = self.X[indTest]

        self.mean_train_dataset = np.mean(self.train_dataset, axis = 0)
        self.std_train_dataset = np.std(self.train_dataset, axis = 0)

        self.normed_train_data = self.norm(self.train_dataset)
        self.normed_test_data = self.norm(self.test_dataset)

        # build model
        #self.model = self.build_model()

        # train model
        #self.train()

        # test model
        #self.test()



    def processRawData(self):
        numNei = self.numNei
        perNei = 2
        numPre = 2

        # Defining input size, hidden layer size, output size and batch size respectively
        n_in, n_h, n_out, batch_size = numNei*perNei+numPre, 10, 1, 1000

        rawData = self.rawData[:,0:2]
        rawTarget = self.rawData[:,2:]

        # Create data
        coordsAll = np.array(rawData, dtype=np.float32)
        kdTree = spatial.KDTree(coordsAll)
        data = []
        for i in range(len(rawTarget)):
            distance,index = kdTree.query(rawData[i,0:2],numNei+1) # nearest 10 points
            distance = distance[1:]
            index = index[1:]
            datatmp = rawData[i,:]

            for j in range(numNei):
                datatmp = np.append(np.append(datatmp, distance[j]*100000), rawTarget[index[j]])
            data.append(datatmp.tolist())
        data = np.array(data)
        self.X = data
        self.Y = rawTarget


    def norm(self, v):

        return (v - self.mean_train_dataset) / self.std_train_dataset

    # Build the model
    def build_model(self):
        model = keras.Sequential([
          layers.Dense(256, activation=tf.nn.relu, input_shape=[len(self.train_dataset.T)]),
          layers.Dense(64, activation=tf.nn.relu),
          layers.Dense(64, activation=tf.nn.relu),
          layers.Dense(64, activation=tf.nn.relu),
          layers.Dense(1, activation=tf.nn.sigmoid)#layers.Dense(1)
        ])
        #optimizer = tf.train.RMSPropOptimizer(0.001)
        optimizer = tf.train.AdamOptimizer(1e-4)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        self.model = model
        return model

    # Build the classification model
    def build_classification_model(self, numTypes):
        model = keras.Sequential([
          layers.Dense(len(self.train_dataset.T), activation=tf.nn.relu, input_shape=[len(self.train_dataset.T)]),
          layers.Dense(len(self.train_dataset.T), activation=tf.nn.relu),
          layers.Dense(len(self.train_dataset.T)/2, activation=tf.nn.relu),
          layers.Dense(numTypes, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
        self.model = model
        return model


    def train_classification_model(self):
        self.model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        history = self.model.fit(self.normed_train_data, self.train_labels.astype(int).flatten(), epochs=self.EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print('\n')
        print(hist.tail())
        #self.plot_history(history)
        #plt.savefig('data/NN_ContinuumWall_TrainingLoss_V1.png')

        #loss, mae, mse = self.model.evaluate(self.normed_test_data, self.test_labels, verbose=0)
        #print("Testing set Mean Abs Error: {:5.2f} ".format(mae))


        '''
        # save model

        # serialize model to JSON
        model_json = self.model.to_json()
        with open("data/NNModel_ContinuumWall_V1.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("Data/NNModel_ContinuumWall_V1.h5")
        print("Saved model to disk")
        '''

    def train(self):
        self.model.summary()
        # The patience parameter is the amount of epochs to check for improvement
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        print(self.train_labels)
        history = self.model.fit(self.normed_train_data, self.train_labels, epochs=self.EPOCHS,
                            validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print('\n')
        print(hist.tail())
        self.plot_history(history)
        #plt.savefig('data/NN_ContinuumWall_TrainingLoss_V1.png')

        loss, mae, mse = self.model.evaluate(self.normed_test_data, self.test_labels, verbose=0)
        print("Testing set Mean Abs Error: {:5.2f} ".format(mae))




    def test(self):
        # test
        test_predictions = self.model.predict(self.normed_test_data).flatten()

        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        trueValues = self.test_labels
        #predictValues = test_predictions[0::5]
        predictValues = test_predictions
        print(trueValues.flatten())
        print(predictValues)

        plt.scatter(trueValues.flatten(), predictValues)
        plt.xlabel('True Values [label]')
        plt.ylabel('Predictions [label]')
        plt.axis('equal')
        plt.axis('square')
        
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])

        

        plt.subplot(1,2,2)
        error = predictValues - trueValues.flatten()
        print('errors:  ')
        print(error)
        plt.hist(error, bins=25)

        plt.xlabel("Prediction Error [label]")
        _ = plt.ylabel("Count")

        #plt.savefig('data/Predictions_error.png')
        plt.show()

    def test_classification_model(self):
        # test
        test_predictions = self.model.predict(self.normed_test_data)


        plt.figure(figsize=(20,10))
        plt.subplot(1,2,1)
        trueValues = self.test_labels.flatten()
        #predictValues = test_predictions[0::5]
        predictValues = np.argmax(test_predictions,axis=1)
        print(trueValues)
        print(predictValues)
        print(len(predictValues))
        plt.scatter(trueValues, predictValues)
        plt.xlabel('True Values [label]')
        plt.ylabel('Predictions [label]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])

        plt.subplot(1,2,2)
        error = predictValues - trueValues
        print(self.train_dataset)
        print('errors:  ')
        plt.hist(error, bins = 25)
        plt.xlabel("Prediction Error [label]")
        _ = plt.ylabel("Count")

        #plt.savefig('data/Predictions_classification_error.png')
        plt.show()

    def predict(self, pt):
        X = self.getX(pt, N=self.numNei)
        X = self.norm(X)
        Y = self.model.predict([X]).flatten().item()
        return Y

    def predict_classification_model(self, pt):
        X = self.getX(pt, N=self.numNei)
        X = self.norm(X)
        Y = np.argmax(self.model.predict([X]))
        return Y


    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error ')
        plt.plot(hist['epoch'], hist['mean_absolute_error'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
                 label = 'Val Error')
        plt.legend()
        #plt.ylim([0,1])
        '''
        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$Ap^2$]')
        plt.plot(hist['epoch'], hist['mean_squared_error'],
                 label='Train Error')       
        plt.plot(hist['epoch'], hist['val_mean_squared_error'],
                 label = 'Val Error')
        plt.legend()
        #plt.ylim([0,1])
        '''
        #plt.show()


    def getX( self, n, N=0 ):

        # check dimensions of next 
        if np.ndim( n ) == 1:
            n = [n]

        # get dnp
        d = cdist( self.rawData[:,:2], n )
        P = np.hstack(( self.rawData, d ))

        if N > 0: # use N nearest neighbor
            P = P[d[:,0].argsort()[:N]]
        else: # include all known data
            N = len(P)

        rawData = self.rawData
        coordsAll = np.array(rawData[:,0:2], dtype=np.float32)
        kdTree = spatial.KDTree(coordsAll)
        X = []
        distance,index = kdTree.query(n,N) # nearest N+1 points
        distance = distance[0][0:]
        index = index[0][0:]
        xtmp = n
        for j in range(N):
            xtmp = np.append(np.append(xtmp, distance[j]*100000), rawData[index[j],2])
        X = np.array([xtmp.tolist()])
        return X





    




