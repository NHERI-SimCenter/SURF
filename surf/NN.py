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
import os
import json
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


class SpatialNeuralNet:
    """ A Neural Net Doing Spatial Predictions. """


    def __init__(self, X=None, Y=None, rawData=None, architecture=None, activation=None,modelType='regression', distScaler = 100000., numNei=10, trainFrac=0.8,testFrac=None, writeTmpData=False, workDir='./tmp', saveFigs=True, plotFigs=True):
        '''
        X: input
        Y: output
        rawData: [x1,x2,value]
        numNei: number of neighbor to be considered
        trainFrac: fraction of data used for training
        '''

        if architecture is None:
            # default architecture
            self.architecture = [256, 64, 64, 64, 1]
        else:
            if len(architecture)<2:
                print("Length of NN architecture must be greater than 1")
                exit()
            self.architecture = architecture
        
        self.activation = activation
        self.modelType = modelType

        self.numNei = numNei
        self.distScaler = distScaler

        self.writeTmpData = writeTmpData
        self.workDir = workDir
        self.saveFigs = saveFigs
        self.plotFigs = plotFigs

        hasInput = True
        if rawData is not None:
            self.rawData = rawData
            self.processRawData()
        elif X is not None:
            self.X = X
            self.Y = Y
        else:
            print("No input is provided, assuming this is model will be used for predicting. ")
            hasInput = False

        if hasInput:
            if testFrac is not None: # testFrac dominates
                self.trainFrac = 1.0 - testFrac
            else: self.trainFrac = trainFrac

            self.EPOCHS = 5000


            n = self.X.shape[0]
            ind = random.sample(range(n),n)
            indTrain = ind[0:np.floor(n*trainFrac).astype(int)]
            indTest = ind[np.floor(n*trainFrac).astype(int):]

            self.train_dataset = self.X[indTrain]
            self.test_dataset = self.X[indTest]
            if self.Y is not None:
                self.train_labels = self.Y[indTrain]
                self.test_labels = self.Y[indTest]


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

            if not os.path.exists(workDir):
                pathlib.Path(workDir).mkdir(parents=True, exist_ok=True)

            if writeTmpData:
                if rawData is not None:
                    np.savetxt(workDir+'/test_dataset.txt', self.rawData[indTest,:])
                    np.savetxt(workDir+'/train_dataset.txt', self.rawData[indTrain,:])


    def processRawData(self,rawData=None,numColumnsY=1):
        numNei = self.numNei
        perNei = 2
        numPre = 2

        # Defining input size, hidden layer size, output size and batch size respectively
        n_in, n_h, n_out, batch_size = numNei * perNei + numPre, 10, 1, 1000

        if rawData is None:# normally built model
            if numColumnsY == 1:
                rawData = self.rawData[:,:0-numColumnsY]
                rawTarget = self.rawData[:,-numColumnsY:]
                self.Y = rawTarget
            elif  numColumnsY == 0:# no target
                rawData = self.rawData
            else:
                print('SURF currently can not deal with multi-dimensional targets.')
                exit()
        else:# loaded model
            if numColumnsY == 1:
                rawTarget = self.rawData[:,-numColumnsY:]
                self.Y = rawTarget
            elif  numColumnsY == 0:# no target
                rawData = rawData
            else:
                print('SURF currently can not deal with multi-dimensional targets.')
                exit()

        # Create data
        coordsAll = np.array(rawData, dtype=np.float32)
        kdTree = spatial.KDTree(coordsAll)
        data = []
        for i in range(rawData.shape[0]):
            distance,index = kdTree.query(rawData[i,:],numNei+1) # nearest 10 points
            distance = distance[1:]
            index = index[1:]
            datatmp = rawData[i,:]

            for j in range(numNei):
                if numColumnsY==1:
                    datatmp = np.append(np.append(datatmp, distance[j]*self.distScaler), rawTarget[index[j]])
                elif numColumnsY==0:
                    datatmp = np.append(datatmp, distance[j]*self.distScaler)
                else:
                    print('SURF currently can not deal with multi-dimensional targets.')
                    exit()
            data.append(datatmp.tolist())
        data = np.array(data)
        self.X = data
        return data

    def processRawDataLoad(self,rawData=None):
        numNei = self.numNei
        perNei = 2
        numPre = 2

        # Defining input size, hidden layer size, output size and batch size respectively
        n_in, n_h, n_out, batch_size = numNei * perNei + numPre, 10, 1, 1000

        # Create data
        coordsAll = np.array(self.rawData[:,0:-1], dtype=np.float32)
        rawTarget = self.rawData[:,-1]
        kdTree = spatial.KDTree(coordsAll)
        data = []
        for i in range(rawData.shape[0]):
            distance,index = kdTree.query(rawData[i,:],numNei+1) # nearest 10 points
            distance = distance[1:]
            index = index[1:]
            datatmp = rawData[i,:]

            for j in range(numNei):
                datatmp = np.append(np.append(datatmp, distance[j]*self.distScaler), rawTarget[index[j]])

            data.append(datatmp.tolist())
        data = np.array(data)
        #self.X = data
        return data       


    def norm(self, v):
        return (v - self.mean_train_dataset) / self.std_train_dataset

    # Build the model
    def build_model(self,numTypes=None):
        print("Building the neural network ...\n")
        if self.modelType == "classification":
            model = self.build_classification_model(numTypes)
            return model
        else:
            archi = []
            archi.append(layers.Dense(self.architecture[0], activation=tf.nn.relu, input_shape=[len(self.train_dataset.T)]))
            for i in self.architecture[1:-1]:
                archi.append(layers.Dense(i, activation=tf.nn.relu))
            if self.activation is None:
                archi.append(layers.Dense(self.architecture[-1]))
            elif self.activation == "sigmoid":
                archi.append(layers.Dense(self.architecture[-1], activation=tf.nn.sigmoid)) # for 0~1
            else:#
                #TODO: add more activation fuctions
                archi.append(layers.Dense(self.architecture[-1]))

            model = keras.Sequential(archi)
            #optimizer = tf.train.RMSPropOptimizer(0.001)
            #optimizer = tf.train.AdamOptimizer(1e-4)
            model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
            self.model = model
            return model
    
    def load_model(self, modelName):
        if os.path.isdir(modelName):
            self.modelLoadedModelPath = modelName
        else: self.modelLoadedModelPath = self.workDir + '/' + modelName

        with open(self.modelLoadedModelPath+'/config.json') as json_file:
            m = json.load(json_file)
        self.numNei = m['numNei']
        self.modelType = m['modelType']
        
        self.model = tf.keras.models.load_model(self.modelLoadedModelPath)

        # Check its architecture
        self.model.summary()

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
    def save(self, modelName = 'surf_model'):
        modelDir = self.workDir+'/'+modelName
        self.model.save(modelDir)
        self.model.save(modelDir + '/saved_model.h5')
        np.savetxt(modelDir+'/mean_train_dataset.txt',self.mean_train_dataset)
        np.savetxt(modelDir+'/std_train_dataset.txt',self.std_train_dataset)
        m = {'modelName':modelName,
            'numNei':self.numNei,
            'modelType':self.modelType}
        with open(modelDir+'/config.json', 'w') as outfile:
            json.dump(m, outfile)

        print('model saved at ',modelDir)


    def train(self):
        if self.modelType == "classification":
            model = self.train_classification_model()

        else:

            print("Training the neural network ... \n")
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
            if self.plotFigs:
                self.plot_history(history)
            #plt.savefig('data/NN_TrainingLoss.png')
            #plt.savefig('data/NN_TrainingLoss.pdf')

            loss, mae, mse = self.model.evaluate(self.normed_test_data, self.test_labels, verbose=0)
            print("Testing set Mean Abs Error: {:5.2f} ".format(mae))


    def predictMulti(self,X):
        self.mean_train_dataset = np.loadtxt(self.modelLoadedModelPath+'/mean_train_dataset.txt')
        self.std_train_dataset = np.loadtxt(self.modelLoadedModelPath+'/std_train_dataset.txt')
        X = self.processRawDataLoad(rawData=X)
        
        #print([X.shape,self.mean_train_dataset.shape,self.std_train_dataset.shape])
        X = (X - self.mean_train_dataset) / self.std_train_dataset
        print(self.modelType)

        #X = self.norm(X)[:,0:-1]
        if self.modelType == 'classification':
            Y = self.model.predict(X)
            Y = np.argmax(Y,axis=1)
        else: Y = self.model.predict(X).flatten()
        np.savetxt(self.modelLoadedModelPath+'/Y.txt', Y)
        print("Predictions are saved in ", self.modelLoadedModelPath+'/Y.txt')
        return Y

    def plot(self, trueValues, predictValues):
        print(trueValues.shape, predictValues.shape)
        if self.Y is not None:
            plt.figure(figsize=(20,10))
            plt.subplot(1,2,1)
            #trueValues = self.test_labels.flatten()
            ##predictValues = test_predictions[0::5]
            #predictValues = test_predictions
            print(trueValues)
            print(predictValues)

            plt.scatter(trueValues, predictValues, marker='o', c="red", alpha=0.01)
            plt.xlabel('True Values', fontsize=30)
            plt.ylabel('Predictions', fontsize=30)
            plt.axis('equal')
            plt.axis('square')

            #minV = min([min(predictValues),min(trueValues)])
            #maxV = max([max(predictValues),max(trueValues)])
            minV = min(trueValues)
            maxV = max(trueValues)
            marginV = 0.1 * (maxV - minV)

            plt.xlim(minV-marginV,maxV+marginV)
            plt.ylim(minV-marginV,maxV+marginV)
            plt.tick_params(axis='x', labelsize=25)
            plt.tick_params(axis='y', labelsize=25)
            plt.plot([minV-marginV, minV-marginV,maxV+marginV], [minV-marginV, minV-marginV,maxV+marginV],'k-')


            '''
            # year built
            plt.xlim(1875, 2050)
            plt.ylim(1875, 2050)
            '''


            '''
            # num of stories
            plt.xlim([plt.xlim()[0],plt.xlim()[1]])
            plt.ylim([plt.xlim()[0],plt.ylim()[1]])
            plt.plot([0, 2050], [0, 2050],'k-')
            '''

            plt.subplot(1,2,2)
            error = trueValues - predictValues
            lenV = max([abs(min(error)),abs(max(error))])


            print('errors:  ')
            print(error)
            plt.xlim(0.-lenV*1.2, lenV*1.2)
            plt.hist(error, facecolor='g') 
            #plt.hist(error, bins=25, facecolor='g') #year built
            #plt.xlim(-100, 100) # year built
            #plt.hist(error, bins=36, facecolor='g') #num of stories
            #plt.xlim(-26, 26) # num of stories

            plt.xlabel("Prediction Error", fontsize=30)
            plt.ylabel("Count", fontsize=30)
            plt.tick_params(axis='x', labelsize=25)
            #plt.savefig('data/Predictions_error.pdf')
            #plt.savefig('data/Predictions_error.png')
            if self.saveFigs:
                plt.savefig(self.workDir+'/Prediction_errors.png')
                plt.savefig(self.workDir+'/Prediction_errors.pdf')
                print("Figures are saved in ", self.workDir)

            plt.show()

    def test(self):
        # test
        if self.modelType == "classification":
            model = self.test_classification_model()

        else:
            test_predictions = self.model.predict(self.normed_test_data).flatten()

            if self.writeTmpData:
                np.savetxt(self.workDir+'/test_predictions.txt', test_predictions)
                print("Figures are saved in ", self.workDir+'/test_predictions.txt')

            trueValues = self.test_labels.flatten()
            self.plot(trueValues, test_predictions)


    def test_classification_model(self):
        # test
        test_predictions = self.model.predict(self.normed_test_data)

        if self.writeTmpData:
            np.savetxt(self.workDir+'/test_predictions.txt', test_predictions)
            print("Results are saved in ", self.workDir+'/test_predictions.txt')


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

        if self.saveFigs:
            plt.savefig(self.workDir+'/Prediction_errors.png')
            plt.savefig(self.workDir+'/Prediction_errors.pdf')
            print("Figures are saved in ", self.workDir)

        #plt.savefig('data/Predictions_classification_error.png')
        plt.show()

    def predict(self, pt):
        X = self.getX(pt, N=self.numNei)
        X = self.norm(X)
        Y = self.model.predict([X]).flatten().item()
        return Y
    
    def predict_simple(self, pt):
        X = [self.norm(pt)]
        Y = self.model.predict([X]).flatten().item()
        return Y

    def predict_classification_model(self, pt):
        X = self.getX(pt, N=self.numNei)
        X = self.norm(X)
        Y = np.argmax(self.model.predict([X]))
        return Y

    def predictMulti_classification_model(self, X):
        self.mean_train_dataset = np.loadtxt(self.modelLoadedModelPath+'/mean_train_dataset.txt')
        self.std_train_dataset = np.loadtxt(self.modelLoadedModelPath+'/std_train_dataset.txt')
        X = self.processRawDataLoad(rawData=X)
        
        #print([X.shape,self.mean_train_dataset.shape,self.std_train_dataset.shape])
        X = (X - self.mean_train_dataset) / self.std_train_dataset

        #X = self.norm(X)[:,0:-1]
        if self.modelType == 'classification':
            Y = self.model.predict(X)
            Y = np.argmax(Y,axis=1)
        else: Y = self.model.predict(X).flatten()
        np.savetxt(self.modelLoadedModelPath+'/Y.txt', Y)
        print("Predictions are saved in ", self.modelLoadedModelPath+'/Y.txt')
        return Y

    def plot_history(self, history):
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error ')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
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
        if self.saveFigs:
            plt.savefig(self.workDir + '/NN_TrainingLoss.png')
            plt.savefig(self.workDir + '/NN_TrainingLoss.pdf')
        plt.show()


    def getX( self, n, N=0 ):

        # check dimensions of next 
        if np.ndim( n ) == 1:
            n = [n]

        # get dnp
        d = cdist( self.rawData[:,:-1], n )
        P = np.hstack(( self.rawData, d ))

        if N > 0: # use N nearest neighbor
            P = P[d[:,0].argsort()[:N]]
        else: # include all known data
            N = len(P)

        rawData = self.rawData
        coordsAll = np.array(rawData[:,:-1], dtype=np.float32)
        kdTree = spatial.KDTree(coordsAll)
        X = []
        distance,index = kdTree.query(n,N) # nearest N+1 points
        distance = distance[0][0:]
        index = index[0][0:]
        xtmp = n
        for j in range(N):
            xtmp = np.append(np.append(xtmp, distance[j]*self.distScaler), rawData[index[j],2])
        X = np.array([xtmp.tolist()])
        return X





    




