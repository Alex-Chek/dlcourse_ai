import numpy as np


class KNN:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    def __init__(self, k=1):
        self.k = k

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y

    def predict(self, X, num_loops=0):
        '''
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        '''
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        else:
            dists = self.compute_distances_two_loops(X)

        if self.train_y.dtype == np.bool:
            return self.predict_labels_binary(dists)
        else:
            return self.predict_labels_multiclass(dists)
        

    def compute_distances_two_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            for i_train in range(num_train):
                # TODO: Fill dists[i_test][i_train]
#                 pass
                dists[i_test][i_train] = np.sum(np.abs(X[i_test] - self.train_X[i_train]))
        return dists


    
    def compute_distances_one_loop(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        dists = np.zeros((num_test, num_train), np.float32)
        for i_test in range(num_test):
            dists[i_test] = np.abs(
                X[i_test] - self.train_X[i_test]
            )
        return dists

        
            # TODO: Fill the whole row of dists[i_test]
            # without additional loops or list comprehensions
#             pass


    def compute_distances_no_loops(self, X):
        '''
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        '''
        num_train = self.train_X.shape[0]
        num_test = X.shape[0]
        # Using float32 to to save memory - the default is float64
#         dists = np.zeros((num_test, num_train), np.float32)
        # TODO: Implement computing all distances with no loops!
        dists  = np.abs(
            X[:, None] - self.train_X
        ).sum(axis=2)
        return dists

#         pass


    def predict_labels_binary(self, dists):
    # def predict_labels_binary(dists):
            '''
            Returns model predictions for binary classification case

            Arguments:
            dists, np array (num_test_samples, num_train_samples) - array
               with distances between each test and each train sample

            Returns:
            pred, np array of bool (num_test_samples) - binary predictions 
               for every test sample
            '''
            num_test = dists.shape[0]
            pred = np.zeros(num_test, np.bool)

            distances = []
            neighborses = []

            for i in range(num_test):
                neighbors = []
                distances.append(sorted([ (a,b) for a,b in zip(dists[i], self.train_y) ]))

                for x in range(self.k):
                    neighbors.append(distances[i][x][1])    

                neighborses.append(neighbors)

                for n in neighborses:
                    if sum(n) > len(n) // 2:
                        pred[i]=(True)
                    else:
                        pred[i]=(False)
                # TODO: Implement choosing best class based on k
                # nearest training samples
    #             pass
            return pred
    
 
    def predict_labels_multiclass(self, dists):
        '''
        Returns model predictions for multi-class classification case
        
        Arguments:
        dists, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample

        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        '''
        num_test = dists.shape[0]
        pred = np.zeros(num_test, np.int)
        
        distances = []
        neighborses = []

        for i in range(num_test):
            distances.append(sorted([ (a,b) for a,b in zip(dists[i], self.train_y) ]))

            neighbors = []
            for x in range(self.k):
                neighbors.append(distances[i][x][1])
                classVotes={}
                for x in range(len(neighbors)):
                    response = neighbors[x]
                    if response in classVotes:
                        classVotes[response] += 1
                    else:
                        classVotes[response] = 1

            sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)

            pred[i] = (sortedVotes[0][0])

        return pred