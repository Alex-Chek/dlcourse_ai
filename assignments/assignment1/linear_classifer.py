import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
#     predictions -= np.max(predictions)
    m = np.array(np.max(predictions, axis=1)).reshape(predictions.shape[0], 1)
    
    predictions -= m
    
#     exponents = np.exp(predictions)
    
    try:
#         probs = exponents / np.sum(exponents)
        probs = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
    except:
        raise Exception("Not implemented!")
    
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    try:
        loss = -1 * np.log(probs)
    except:
        raise Exception("Not implemented!")

    return loss[target_index]


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    predictions1 = predictions.copy()
    try:
#         predictions1 -= np.max(predictions1)    
        predictions1 -= np.array(np.max(predictions, axis=1)).reshape(predictions.shape[0], 1)
    
#         exponents = np.exp(predictions1)
#         probs = exponents / np.sum(exponents)
        
        probs = np.exp(predictions1) / np.sum(np.exp(predictions1), axis=1, keepdims=True)
#         print(probs.shape)
        h = -1 * (np.log(probs))
#         print('h :\n', h)
#         loss = (h[:, target_index])
        loss = (h[np.arange(target_index.shape[0]) ,target_index.T])

        loss = np.sum(loss) / loss.shape[0]
#         loss = np.sum(h) / h.shape[0]
#         loss = np.argmin(loss)
    
        zeros = np.zeros_like(predictions)
        zeros[np.arange(target_index.shape[0]) ,target_index.T] = 1
#         zeros[:, target_index] = 1
 
        dprediction =  (probs - zeros)
#         dprediction =  (probs - h)

    except:
        raise Exception("Not implemented !")

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    try:
        loss = reg_strength * np.sum(W**2)

        grad = reg_strength * 2 * (W)
    except:
        raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    
    predictions = np.dot(X, W)
    predictions1 = predictions.copy()
#     print(predictions)
    try:
#         predictions1 -= np.max(predictions1)    
        predictions1 -= np.array(np.max(predictions, axis=1)).reshape(predictions.shape[0], 1)
#         print(predictions1)
#         exponents = np.exp(predictions1)
#         probs = exponents / np.sum(exponents)
        
        probs = np.exp(predictions1) / np.sum(np.exp(predictions1), axis=1, keepdims=True)
        h = -1 * (np.log(probs))
#         print('h :\n', h)
#         loss = (h[:, target_index])
        loss = (h[np.arange(target_index.shape[0]), target_index.T])
#         print('loss array :', loss)
        
        loss = np.sum(loss) / loss.shape[0]
#         print('loss :', loss)
        
        zeros = np.zeros_like(predictions)
#         zerosW = np.zeros_like(W)

#         zeros = np.zeros_like(W)
    
        zeros[np.arange(target_index.shape[0]), target_index.T] = 1
#         zerosW[np.arange(target_index.shape[0]), target_index.T] = 1
#         zeros[:, target_index] = 1
 
        dp =  (probs - zeros)
#         dp = (probs**(-1))
#         print('dp :', dp)
#         dw =  (W - zerosW)
#         dW = W.dot(dp)
        dW = (X.T.dot(dp)) / X.shape[0]
#         (self.p - self.y) / self.y.shape[0]


    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    except:
        raise Exception("Not implemented!!!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None
        self.probs = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

#             target_index = np.ones(batch_size, dtype=np.int)
            
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # target_index = np.ones(batch_size, dtype=np.int)
#             np.random.seed(41)
#             target_index = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int)

#             print('target index shape: ', target_index.shape)
#             print('***\ntarget index: ', *target_index, '\n***')
            try:
                losses = []

                for batch in batches_indices:
    #                 losses = []
                    X_b = X[batch]
    #                 np.random.seed(41)

    #                 target_index = np.random.randint(0, num_classes, size=(batch_size, 1)).astype(np.int)
                    target_index = y[batch]
    #                 print('target index shape: ', target_index.shape)
    #                 print('***\ntarget index: ', *target_index, '\n***')
                    predictions = np.dot(X_b, self.W)
                    predictions1 = predictions.copy()


                    predictions1 -= np.array(np.max(predictions, axis=1)).reshape(predictions.shape[0], 1)

                    probs = np.exp(predictions1) / np.sum(np.exp(predictions1), axis=1, keepdims=True)
                    self.probs = probs
#                     print(probs.shape, ' >probs')
                    h = -1*(np.log(probs))#*probs
#                     print(h.shape, ' >h')

                    loss = (h[np.arange(target_index.shape[0]), target_index.T])#+ reg * np.sum(self.W**2)
    #                     print(loss.shape, ' >loss')
#                     loss = -1* np.sum(h, axis=1)#+ reg * np.sum(self.W**2)
#                     print(loss.shape, ' >loss.shape')
                    loss = np.sum(loss) / loss.shape[0] + reg * np.sum(self.W**2)
                    print(loss, ' >loss')
#                     loss = np.sum(loss) + reg * np.sum(self.W**2)

#                     print(loss, ' >loss')
#                     loss = np.sum(loss)/ loss.shape #+ reg * np.sum(self.W**2)
#                     loss = np.sum(loss)+ reg * np.sum(self.W**2)

    #                     loss = np.sum(h) #/ h.shape[0]
                    zeros = np.zeros_like(predictions)
                    zeros[np.arange(target_index.shape[0]), target_index.T] = 1

#                     dp =  (probs - zeros)
                    dp =  (probs - zeros)
                    dW = (X_b.T.dot(dp)) #/ X_b.shape[0]
#                     dp_1 = -1 * (probs**(-1))
#                     dW = (X_b.T.dot(dp_1)) #/ X_b.shape[0]

#                     print(reg * np.sum(self.W**2))    
#                     loss = loss + reg * np.sum(self.W**2)

                    self.W = self.W - learning_rate * dW
#                     loss = loss + reg * np.sum(self.W**2)
#                     print(reg * np.sum(self.W**2))    
    #                     print('loss : ', loss)
    #                     loss_history.append(loss)
                    losses.append(loss)
                loss1 = sum(losses)/len(losses)# + reg * np.sum(self.W**2)
                loss_history.append(loss)
                
#                 self.W = self.W - learning_rate * dW

            except:
                raise Exception("Not implemented NOW!!!")

#             loss1 = max(losses)

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history
    

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
#         try:
#             y_pred = np.zeros_like(y_test)

#             for i, x in enumerate(X):

#                 prob = self.probs

#                 y = np.argmax(prob)
                
#                 y_pred[i] = y
            y_pred = np.argmax((X.dot(self.W)), axis=1) 
        except:
            raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
