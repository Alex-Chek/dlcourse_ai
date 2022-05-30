import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    
    try:
        loss = reg_strength * np.sum(W**2)

        grad = reg_strength * 2 * (W)
    except:
        raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
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
    """
    # TODO: Copy from the previous assignment
    
    preds1 = preds.copy()
    try:
        preds1 -= np.array(np.max(preds1, axis=1)).reshape(preds1.shape[0], 1)
            
        probs = np.exp(preds1.astype(np.float)) / np.sum(np.exp(preds1.astype(np.float)), axis=1, keepdims=True)
        h = -1 * (np.log(probs))
        
        loss = (h[np.arange(target_index.shape[0]) ,target_index.T])    
        loss = np.sum(loss) / target_index.shape[0]
        
        zeros = np.zeros_like(preds)
        zeros[np.arange(target_index.shape[0]) ,target_index.T] = 1
        
        probs[np.arange(target_index.shape[0]) ,target_index.T]-=1
        d_preds =  probs / target_index.shape[0]

#         d_preds =  (probs - zeros) / target_index.shape[0]
    except:
        raise Exception("Not implemented !")

    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        
        try:
#             self.X = X
#             res = np.zeros_like(X)
#             for i in range(X.shape[0]):
#                 res[i] = np.array([0 if x < 0 else x for x in X[i]])
            X_c = X.copy()    
            X_c[X_c < 0] = 0
            self.X = X_c
            return X_c
        except:
            raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        try:
            a = self.X
            a[a>0] = 1
            d_result = np.multiply(a, d_out)
            
        except:
            raise Exception("Not implemented!")
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None
        
#         self.W1 = self.W.grad
#         self.B1 = self.B.grad

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops        
        try:
            self.X = X
            X_c = X.copy()
#             X_c = X_c.dot(self.W.grad) + self.B.grad
            X_c = X_c.dot(self.W.value) + self.B.value

#             X_c = X_c.dot(self.params()['W']) + self.params()['B']
            X_c[X_c < 0] = 0
            
            return X_c
        
        except:
            raise Exception("Not implemented!")
        

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
        try:          
#             a = FullyConnectedLayer.forward(self, self.X)
            a = self.forward(self.X)
            a[a>0] = 1
            a[a<0] = -1
            d_out = np.multiply(a, d_out)
            
            dw = self.X.T.dot(d_out)
            dx = d_out.dot(self.W.value.T)
            db = d_out.sum(axis=0)
            
            self.W.grad += dw
            self.B.grad += db
            
            d_input = dx
            
        except:
            raise Exception("Not implemented!")

        return d_input
    

    def params(self):
        return {'W': self.W, 'B': self.B}
