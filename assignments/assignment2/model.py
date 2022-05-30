import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, Param


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.hidden_layer_size = hidden_layer_size
        self.X = None
#         self.probabylities = None
        
#         self.w_list = []
#         self.b_list = []
        try:
            self.W = Param(np.random.randn(n_input, hidden_layer_size))
            self.B = Param(np.random.randn(1, hidden_layer_size))
            self.W2 = Param(np.random.randn(hidden_layer_size, n_output))
            self.B2 = Param(np.random.randn(1, n_output))
#             self.W = np.random.randn(n_input, n_output)
#             self.B = np.random.randn(1, n_output)
#             self.W2 = np.random.randn(n_output, n_output)
#             self.B2 = np.random.randn(1, n_output)

#         for hidden_layer in range(self.hidden_layer_size):
#             self.w_list.append(self.W)
#             self.b_list.append(self.B)
               
#         self.value = value
#         self.grad = np.zeros_like(value)
        # TODO Create necessary layers

        except:
            raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        try:
            self.W.grad = np.zeros_like(self.params()['W'].grad)
            self.B.grad = np.zeros_like(self.params()['B'].grad)
            self.W2.grad = np.zeros_like(self.params()['W2'].grad)
            self.B2.grad = np.zeros_like(self.params()['B2'].grad)
#             
#             print('X.shape: ', X.shape, '\tW1.shape: ', self.W.value.shape, '\tB1.shape: ', self.B.value.shape)
            X1 = X.dot(self.W.value) + self.B.value
#             print('X1.shape: ', X1.shape, '\tW2.shape: ', self.W2.value.shape, '\tB2.shape: ', self.B2.value.shape)
            X11 = ReLULayer.forward(self, X1)
#             print('X11.shape: ', X11.shape)
            X2 = X11.dot(self.W2.value) + self.B2.value
#             print('X2.shape: ', X2.shape)
            X22 = ReLULayer.forward(self, X2)
#             print('X22.shape: ', X2.shape)
            losses = softmax_with_cross_entropy(X22, y)
#             self.probabylities = losses[2]        
        
            loss = losses[0] 
        
            l_reg = self.reg * (np.sum(self.W.value**2) + np.sum(self.B.value**2) + np.sum(self.W2.value**2) + np.sum(self.B2.value**2) )

            loss += l_reg
#             print('Loss: ', loss)
            d_out = losses[1] 
#             print('d_out.shape: ', d_out.shape)
#                 dx_2 = FullyConnectedLayer.backward(self, X1)
#                 FullyConnectedLayer.backward(dx_2)
            a = X2
            a[a>0] = 1
            a[a<=0] = 0
            d_out = np.multiply(a, d_out)
#             print('d_out_2.multiply shape: ', d_out.shape)
            dw2 = X11.T.dot(d_out)
            dx2 = d_out.dot(self.W2.value.T)
            db2 = d_out.sum(axis=0)
#             print('dw2.shape: ', dw2.shape)
#             print('dx2.shape: ', dx2.shape)
#             print('db2.shape: ', db2.shape, '\n')
            self.W2.grad += dw2 + 2*self.W2.value * self.reg
            self.B2.grad += db2 + 2*self.B2.value * self.reg

            a = X1
            a[a>0] = 1
            a[a<=0] = 0
            d_out = np.multiply(a, dx2)
#             print('d_out.multiply shape: ', d_out.shape)

            dw1 = X.T.dot(d_out)
            dx1 = d_out.dot(self.W.value.T)
            db1 = d_out.sum(axis=0)
#             print('dw1.shape: ', dw1.shape)
#             print('dx1.shape: ', dx1.shape)
#             print('db1.shape: ', db1.shape)

            self.W.grad += dw1 + 2*self.W.value * self.reg
            self.B.grad += db1 + 2*self.B.value * self.reg
        
        except:
            raise Exception("Not implemented!")
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
#         raise Exception("Not implemented!")

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        try:
            pred = np.zeros(X.shape[0], np.int)
        
            X1 = X.dot(self.W.value) + self.B.value
            X11 = ReLULayer.forward(self, X1)
            X2 = X11.dot(self.W2.value) + self.B2.value
            preds1 = ReLULayer.forward(self, X2)
#             print(preds1.shape)

#             preds1 -= np.array(np.max(preds1, axis=1)).reshape(preds1.shape[0], 1)
            
#             probs = np.exp(preds1.astype(np.float)) / np.sum(np.exp(preds1.astype(np.float)), axis=1, keepdims=True)
#             h = -1 * (np.log(probs))
            predd = np.argmax(preds1, axis=1) 
            
        except:
            raise Exception("Not implemented!")
        
        print('pred:', predd)   
        return predd

    
    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        try:
            result['B'] = self.B
            result['B2'] = self.B2
            result['W2'] = self.W2
            result['W'] = self.W

#             result['W'] = self.W
#             result['B'] = self.B
#             result['W2'] = self.W2
#             result['B2'] = self.B2
        except:
            raise Exception("Not implemented!")

        return result
