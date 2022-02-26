import numpy as np

def sigmoid(x):
    x = np.clip(x, a_min = -709, a_max = 709)
    return 1 / (1 + np.exp(-x))

class Model(object):

    def __init__(self):
        self.num_input_features = None

    def fit(self, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

class LogisticRegressionSGD(Model):

    def __init__(self, n_features, learning_rate = 0.1):
        super().__init__()
        # TODO: Initialize parameters, learning rate
        self.n_features = n_features
        #self.learning_rate = learning_rate
        self.w = np.tile(0.0, n_features)
        self.lamda = learning_rate
        pass

    def fit(self, X, y):
        # TODO: Write code to fit the model
        for i in range(X.shape[0]):
            update = self.lamda * (y[i] - sigmoid(np.dot(self.w, np.transpose([X.toarray()[i,]])))) * X.toarray()[i,]
            self.w += update
        pass

    def predict(self, X):
        # TODO: Write code to make predictions
        
        #padding 0s to X for X.shape less than n_features. 
        if (X.shape[1] != self.n_features):
            pad = np.tile(0, (X.shape[0], self.n_features - X.shape[1]))
            X = np.append(X.toarray(), pad, axis = 1)
            y = np.dot(self.w, np.transpose(X))
        else:
            y = np.dot(self.w, np.transpose(X.toarray()))
        for i in range(len(y)):
            if sigmoid(y[i]) >= 0.5:
                y[i] = 1
            else:
                y[i] = 0
        
        return y
        pass

class LogisticRegressionNewton(Model):

    def __init__(self, n_features):
        super().__init__()
        # TODO: Initialize parameters
        self.n_features = n_features
        self.w = np.asmatrix(np.zeros((1, n_features)))
        pass

    def fit(self, X, y):
        # TODO: Write code to fit the model
        # asmatrix allows us to calculate in matrix form
        X = np.asmatrix(X.toarray())
        y = np.asmatrix(y)
        D_sigma = np.asmatrix(np.zeros((X.shape[0], X.shape[0]))) 
        itera = 0
        while itera < 20:
            for i in range(X.shape[0]):
                D_sigma[i,i] = sigmoid(self.w * X[i,].transpose()) * (1 - sigmoid(self.w * X[i,].transpose()))
            H_matrix = X.transpose() * D_sigma * X
            G = np.asmatrix(np.zeros((1, X.shape[1])))
            for i in range(X.shape[1]):
                G[0,i] = (y -1/(1+np.exp(-(self.w*X.transpose())))) * X[:,i]
                
            #update ascent
            self.w += (np.linalg.pinv(H_matrix)*G.transpose()).transpose()    
            itera += 1
            
        pass

    def predict(self, X):
        # TODO: Write code to make predictions
        if (X.shape[1] != self.n_features):
            pad = np.zeros((X.shape[0], self.n_features - X.shape[1]))
            X = np.append(X.toarray(), pad, axis = 1)
            X = np.asmatrix(X)
        else:
            X = np.asmatrix(X.toarray())
        y = 1/(1+np.exp(-(self.w*X.transpose())))
        y = np.ravel(y)
        for i in range(len(y)):
            if y[i] >= 0.5:
                y[i] = 1
            else:
                y[i] = 0
        return y
        
        
        
        pass
