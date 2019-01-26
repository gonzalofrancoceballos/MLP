import numpy as np


class Sigmoid():
    """
    Sigmoid activation function
    """
    
    def forward(self, x):
        """
        Forward propagation
        """
        return 1. / (1. + np.exp(-x))
        
    def derivate(self, x):
        """
        Derivative of the function at one point
        """
        return (1-self.forward(x)) * self.forward(x)
    
    
class Relu():
    """
    ReLu activation function
    """
    
    def forward(self, x):
        """
        Forward propagation
        """
        return np.where(x>0, x, 0.)
    
    def derivate(self, x):
        """
        Derivative of the function at one point
        """
        return np.where(x>0, 1., 0.)
    
    
class Tanh():
    """
    Tanh activation function
    """
    
    def forward(self, x):
        """
        Forward propagation
        """
        return np.tanh(x)
    
    def derivate(self, x):
        """
        Derivative of the function at one point
        """
        return 1-np.tanh(x)**2
    
    
class Linear():
    """
    Linear activation function
    """
    
    def forward(self, x):
        """
        Forward propagation
        """
        return x
    
    def derivate(self, x):
        """
        Derivative of the function at one point
        """
        return 1
    
    
# Losses
class MSE():
    """
    Mean Squared error class
    """
    def forward(self, actual, prediction):
        return 0.5*((prediction-actual)**2) 
    
    def derivate(self, actual, prediction):
        return prediction - actual
    
class Logloss():
    """
    Mean Squared error class
    """
    def forward(self, actual, prediction):
        return actual*np.log(prediction) + (1-actual)*log(1-prediction)
    
    def derivate(self, actual, prediction):
        return (actual-prediction)/(prediction*(1-actual))
    
# Layers
class Dense():
    def __init__(self, input_dim, output_dim, activation = "sigmoid"):
        """
        Initialize layer
        """
    
        if activation == "sigmoid": 
            self.activation = activations.Sigmoid()
        if activation == "relu": 
            self.activation = activations.Relu()
        if activation == "tanh": 
            self.activation = activations.Tanh()
        if activation == "linear": 
            self.activation = activations.Linear()
            
        self.reset_layer(input_dim, output_dim)
        
    def reset_layer(self, input_dim, output_dim):
        """
        Randomly reset weight
        """
        self.W = np.random.rand(input_dim, output_dim)
        self.b = np.random.rand(1, output_dim)
        
    def forward(self, X, update=True):
        """
        Forward pass through layer
        """
        Z = np.matmul(X, self.W) + self.b
        A = self.activation.forward(Z)
        if update:
            self.Z = Z
            self.A = A
        return self.A 
    
    
class MLP():
    """
    Multi-layer perceptron
    """
    def __init__(self, 
                 X, 
                 hidden_layers = [2,2], 
                 activation = "sigmoid",
                 loss = "mse", 
                 problem="regression"):
        """
        Initialize network
        
        :param X: input data (type: np.array)
        :param hidden layers: size of hidden layers (type: list[int])
        :param activation: activartion (type: str)
        :param loss: loss function to be used(type: str)
        :param problem: regression, binary_classification, quantile (type: str)
        
        """
        
        self._activation = activation
        self.dims = [X.shape[1]] + hidden_layers + [1]

        if loss == "mse":
            self._loss = MSE()
        self._build_architecture()
        
        if problem == "regression" : 
            self._layers[-1].activation = Linear()
        if problem == "quantile" : 
            self._layers[-1].activation = Linear()
        if problem == "binary_classification":
            self.layers[-1].activation = Sigmoid()
        
        
    def _build_architecture(self):
        """
        Build architecture of MLP
        """
        
        self._layers = []
        for input_dim, output_dim in  zip(self.dims[:-1], self.dims[1:]):
            self._layers.append(Dense(input_dim, output_dim, activation=self._activation))
    
    def _forward_prop(self, X, update=True):
        """
        Computes a forward pass though the architecture of the network
        """
        
        A = self._layers[0].forward(X, update=update)
        for layer in self._layers[1:]:
            A = layer.forward(A, update=update)
            
        return A
    
    def predict(self, X):
        """
        Computes a forward pass and returns prediction
        """
        pred = self._forward_prop(X, update=False)
        return pred
            
    def _back_prop(self, X, y):
        """
        
        """
        # Loop to compute layer errors 
        for i in np.arange(len(self._layers))+1:
            layer = self._layers[-i]
            if i==1:
                error = self._loss.derivate(y, layer.A) * layer.activation.derivate(layer.Z)
                self._layers[-i].error = error
            else:
                i_next = i-1
                layer_next = self._layers[-i_next]
                error = np.matmul(layer_next.error, layer_next.W.T) * layer.activation.derivate(layer.Z)
                self._layers[-i].error = error
        
        # Loop to compute deltas
        for i in range(len(self._layers)):
            layer = self._layers[i]
            # Compute for bias
            if i==0:
                a_in = X
            else:
                a_in = self._layers[i-1].A
            error_out = self._layers[i].error
            self._layers[i].delta_b = error_out
            self._layers[i].delta_W = np.matmul(a_in.T,  error_out) 
            
    def _update_weights(self):
        """
        Implements update rule
        """
        for i in range(len(self._layers)):
            self._layers[i].W = self._layers[i].W - self.learning_rate * self._layers[i].delta_W
            self._layers[i].b = self._layers[i].b - self.learning_rate * self._layers[i].delta_b
        
    def train(self, X, y, n_iter=100, learning_rate=0.0001, verbose=True):
        self.learning_rate = learning_rate
        
        for i in range(n_iter):
            
            # Forward propagation
            _ = self._forward_prop(X)
            
            if  verbose:
                print(self._compute_loss(self._layers[-1].A,y))
            
            # Back propagation
            self._back_prop(X,y)
            
            # Update rule
            self._update_weights()
            
            
    def _compute_loss(self, actual, prediction):
        """
        Computes loss between prediction and target
        """
        
        current_loss = self._current_loss = self._loss.forward(actual, prediction)
        current_loss = np.mean(current_loss)
        return current_loss   