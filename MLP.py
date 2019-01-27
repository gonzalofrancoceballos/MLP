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
        return actual*np.log(prediction) + (1-actual)*np.log(1-prediction)
    
    def derivate(self, actual, prediction):
        return (actual-prediction)/(prediction*(1-actual))
    
    
class Dense():
    def __init__(self, input_dim, output_dim, activation = "sigmoid"):
        """
        Initialize layer
        """
    
        if activation == "sigmoid": 
            self.activation = Sigmoid()
        if activation == "relu": 
            self.activation = Relu()
        if activation == "tanh": 
            self.activation = Tanh()
        if activation == "linear": 
            self.activation = Linear()
            
        self.reset_layer(input_dim, output_dim)
        
    def reset_layer(self, input_dim, output_dim):
        """
        Randomly reset weight
        """
        self.W = np.random.rand(input_dim, output_dim)
        self.b = np.random.rand(1, output_dim)
        self.dW = np.zeros([input_dim, output_dim])
        self.db = np.zeros([1, output_dim])
        
    def forward(self, X, update=True):
        """
        Forward pass through layer
        """
        Z = np.matmul(X, self.W) + self.b
        A = self.activation.forward(Z)
        if update:
            self.Z = Z
            self.A = A
        return A 
    
    
# Optimizers
class Gradient_descend():
    """
    Implements gradient descend optimizer
    """
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        
    def update_weights(self, layers):
        for i in range(len(layers)):
            layers[i].W = layers[i].W - self.learning_rate * layers[i].dW
            layers[i].b = layers[i].b - self.learning_rate * layers[i].db
        
        return layers
    
    
class Adam():
    """
    Implements Adam optimizer
    """
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8
        self.learning_rate = 0.0001
        self.t = 1
        
    def initialize_parameters(self, layers):
        """
        
        """
        for i, layer in enumerate(layers):
            adam = {
                "mW" : np.zeros(layer.dW.shape),
                "mb" : np.zeros(layer.db.shape),
                "vW" : np.zeros(layer.dW.shape),
                "vb" : np.zeros(layer.db.shape)}
            layers[i].adam = adam
        return layers

        
    def update_weights(self, layers):
        t = self.t
        for i, layer in enumerate(layers):            
            adam  = {
                "mW" : (self.beta_1*layer.adam["mW"] + (1-self.beta_1)*layer.dW),
                "mb" : (self.beta_1*layer.adam["mb"] + (1-self.beta_1)*layer.db),
                "vW" : (self.beta_2*layer.adam["vW"] + (1-self.beta_2)*layer.dW**2),
                "vb" : (self.beta_2*layer.adam["vb"] + (1-self.beta_2)*layer.db**2)}
            
            layer.adam = adam
            
            mW_corrected  = adam["mW"] / (1-(self.beta_1**t))
            mb_corrected  = adam["mb"] / (1-(self.beta_1**t))
            vW_corrected  = adam["vW"] / (1-(self.beta_2**t))
            vb_corrected  = adam["vb"] / (1-(self.beta_2**t))
            
            layer.W = layer.W - (self.learning_rate * mW_corrected/(np.sqrt(vW_corrected)+ self.epsilon))
            layer.b = layer.b - (self.learning_rate * mb_corrected/(np.sqrt(vb_corrected)+ self.epsilon))
            
            layers[i] = layer
        self.t = t+1
        return layers
    
    
class MLP():
    """
    Multi-layer perceptron
    """
    def __init__(self, 
                 X, 
                 hidden_layers = [2,2], 
                 activation = "sigmoid",
                 loss = "mse", 
                 problem="regression",
                 optimizer ="gradient_descent"):
        """
        Initialize network
        
        :param X: input data (type: np.array)
        :param hidden layers: size of hidden layers (type: list[int])
        :param activation: activartion (type: str)
        :param loss: loss function to be used(type: str)
        :param problem: regression, binary_classification, quantile (type: str)
        
        """
        
        self.print_rate = 100
        self._activation = activation
        self.dims = [X.shape[1]] + hidden_layers + [1]

        # Create layers
        self._build_architecture()
        
        # Losses
        if loss == "mse":
            self._loss = MSE()
        if loss == "logloss":
            self._loss = Logloss()
        
        # Output activation
        if problem == "regression" : 
            self._layers[-1].activation = Linear()
        if problem == "quantile" : 
            self._layers[-1].activation = Linear()
        if problem == "binary_classification":
            self._layers[-1].activation = Sigmoid()
        
        # Set optimizer
        if optimizer == "gradient_descent":
            self._optimizer = Gradient_descent()
        if optimizer == "adam":
            self._optimizer = Adam()
            self._layers = self._optimizer.initialize_parameters(self._layers)
            
            
    def _build_architecture(self):
        """
        Build architecture of MLP
        """
        
        self._layers = []
        for input_dim, output_dim in  zip(self.dims[:-1], self.dims[1:]):
            self._layers.append(Dense(input_dim, output_dim, activation=self._activation))
        self.n_layers = len(self._layers)
        
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
        for i in np.arange(len(self._layers))+1:
            # Compute deltas deltas 
            layer = self._layers[-i]
            if i==1:
                delta = self._loss.derivate(y, layer.A) * layer.activation.derivate(layer.Z)
            else:
                i_next = i-1
                layer_next = self._layers[-i_next]
                delta = np.matmul(layer_next.delta, layer_next.W.T) * layer.activation.derivate(layer.Z)
            self._layers[-i].delta = delta
            
            # Compute gradients
            if i==self.n_layers:
                a_in = X
            else:
                i_prev = i+1
                a_in = self._layers[-i_prev].A
            delta_out = self._layers[-i].delta
            self._layers[-i].db = delta_out.sum(axis=0).reshape([1,-1])
            self._layers[-i].dW = np.matmul(a_in.T,  delta_out) 
            self._layers[-i].dW += self.reg_lambda * self._layers[-i].W
            
    def _train_step(self, X, y):
        """
        
        """
        # Forward propagation
        _ = self._forward_prop(X)

        # Back propagation
        self._back_prop(X,y)

        # Update rule
        self._layers = self._optimizer.update_weights(self._layers)
            

    def train(self, X, y, n_iter=100, learning_rate=0.0001, reg_lambda=0.01,  verbose=True):
        """
        
        """
        self._optimizer.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        for i in range(n_iter):
            self._train_step(X,y)         
            if  verbose and i%self.print_rate==0:
                train_loss = self._compute_loss(self._layers[-1].A,y)
                print(f"iter: {i}  loss: {train_loss}")
                

    def _compute_loss(self, actual, prediction):
        """
        Computes loss between prediction and target
        """
        
        current_loss = self._loss.forward(actual, prediction)
        current_loss = np.mean(current_loss)
        return current_loss
    
