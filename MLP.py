import numpy as np


# Activation functions
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