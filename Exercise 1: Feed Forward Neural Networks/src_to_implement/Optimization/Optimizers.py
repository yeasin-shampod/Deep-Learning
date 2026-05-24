import numpy as np

class Sgd:
    #constructor with a "learning_rate" parameter which should be float
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate
        
    #tensor passes as parameter
    def calculate_update(self, weight_tensor: np.ndarray , gradient_tensor: np.ndarray):
        return weight_tensor - self.learning_rate * gradient_tensor