import numpy as np


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.lr = learning_rate

    def calculate_update(self, current_weights, current_grads):
        if self.regularizer is not None:
            current_weights = current_weights - self.lr * self.regularizer.calculate_gradient(current_weights)

        return current_weights - self.lr * current_grads


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.lr = learning_rate
        self.momentum = momentum_rate
        self.velocity = None

    def calculate_update(self, current_weights, current_grads):
        if self.regularizer is not None:
            current_weights = current_weights - self.lr * self.regularizer.calculate_gradient(current_weights)

        if self.velocity is None:
            self.velocity = np.zeros_like(current_weights)

        self.velocity = self.momentum * self.velocity - self.lr * current_grads
        return current_weights + self.velocity


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.lr = learning_rate
        self.beta1 = mu
        self.beta2 = rho
        self.epsilon = 1e-8
        self.v = None
        self.r = None
        self.k = 0

    def calculate_update(self, current_weights, current_grads):
        if self.regularizer is not None:
            current_weights = current_weights - self.lr * self.regularizer.calculate_gradient(current_weights)

        if self.v is None:
            self.v = np.zeros_like(current_weights)
            self.r = np.zeros_like(current_weights)

        self.k += 1

        self.v = self.beta1 * self.v + (1 - self.beta1) * current_grads
        self.r = self.beta2 * self.r + (1 - self.beta2) * (current_grads ** 2)

        v_hat = self.v / (1 - self.beta1 ** self.k)
        r_hat = self.r / (1 - self.beta2 ** self.k)

        return current_weights - self.lr * v_hat / (np.sqrt(r_hat) + self.epsilon)