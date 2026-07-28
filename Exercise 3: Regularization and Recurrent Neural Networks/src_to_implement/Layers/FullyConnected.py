import numpy as np
from Layers import Base
import copy


class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size

        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))

        self.gradient_weights = None
        self._optimizer = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = copy.deepcopy(opt)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias_column = np.ones((input_tensor.shape[0], 1))
        input_with_bias = np.concatenate((input_tensor, bias_column), axis=1)
        return np.dot(input_with_bias, self.weights)

    def backward(self, error_tensor):
        bias_column = np.ones((self.input_tensor.shape[0], 1))
        input_with_bias = np.concatenate((self.input_tensor, bias_column), axis=1)

        self.gradient_weights = np.dot(input_with_bias.T, error_tensor)

        input_gradient = np.dot(error_tensor, self.weights[:-1, :].T)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(
                self.weights,
                self.gradient_weights
            )

        return input_gradient

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            self.weights.shape,
            self.input_size,
            self.output_size
        )

        self.weights[-1, :] = bias_initializer.initialize(
            (self.output_size,),
            1,
            self.output_size
        )