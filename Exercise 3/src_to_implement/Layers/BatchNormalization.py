import copy
import numpy as np

from Layers import Base
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()

        self.trainable = True
        self.channels = channels

        self.weights = np.ones(channels)   # gamma
        self.bias = np.zeros(channels)     # beta

        self.gradient_weights = None
        self.gradient_bias = None

        self._optimizer = None

        self.epsilon = 1e-10
        self.momentum = 0.8

        self.mean = None
        self.variance = None
        self.moving_mean = None
        self.moving_variance = None

        self.input_tensor = None
        self.normalized_input = None
        self.input_shape = None
        self.is_convolutional = False

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.input_shape = tensor.shape
            batch_size, channels, height, width = tensor.shape
            return tensor.transpose(0, 2, 3, 1).reshape(-1, channels)

        if tensor.ndim == 2 and self.is_convolutional:
            batch_size, channels, height, width = self.input_shape
            return tensor.reshape(batch_size, height, width, channels).transpose(0, 3, 1, 2)

        return tensor

    def forward(self, input_tensor):
        self.is_convolutional = input_tensor.ndim == 4

        if self.is_convolutional:
            input_vector = self.reformat(input_tensor)
        else:
            input_vector = input_tensor

        if self.testing_phase:
            normalized = (input_vector - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)
        else:
            self.input_tensor = input_vector

            self.mean = np.mean(input_vector, axis=0)
            self.variance = np.var(input_vector, axis=0)

            if self.moving_mean is None:
                self.moving_mean = self.mean.copy()
                self.moving_variance = self.variance.copy()
            else:
                self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
                self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.variance

            normalized = (input_vector - self.mean) / np.sqrt(self.variance + self.epsilon)
            self.normalized_input = normalized

        output_vector = self.weights * normalized + self.bias

        if self.is_convolutional:
            return self.reformat(output_vector)

        return output_vector

    def backward(self, error_tensor):
        if self.is_convolutional:
            error_vector = self.reformat(error_tensor)
        else:
            error_vector = error_tensor

        self.gradient_weights = np.sum(error_vector * self.normalized_input, axis=0)
        self.gradient_bias = np.sum(error_vector, axis=0)

        input_gradient = compute_bn_gradients(
            error_vector,
            self.input_tensor,
            self.weights,
            self.mean,
            self.variance
        )

        if self._optimizer is not None:
            self.weights = self._optimizer.weight.calculate_update(
                self.weights,
                self.gradient_weights
            )
            self.bias = self._optimizer.bias.calculate_update(
                self.bias,
                self.gradient_bias
            )

        if self.is_convolutional:
            return self.reformat(input_gradient)

        return input_gradient