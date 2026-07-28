import copy
import numpy as np
from Layers import Base


class RNN(Base.BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.trainable = True

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.memorize = False

        self.weights = np.random.uniform(
            0, 1, (input_size + hidden_size + 1, hidden_size)
        )

        self.output_weights = np.random.uniform(
            0, 1, (hidden_size + 1, output_size)
        )

        self.gradient_weights = None
        self.gradient_output_weights = None

        self.hidden_state = np.zeros((1, hidden_size))

        self._optimizer = None
        self._output_optimizer = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self._output_optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(
            self.weights.shape,
            self.input_size + self.hidden_size,
            self.hidden_size
        )

        self.output_weights = weights_initializer.initialize(
            self.output_weights.shape,
            self.hidden_size,
            self.output_size
        )

        self.weights[-1, :] = bias_initializer.initialize(
            (self.hidden_size,),
            1,
            self.hidden_size
        )

        self.output_weights[-1, :] = bias_initializer.initialize(
            (self.output_size,),
            1,
            self.output_size
        )

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        time_steps = input_tensor.shape[0]

        self.hidden_states = np.zeros((time_steps + 1, self.hidden_size))
        self.outputs = np.zeros((time_steps, self.output_size))
        self.combined_inputs = []
        self.output_inputs = []

        if self.memorize:
            self.hidden_states[0] = self.hidden_state
        else:
            self.hidden_states[0] = np.zeros(self.hidden_size)

        for t in range(time_steps):
            x_t = input_tensor[t].reshape(1, -1)
            h_prev = self.hidden_states[t].reshape(1, -1)

            bias = np.ones((1, 1))
            combined = np.concatenate((x_t, h_prev, bias), axis=1)
            self.combined_inputs.append(combined)

            hidden_linear = np.dot(combined, self.weights)
            h_t = np.tanh(hidden_linear)

            self.hidden_states[t + 1] = h_t

            output_input = np.concatenate((h_t, bias), axis=1)
            self.output_inputs.append(output_input)

            output_linear = np.dot(output_input, self.output_weights)
            self.outputs[t] = self.sigmoid(output_linear)

        self.hidden_state = self.hidden_states[-1].reshape(1, -1)

        return self.outputs

    def backward(self, error_tensor):
        time_steps = error_tensor.shape[0]

        gradient_input = np.zeros_like(self.input_tensor)

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_output_weights = np.zeros_like(self.output_weights)

        hidden_error_next = np.zeros((1, self.hidden_size))

        for t in reversed(range(time_steps)):
            y_t = self.outputs[t].reshape(1, -1)
            error_y = error_tensor[t].reshape(1, -1)

            error_output_linear = error_y * y_t * (1 - y_t)

            output_input = self.output_inputs[t]
            self.gradient_output_weights += np.dot(
                output_input.T,
                error_output_linear
            )

            error_hidden_from_output = np.dot(
                error_output_linear,
                self.output_weights[:-1, :].T
            )

            error_hidden_total = error_hidden_from_output + hidden_error_next

            h_t = self.hidden_states[t + 1].reshape(1, -1)
            error_hidden_linear = error_hidden_total * (1 - h_t ** 2)

            combined = self.combined_inputs[t]
            self.gradient_weights += np.dot(
                combined.T,
                error_hidden_linear
            )

            error_combined = np.dot(
                error_hidden_linear,
                self.weights.T
            )

            gradient_input[t] = error_combined[:, :self.input_size]

            hidden_error_next = error_combined[
                :,
                self.input_size:self.input_size + self.hidden_size
            ]

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(
                self.weights,
                self.gradient_weights
            )

            self.output_weights = self._output_optimizer.calculate_update(
                self.output_weights,
                self.gradient_output_weights
            )

        return gradient_input