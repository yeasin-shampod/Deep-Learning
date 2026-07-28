import copy
import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss_history = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = False

        self.weights_initializer = copy.deepcopy(weights_initializer)
        self.bias_initializer = copy.deepcopy(bias_initializer)

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, testing_phase):
        self._phase = testing_phase
        for layer in self.layers:
            layer.testing_phase = testing_phase

    def forward(self):
        input_batch, self.current_labels = copy.deepcopy(self.data_layer.next())
        activation = input_batch

        for layer in self.layers:
            activation = layer.forward(activation)

        loss = self.loss_layer.forward(activation, copy.deepcopy(self.current_labels))

        for layer in self.layers:
            if layer.trainable and layer.optimizer is not None:
                if layer.optimizer.regularizer is not None:
                    loss += layer.optimizer.regularizer.norm(layer.weights)

        self.loss = loss
        return loss

    def backward(self):
        error = self.loss_layer.backward(copy.deepcopy(self.current_labels))

        for layer in reversed(self.layers):
            error = layer.backward(error)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        self.phase = False

        for _ in range(iterations):
            loss = self.forward()
            self.loss_history.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.phase = True

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor