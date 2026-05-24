import copy

#NeuralNetwork class definition with optimizer parameter in the constructor

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer #base optimizer
        self.loss = [] #loss per iteration
        self.layers = [] #list of layers information
        self.data_layer = None 
        self.loss_layer = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next() #input and label retriev

        #pass input through all layers except loss layer
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        #final forward through loss layer with prediction + loss computation   
        loss_value = self.loss_layer.forward(input_tensor, label_tensor)

        #input + label for backward
        self.label_tensor = label_tensor
        self.prediction_tensor = input_tensor

        return loss_value
    
    #backpropagation
    def backward(self):
        #loss calculation
        error_tensor = self.loss_layer.backward(self.label_tensor)

        #gradients passing through the layers
        for layer in reversed (self.layers):
            error_tensor = layer.backward(error_tensor)

    #trainable layers according to the definition
    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            loss_value = self.forward()
            self.loss.append(loss_value)
            self.backward()

    def test(self, input_tensor):
        # Propagate input through all layers (excluding data/loss)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor