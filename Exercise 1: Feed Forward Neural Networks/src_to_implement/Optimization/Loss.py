import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, input_tensor, label_tensor):
        #epsilon to avoid log(0)
        eps = np.finfo(float).eps
        clipped_input = np.clip(input_tensor, eps, 1.0)
        self.input = clipped_input  # store for backward

        #cross-entropy loss
        log_preds = np.log(clipped_input)
        loss = -np.sum(label_tensor * log_preds)
        return loss

    def backward(self, label_tensor):
        loss = -label_tensor / self.input
        return loss