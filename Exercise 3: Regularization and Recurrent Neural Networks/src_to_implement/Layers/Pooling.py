import numpy as np
from Layers import Base

#Initialize the pooling layer with stride and pooling shape.
class Pooling(Base.BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape         # (stride_height, stride_width)
        self.pooling_shape = pooling_shape       # (pool_height, pool_width)

    #Apply 2D max pooling using 'valid' padding (no padding).

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape    # Store input shape for backward
        batch_size, channels, height, width = input_tensor.shape
        pool_h, pool_w = self.pooling_shape
        stride_h, stride_w = self.stride_shape

        # Calculate output dimensions based on valid padding
        out_h = int(np.ceil((height - pool_h + 1) / stride_h))
        out_w = int(np.ceil((width - pool_w + 1) / stride_w))

        # Initialize output and position tracking tensors
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices_x = np.zeros((batch_size, channels, out_h, out_w), dtype=int)
        self.max_indices_y = np.zeros((batch_size, channels, out_h, out_w), dtype=int)

        for out_row, i in enumerate(range(0, height - pool_h + 1, stride_h)):
            for out_col, j in enumerate(range(0, width - pool_w + 1, stride_w)):
                # Extract patch and flatten pooling region
                patch = input_tensor[:, :, i:i+pool_h, j:j+pool_w].reshape(batch_size, channels, -1)

                # Max index in flattened pooling window
                max_flat_idx = np.argmax(patch, axis=2)

                # Convert flat index to 2D coordinates within pooling window
                max_x = max_flat_idx // pool_w
                max_y = max_flat_idx % pool_w

                # Save max positions
                self.max_indices_x[:, :, out_row, out_col] = max_x
                self.max_indices_y[:, :, out_row, out_col] = max_y

                # Collect max values
                output[:, :, out_row, out_col] = np.choose(max_flat_idx, np.moveaxis(patch, 2, 0))

        return output

    #Propagates error only to positions that had maximum values during the forward pass.
    def backward(self, error_tensor):

        batch_size, channels, height, width = self.input_shape
        stride_h, stride_w = self.stride_shape
        pool_h, pool_w = self.pooling_shape

        out_h, out_w = self.max_indices_x.shape[2], self.max_indices_x.shape[3]
        grad_input = np.zeros((batch_size, channels, height, width))

        for n in range(batch_size):
            for c in range(channels):
                for out_row in range(out_h):
                    for out_col in range(out_w):
                        # Find max position in original input
                        max_i = out_row * stride_h + self.max_indices_x[n, c, out_row, out_col]
                        max_j = out_col * stride_w + self.max_indices_y[n, c, out_row, out_col]

                        # Assign gradient from error tensor to that max position
                        grad_input[n, c, max_i, max_j] += error_tensor[n, c, out_row, out_col]

        return grad_input
